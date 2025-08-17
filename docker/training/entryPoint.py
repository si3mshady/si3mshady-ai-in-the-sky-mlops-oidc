import os
import json
import tarfile
import zipfile
import random
from pathlib import Path
from typing import List, Tuple

import boto3
import numpy as np
import soundfile as sf
import librosa
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from models.audio_cnn import AudioCNN

# -------------------------
# Configuration (hard-coded defaults)
# -------------------------
DEFAULT_BUCKET = "urbansound-mlops-56423506"
DEFAULT_PREFIX = "training/"

BUCKET = os.environ.get("S3_TRAIN_BUCKET", DEFAULT_BUCKET)
PREFIX = os.environ.get("S3_TRAIN_PREFIX", DEFAULT_PREFIX).rstrip("/") + "/"

LOCAL_DIR = Path("/opt/ml/input/data/training")
LOCAL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_DIR = Path(os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
MODEL_DIR.mkdir(parents=True, exist_ok=True)

EPOCHS = int(os.environ.get("EPOCHS", "5"))

# -------------------------
# Download from S3
# -------------------------
def download_s3_prefix(bucket: str, prefix: str, local_dir: Path):
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            rel_path = key[len(prefix):]
            if not rel_path:
                continue
            dest = local_dir / rel_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(bucket, key, str(dest))
    print(f"Downloaded s3://{bucket}/{prefix} → {local_dir}")

# -------------------------
# Data preparation helpers
# -------------------------
AUDIO_EXTS = (".wav", ".mp3", ".flac", ".aif", ".aiff")
SR = 22050
N_MELS = 128
WIN_LENGTH = 2048
HOP_LENGTH = 512
TARGET_SAMPLES = int(SR * 2.0)

def load_audio_mono(path: Path) -> np.ndarray:
    y, file_sr = sf.read(str(path), always_2d=False)
    if isinstance(y, np.ndarray) and y.ndim > 1:
        y = y.mean(axis=1)
    if file_sr != SR:
        # Updated to use keyword args for librosa.resample
        y = librosa.resample(y, orig_sr=file_sr, target_sr=SR)
    return y.astype(np.float32)

def to_logmel(y: np.ndarray) -> np.ndarray:
    if len(y) < TARGET_SAMPLES:
        y = np.pad(y, (0, TARGET_SAMPLES - len(y)))
    else:
        y = y[:TARGET_SAMPLES]
    S = librosa.feature.melspectrogram(
        y=y, sr=SR, n_mels=N_MELS,
        n_fft=WIN_LENGTH, hop_length=HOP_LENGTH, power=2.0
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    return ((S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-8)).astype(np.float32)

class UrbanSoundDataset(Dataset):
    def __init__(self, root: Path, classes: List[str]):
        self.samples: List[Tuple[Path, int]] = []
        class_to_idx = {c: i for i, c in enumerate(classes)}
        for c in classes:
            folder = root / c
            for ext in AUDIO_EXTS:
                for f in folder.rglob(f"*{ext}"):
                    self.samples.append((f, class_to_idx[c]))
        if not self.samples:
            raise RuntimeError(f"No audio found under {root}")
        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        y = load_audio_mono(path)
        m = to_logmel(y)[None]  # shape (1, 128, T)
        return torch.from_numpy(m), torch.tensor(label, dtype=torch.long)

# -------------------------
# Main training logic
# -------------------------
def main():
    # 1) Download data from S3
    download_s3_prefix(BUCKET, PREFIX, LOCAL_DIR)

    # 2) Prepare dataset
    classes = [
        "air_conditioner","car_horn","children_playing","dog_bark","drilling",
        "engine_idling","gun_shot","jackhammer","siren","street_music"
    ]
    ds = UrbanSoundDataset(LOCAL_DIR, classes)
    n = len(ds)
    split = int(n * 0.8)
    train_ds, val_ds = torch.utils.data.random_split(ds, [split, n - split])

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

    # 3) Model setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AudioCNN(n_classes=len(classes)).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    # 4) Training loop
    best_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optim.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optim.step()

        # Optional: validation and checkpointing omitted for brevity

    # 5) Save model artifact
    torch.save(model.state_dict(), MODEL_DIR / "model.pt")
    print(f"Training complete, model saved to {MODEL_DIR}/model.pt")

if __name__ == "__main__":
    main()
