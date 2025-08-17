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
# Configuration from env
# -------------------------
BUCKET     = os.environ.get("S3_TRAIN_BUCKET", "urbansound-mlops-56423506")
PREFIX     = os.environ.get("S3_TRAIN_PREFIX", "training/").rstrip("/") + "/"
LOCAL_DIR  = Path("/opt/ml/input/data/training")
LOCAL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR  = Path(os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
MODEL_DIR.mkdir(parents=True, exist_ok=True)
EPOCHS     = int(os.environ.get("EPOCHS", "5"))

# -------------------------
# Download from S3
# -------------------------
def download_s3_prefix(bucket: str, prefix: str, local_dir: Path):
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            rel = key[len(prefix):]
            if not rel:
                continue
            dest = local_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(bucket, key, str(dest))
    print(f"Downloaded s3://{bucket}/{prefix} → {local_dir}")

# -------------------------
# Audio processing
# -------------------------
AUDIO_EXTS      = (".wav", ".mp3", ".flac", ".aif", ".aiff")
SR              = 22050
TARGET_SECONDS  = 2.0
TARGET_SAMPLES  = int(SR * TARGET_SECONDS)

def load_audio_mono(path: Path, sr: int = SR) -> np.ndarray:
    """Load an audio file as mono and resample to fixed sample rate."""
    y, file_sr = sf.read(str(path), always_2d=False)
    if isinstance(y, np.ndarray) and y.ndim > 1:
        y = y.mean(axis=1)
    if file_sr != sr:
        # Corrected: use keyword args for current librosa API
        y = librosa.resample(y, orig_sr=file_sr, target_sr=sr)
    return y.astype(np.float32)

def to_logmel(y: np.ndarray) -> np.ndarray:
    """Convert waveform to normalized log-Mel spectrogram."""
    if len(y) < TARGET_SAMPLES:
        pad = TARGET_SAMPLES - len(y)
        y = np.pad(y, (0, pad))
    else:
        y = y[:TARGET_SAMPLES]
    S = librosa.feature.melspectrogram(
        y=y, sr=SR, n_mels=128, n_fft=2048, hop_length=512, power=2.0
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    S_norm = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-8)
    return S_norm.astype(np.float32)

# -------------------------
# Dataset
# -------------------------
class UrbanSoundDataset(Dataset):
    def __init__(self, root: Path, classes: List[str]):
        self.samples: List[Tuple[Path, int]] = []
        class_to_idx = {c: i for i, c in enumerate(classes)}
        for c in classes:
            for ext in AUDIO_EXTS:
                for f in (root / c).rglob(f"*{ext}"):
                    self.samples.append((f, class_to_idx[c]))
        if not self.samples:
            raise RuntimeError(f"No audio found under {root}")
        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        y = load_audio_mono(path)
        m = to_logmel(y)[None]     # shape (1, 128, T)
        return torch.from_numpy(m), torch.tensor(label, dtype=torch.long)

# -------------------------
# Training & Evaluation
# -------------------------
def train_one_epoch(model, loader, optim, device):
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optim.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optim.step()
        total_loss += loss.item() * y.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return total_loss / total, correct / total

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = loss_fn(logits, y)
        total_loss += loss.item() * y.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return total_loss / total, correct / total

# -------------------------
# Main
# -------------------------
def main():
    # 1) Download data
    download_s3_prefix(BUCKET, PREFIX, LOCAL_DIR)

    # 2) Prepare dataset
    classes = [
        "air_conditioner","car_horn","children_playing","dog_bark","drilling",
        "engine_idling","gun_shot","jackhammer","siren","street_music"
    ]
    ds = UrbanSoundDataset(LOCAL_DIR, classes)
    n = len(ds)
    train_size = int(n * 0.8)
    train_ds, val_ds = torch.utils.data.random_split(ds, [train_size, n-train_size])

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False, num_workers=2)

    # 3) Model setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = AudioCNN(n_classes=len(classes)).to(device)
    optim  = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 4) Train loop
    best_acc, best_path = 0.0, MODEL_DIR / "model.pt"
    for epoch in range(1, EPOCHS+1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optim, device)
        va_loss, va_acc = evaluate(model, val_loader, device)
        print(f"[{epoch}/{EPOCHS}] train_loss={tr_loss:.4f} train_acc={tr_acc:.3f} | val_loss={va_loss:.4f} val_acc={va_acc:.3f}")
        if va_acc > best_acc:
            best_acc = va_acc
            torch.save(model.state_dict(), best_path)

    # 5) Save classes
    with open(MODEL_DIR / "classes.json", "w") as f:
        json.dump(classes, f)

    print(f"Training complete. Best val_acc={best_acc:.3f}. Model saved to {best_path}")

if __name__ == "__main__":
    main()
