import os, json, tarfile, zipfile, random, math, time
from pathlib import Path
from typing import List, Tuple
import numpy as np
import soundfile as sf
import librosa
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from models.audio_cnn import AudioCNN

# -------------------------
# Helper: find and prepare data
# -------------------------
def _maybe_extract(chn: Path) -> Path:
    """If channel has a single .tar.gz or .zip, extract and return the extracted root."""
    files = list(chn.glob("*"))
    if len(files) == 1 and files[0].is_file():
        f = files[0]
        outdir = chn / "_extracted"
        outdir.mkdir(exist_ok=True)
        if f.suffixes[-2:] == [".tar", ".gz"] or f.suffix == ".tgz":
            with tarfile.open(f, "r:gz") as t:
                t.extractall(outdir)
            return outdir
        if f.suffix == ".zip":
            with zipfile.ZipFile(f, "r") as z:
                z.extractall(outdir)
            return outdir
    return chn

def _find_audio_root(root: Path) -> Path:
    """
    Try to locate the folder that contains class subfolders.
    Expected patterns:
      UrbanSound/data/<class>/*.wav|*.mp3
      <anything>/<class>/*.wav|*.mp3
    """
    candidates = [root, root / "UrbanSound" / "data", root / "data"]
    for c in candidates:
        if c.exists():
            # Does it have subfolders that contain at least one audio file?
            subdirs = [d for d in c.glob("*") if d.is_dir()]
            for d in subdirs:
                if any(d.glob("*.wav")) or any(d.glob("*.mp3")):
                    return c
    return root  # fallback; dataset class will error if no files

# -------------------------
# Dataset
# -------------------------
AUDIO_EXTS = (".wav", ".mp3")
SR = 22050
N_MELS = 128
WIN_LENGTH = 2048
HOP_LENGTH = 512
TARGET_SECONDS = 2.0
TARGET_SAMPLES = int(SR * TARGET_SECONDS)

def load_audio_mono(path: Path, sr: int = SR) -> np.ndarray:
    y, file_sr = sf.read(str(path), always_2d=False)
    if isinstance(y, np.ndarray) and y.ndim > 1:
        y = y.mean(axis=1)
    if file_sr != sr:
        y = librosa.resample(y, orig_sr=file_sr, target_sr=sr)
    return y.astype(np.float32)

def to_logmel(y: np.ndarray) -> np.ndarray:
    # pad/crop to fixed length for batching
    if len(y) < TARGET_SAMPLES:
        pad = TARGET_SAMPLES - len(y)
        y = np.pad(y, (0, pad))
    else:
        y = y[:TARGET_SAMPLES]
    S = librosa.feature.melspectrogram(
        y=y, sr=SR, n_mels=N_MELS, n_fft=WIN_LENGTH, hop_length=HOP_LENGTH, power=2.0
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    # normalize roughly to 0-1
    S_db = (S_db - S_db.min()) / max(1e-8, (S_db.max() - S_db.min()))
    return S_db.astype(np.float32)  # (128, T)

class UrbanSoundDataset(Dataset):
    def __init__(self, root: Path, classes: List[str] = None):
        self.files: List[Tuple[Path, int]] = []
        self.class_to_idx = {}
        if classes is None:
            # infer classes by subfolders
            classes = sorted([d.name for d in root.glob("*") if d.is_dir()])
        self.classes = classes
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        for c in self.classes:
            for ext in AUDIO_EXTS:
                for f in (root / c).rglob(f"*{ext}"):
                    self.files.append((f, self.class_to_idx[c]))
        if not self.files:
            raise RuntimeError(f"No audio files found under {root}")
        random.shuffle(self.files)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        path, label = self.files[idx]
        y = load_audio_mono(path)
        m = to_logmel(y)             # (128, T)
        m = np.expand_dims(m, 0)     # (1, 128, T)
        x = torch.from_numpy(m)      # float32
        y = torch.tensor(label, dtype=torch.long)
        return x, y

def split_dataset(ds: UrbanSoundDataset, val_ratio=0.2):
    n = len(ds)
    n_val = int(n * val_ratio)
    indices = list(range(n))
    random.shuffle(indices)
    val_idx = set(indices[:n_val])
    idx_train = [i for i in indices if i not in val_idx]
    idx_val = [i for i in indices if i in val_idx]
    return torch.utils.data.Subset(ds, idx_train), torch.utils.data.Subset(ds, idx_val)

# -------------------------
# Training loop
# -------------------------
def train_one_epoch(model, loader, optim, device):
    model.train()
    ce = nn.CrossEntropyLoss()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optim.zero_grad()
        logits = model(x)
        loss = ce(logits, y)
        loss.backward()
        optim.step()
        loss_sum += float(loss) * y.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return loss_sum / total, correct / total

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ce = nn.CrossEntropyLoss()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = ce(logits, y)
        loss_sum += float(loss) * y.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return loss_sum / total, correct / total

def main():
    random.seed(42); np.random.seed(42); torch.manual_seed(42)

    # SageMaker I/O
    chn_train = Path(os.environ.get("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training"))
    model_dir = Path(os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    model_dir.mkdir(parents=True, exist_ok=True)

    # Extract if archive; then find the actual audio/class root
    chn = _maybe_extract(chn_train)
    audio_root = _find_audio_root(chn)

    # If your dataset uses the UrbanSound8K labels, set them here:
    default_classes = [
        "air_conditioner","car_horn","children_playing","dog_bark","drilling",
        "engine_idling","gun_shot","jackhammer","siren","street_music"
    ]
    ds = UrbanSoundDataset(audio_root, classes=default_classes)
    tr, va = split_dataset(ds, val_ratio=0.2)

    train_loader = DataLoader(tr, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(va, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AudioCNN(n_classes=len(default_classes)).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_acc, best_path = 0.0, model_dir / "model.pt"
    epochs = int(os.environ.get("EPOCHS", "5"))  # keep fast by default
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optim, device)
        va_loss, va_acc = evaluate(model, val_loader, device)
        print(f"[{epoch}/{epochs}] train loss {tr_loss:.4f} acc {tr_acc:.3f} | val loss {va_loss:.4f} acc {va_acc:.3f}")
        if va_acc > best_acc:
            best_acc = va_acc
            torch.save(model.state_dict(), best_path)

    # Save artifacts (SageMaker will tar /opt/ml/model automatically)
    with open(model_dir / "classes.json", "w") as f:
        json.dump(default_classes, f)
    print(f"Saved best model to: {best_path} with val_acc={best_acc:.3f}")

if __name__ == "__main__":
    main()

