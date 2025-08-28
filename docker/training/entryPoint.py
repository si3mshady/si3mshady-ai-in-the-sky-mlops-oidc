#!/usr/bin/env python3
import os
import sys
import json
import random
import traceback
from pathlib import Path
from typing import List, Tuple, Optional
import boto3
import numpy as np
import soundfile as sf
import librosa
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from models.audio_cnn import AudioCNN

# -------------------------
# Logging
# -------------------------
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("train")

def log_exc(context: str, exc: BaseException):
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    log.error(f"[EXCEPTION] {context} :: {exc.__class__.__name__}: {exc}\n{tb}", extra={"context": context})

# -------------------------
# Config
# -------------------------
BUCKET     = os.environ.get("S3_TRAIN_BUCKET", "urbansound-mlops-56423506")
PREFIX     = os.environ.get("S3_TRAIN_PREFIX", "training/").rstrip("/") + "/"
LOCAL_DIR  = Path("/opt/ml/input/data/training")
MODEL_DIR  = Path(os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
EPOCHS     = int(os.environ.get("EPOCHS", "10"))  # Increased epochs
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "16"))  # Reduced batch size
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "2"))

# FIXED PARAMETERS
TARGET_SECONDS = float(os.environ.get("TARGET_SECONDS", "3.0"))  # Longer windows
COVERAGE_STRIDE_SEC = 1.0  # Less overlap

# DISABLE PROBLEMATIC FEATURES
AUG_ENABLE = False  # NO AUGMENTATION
USE_CLASS_WEIGHTS = False  # NO CLASS WEIGHTS

LOCAL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------
# Audio params
# -------------------------
AUDIO_EXTS = (".wav", ".mp3", ".flac", ".aif", ".aiff", ".m4a", ".ogg")
SR = 22050
TARGET_SAMPLES = int(SR * TARGET_SECONDS)

def print_env_banner():
    log.info("========== FIXED CONFIGURATION ==========")
    log.info(f"TARGET_SECONDS={TARGET_SECONDS} (increased from 2.0)")
    log.info(f"BATCH_SIZE={BATCH_SIZE} (reduced from 32)")
    log.info(f"EPOCHS={EPOCHS} (increased)")
    log.info(f"AUG_ENABLE={AUG_ENABLE} (DISABLED)")
    log.info(f"USE_CLASS_WEIGHTS={USE_CLASS_WEIGHTS} (DISABLED)")
    log.info("Using FIXED global normalization")
    log.info("==========================================")

def download_s3_prefix(bucket: str, prefix: str, local_dir: Path):
    log.info(f"Starting S3 download: s3://{bucket}/{prefix} → {local_dir}")
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    total = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []) or []:
            key = obj["Key"]
            rel = key[len(prefix):]
            if not rel:
                continue
            dest = local_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            try:
                s3.download_file(bucket, key, str(dest))
                total += 1
                if total <= 20 or total % 200 == 0:
                    log.debug(f"Downloaded: {dest}")
            except Exception as e:
                log_exc(f"S3 download failed for {key}", e)
    log.info(f"Downloaded {total} files")

def safe_soundfile_read(path: Path) -> Optional[tuple]:
    try:
        y, file_sr = sf.read(str(path), always_2d=False)
        return y, file_sr
    except Exception:
        try:
            y, file_sr = librosa.load(str(path), sr=None, mono=False)
            return y, file_sr
        except Exception:
            return None

def load_audio_mono(path: Path, sr: int = SR) -> np.ndarray:
    res = safe_soundfile_read(path)
    if res is None:
        raise RuntimeError(f"Failed to decode audio: {path}")
    y, file_sr = res
    y = np.asarray(y)
    if y.ndim > 1:
        y = y.mean(axis=0)
    if file_sr != sr:
        y = librosa.resample(y, orig_sr=file_sr, target_sr=sr)
    return y.astype(np.float32)

# FIXED NORMALIZATION - Use consistent global values
def to_logmel(y: np.ndarray) -> np.ndarray:
    if len(y) < TARGET_SAMPLES:
        y = np.pad(y, (0, TARGET_SAMPLES - len(y)))
    else:
        y = y[:TARGET_SAMPLES]
    
    S = librosa.feature.melspectrogram(
        y=y, sr=SR, n_mels=128, n_fft=2048, hop_length=512, power=2.0
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    
    # FIXED: Use consistent global normalization values
    # Based on typical audio mel-spectrogram ranges
    GLOBAL_MIN = -80.0  # Typical minimum dB value
    GLOBAL_MAX = 0.0    # Typical maximum dB value
    
    S_norm = (S_db - GLOBAL_MIN) / (GLOBAL_MAX - GLOBAL_MIN)
    S_norm = np.clip(S_norm, 0.0, 1.0)  # Ensure [0,1] range
    
    return S_norm.astype(np.float32)

class CoverageWindowDataset(Dataset):
    def __init__(self, root: Path, classes: List[str],
                 seconds: float = TARGET_SECONDS,
                 stride_sec: float = COVERAGE_STRIDE_SEC,
                 split: str = "train"):
        assert split in ("train", "val")
        self.root = root
        self.classes = classes
        self.seconds = float(seconds)
        self.stride = float(stride_sec)
        self.split = split
        self.sr = SR
        self.files: List[Tuple[Path, int, float]] = []
        
        class_to_idx = {c: i for i, c in enumerate(classes)}
        
        # Filter out silent files
        for c in classes:
            class_dir = root / c
            file_count = 0
            for ext in AUDIO_EXTS:
                for f in class_dir.rglob(f"*{ext}"):
                    # Quick energy check to filter silent files
                    try:
                        y_test = load_audio_mono(f)
                        energy = np.mean(y_test ** 2)
                        if energy > 1e-6:  # Not silent
                            dur = len(y_test) / self.sr
                            if dur > 0.5:  # At least 0.5 seconds
                                self.files.append((f, class_to_idx[c], dur))
                                file_count += 1
                    except Exception as e:
                        log.warning(f"Skipping corrupted file {f}: {e}")
            
            log.info(f"[{split}] {c}: {file_count} valid files")
        
        if not self.files:
            raise RuntimeError(f"No valid audio found under {root}")
        
        # Build window index
        self.index: List[Tuple[int, float]] = []
        for i, (_, _, dur) in enumerate(self.files):
            if dur <= self.seconds:
                self.index.append((i, 0.0))
            else:
                start = 0.0
                while start <= (dur - self.seconds + 1e-6):
                    self.index.append((i, start))
                    start += self.stride
        
        # Log final distribution
        class_windows = {}
        for fidx, _ in self.index:
            label_idx = self.files[fidx][1]
            class_name = self.classes[label_idx]
            class_windows[class_name] = class_windows.get(class_name, 0) + 1
        
        log.info(f"[{split}] Final window distribution: {class_windows}")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx: int):
        fidx, start_sec = self.index[idx]
        path, label, dur = self.files[fidx]
        
        try:
            y = load_audio_mono(path)
            
            # Extract window
            start_sample = int(start_sec * self.sr)
            end_sample = start_sample + TARGET_SAMPLES
            
            if end_sample <= len(y):
                y_window = y[start_sample:end_sample]
            else:
                y_window = y[start_sample:]
                y_window = np.pad(y_window, (0, TARGET_SAMPLES - len(y_window)))
            
            mel = to_logmel(y_window)  # (128, T)
            m = mel[None]  # (1, 128, T)
            
            return torch.from_numpy(m), torch.tensor(label, dtype=torch.long)
        except Exception as e:
            log_exc(f"Failed to load {path}", e)
            return None

def collate_skip_bad(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    Xs, Ys = zip(*batch)
    X = torch.stack(Xs, dim=0)
    Y = torch.stack(Ys, dim=0)
    return X, Y

def train_one_epoch(model, loader, optim, device, epoch_idx: int, loss_fn):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    
    # Track predictions
    pred_counts = {i: 0 for i in range(len(loader.dataset.classes))}
    
    for step, batch in enumerate(loader):
        if batch is None:
            continue
            
        x, y = batch
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
        
        # Track predictions
        for pred in preds.cpu().numpy():
            pred_counts[pred] += 1
        
        if step % 10 == 0:
            log.info(f"[E{epoch_idx} S{step}] loss={loss.item():.4f}")
    
    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)
    
    # Log prediction distribution
    pred_dist = {loader.dataset.classes[i]: count for i, count in pred_counts.items()}
    log.info(f"[E{epoch_idx}] Train: loss={avg_loss:.4f} acc={acc:.4f}")
    log.info(f"[E{epoch_idx}] Train predictions: {pred_dist}")
    
    return avg_loss, acc

@torch.no_grad()
def evaluate(model, loader, device, epoch_idx: int, loss_fn):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    
    pred_counts = {i: 0 for i in range(len(loader.dataset.classes))}
    
    for batch in loader:
        if batch is None:
            continue
            
        x, y = batch
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = loss_fn(logits, y)
        
        total_loss += loss.item() * y.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
        
        for pred in preds.cpu().numpy():
            pred_counts[pred] += 1
    
    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)
    
    pred_dist = {loader.dataset.classes[i]: count for i, count in pred_counts.items()}
    log.info(f"[E{epoch_idx}] Val: loss={avg_loss:.4f} acc={acc:.4f}")
    log.info(f"[E{epoch_idx}] Val predictions: {pred_dist}")
    
    return avg_loss, acc

def main():
    print_env_banner()
    
    # Download data
    try:
        download_s3_prefix(BUCKET, PREFIX, LOCAL_DIR)
    except Exception as e:
        log_exc("download_s3_prefix", e)
        raise

    classes = [ "alarms", "domestic", "forced_entry", "gunfire", "police", "siren"]

    # Create datasets
    train_ds = CoverageWindowDataset(LOCAL_DIR, classes, split="train")
    val_ds = CoverageWindowDataset(LOCAL_DIR, classes, split="val")

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, collate_fn=collate_skip_bad
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, collate_fn=collate_skip_bad
    )

    # Model setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AudioCNN(n_classes=len(classes)).to(device)
    
    # Simple loss (no class weights)
    loss_fn = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=5e-4)  # Lower learning rate

    # Training loop
    best_acc, best_path = 0.0, MODEL_DIR / "model.pt"
    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optim, device, epoch, loss_fn)
        va_loss, va_acc = evaluate(model, val_loader, device, epoch, loss_fn)

        if va_acc > best_acc:
            best_acc = va_acc
            torch.save(model.state_dict(), best_path)
            log.info(f"✅ New best model: {best_acc:.4f}")

    # Save classes
    with open(MODEL_DIR / "classes.json", "w") as f:
        json.dump(classes, f)
    
    # Save normalization constants for inference
    with open(MODEL_DIR / "normalization.json", "w") as f:
        json.dump({"global_min": -80.0, "global_max": 0.0}, f)

    log.info(f"✅ Training complete. Best accuracy: {best_acc:.4f}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log_exc("FATAL", e)
        raise

