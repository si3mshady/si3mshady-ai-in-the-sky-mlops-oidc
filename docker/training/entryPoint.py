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
from models.audio_cnn import AudioCNN  # unchanged
# -------------------------
# Logging (very verbose)
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
# Config via env - FIXED: No duplicate lines, reasonable settings
# -------------------------
BUCKET     = os.environ.get("S3_TRAIN_BUCKET", "urbansound-mlops-56423506")
PREFIX     = os.environ.get("S3_TRAIN_PREFIX", "training/").rstrip("/") + "/"
LOCAL_DIR  = Path("/opt/ml/input/data/training")
MODEL_DIR  = Path(os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
EPOCHS     = int(os.environ.get("EPOCHS", "10"))      # REASONABLE: Not too long
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "8"))   # REASONABLE: Not too small
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "2"))
MAX_BAD_BATCHES_TO_LOG = int(os.environ.get("MAX_BAD_BATCHES_TO_LOG", "50"))
MAX_BAD_FILES_BEFORE_WARN = int(os.environ.get("MAX_BAD_FILES_BEFORE_WARN", "50"))
# coverage / augmentation knobs
TARGET_SECONDS = float(os.environ.get("TARGET_SECONDS", "2.0"))
COVERAGE_STRIDE_SEC = float(os.environ.get("COVERAGE_STRIDE_SEC", "0.5"))
COVERAGE_STRIDE_SEC_VAL = float(os.environ.get("COVERAGE_STRIDE_SEC_VAL", "0.5"))
AUG_ENABLE = os.environ.get("AUG_ENABLE", "1") == "1"
AUG_FREQ_MASKS = int(os.environ.get("AUG_FREQ_MASKS", "1"))
AUG_TIME_MASKS = int(os.environ.get("AUG_TIME_MASKS", "1"))
AUG_FREQ_PCT   = float(os.environ.get("AUG_FREQ_PCT", "0.15"))
AUG_TIME_PCT   = float(os.environ.get("AUG_TIME_PCT", "0.15"))
LOCAL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
BAD_FILE_LOG = MODEL_DIR / "bad_audio_files.txt"
# -------------------------
# Audio params
# -------------------------
AUDIO_EXTS      = (".wav", ".mp3", ".flac", ".aif", ".aiff", ".m4a", ".ogg")
SR              = 22050
TARGET_SAMPLES  = int(SR * TARGET_SECONDS)
# -------------------------
# Utils
# -------------------------
def print_env_banner():
    try:
        import pkg_resources
        pkgs = {p.key: p.version for p in pkg_resources.working_set}
    except Exception:
        pkgs = {}
    log.info("========== ENV / VERSIONS ==========")
    log.info(f"Python: {sys.version.splitlines()}")
    log.info(f"PyTorch: {torch.__version__} | CUDA available: {torch.cuda.is_available()} | device_count={torch.cuda.device_count()}")
    log.info(f"NumPy: {np.__version__} | librosa: {librosa.__version__}")
    try:
        import soundfile as _sf
        log.info(f"soundfile: {_sf.__version__}")
    except Exception:
        log.info("soundfile: (unknown)")
    log.info(f"BUCKET={BUCKET}  PREFIX={PREFIX}  LOCAL_DIR={LOCAL_DIR}  MODEL_DIR={MODEL_DIR}")
    log.info(f"EPOCHS={EPOCHS}  BATCH_SIZE={BATCH_SIZE}  NUM_WORKERS={NUM_WORKERS}  TARGET_SECONDS={TARGET_SECONDS}")
    log.info(f"COVERAGE_STRIDE_SEC={COVERAGE_STRIDE_SEC}  COVERAGE_STRIDE_SEC_VAL={COVERAGE_STRIDE_SEC_VAL}")
    log.info(f"AUG_ENABLE={AUG_ENABLE} Fmasks={AUG_FREQ_MASKS} Fpct={AUG_FREQ_PCT} Tmasks={AUG_TIME_MASKS} Tpct={AUG_TIME_PCT}")
    log.info("Some installed packages: " + ", ".join(sorted([f"{k}={v}" for k,v in list(pkgs.items())[:30]])))
    log.info("====================================")

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
                    log.debug(f"Downloaded: s3://{bucket}/{key} -> {dest}")
            except Exception as e:
                log_exc(f"S3 download failed for {key}", e)
    log.info(f"Downloaded {total} files from s3://{bucket}/{prefix} → {local_dir}")

def safe_soundfile_read(path: Path) -> Optional[tuple]:
    """Try soundfile first; if that fails, fall back to librosa.load."""
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
    """Load audio, convert to mono, and resample to sr. Raises on unrecoverable failure."""
    res = safe_soundfile_read(path)
    if res is None:
        raise RuntimeError(f"Failed to decode audio (soundfile & librosa): {path}")
    y, file_sr = res
    y = np.asarray(y)
    if y.ndim > 1:
        y = y.mean(axis=0)
    if file_sr != sr:
        y = librosa.resample(y, orig_sr=file_sr, target_sr=sr)
    return y.astype(np.float32)

# FIXED: Global normalization function
def to_logmel(y: np.ndarray) -> np.ndarray:
    if len(y) < TARGET_SAMPLES:
        y = np.pad(y, (0, TARGET_SAMPLES - len(y)))
    else:
        y = y[:TARGET_SAMPLES]
    S = librosa.feature.melspectrogram(
        y=y, sr=SR, n_mels=128, n_fft=2048, hop_length=512, power=2.0
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    
    # FIXED: Global normalization instead of per-sample min-max
    GLOBAL_MIN = -80.0
    GLOBAL_MAX = 0.0
    S_norm = (S_db - GLOBAL_MIN) / (GLOBAL_MAX - GLOBAL_MIN)
    S_norm = np.clip(S_norm, 0.0, 1.0)
    
    return S_norm.astype(np.float32)

def spec_augment_np(mel: np.ndarray,
                    freq_masks: int, time_masks: int,
                    freq_pct: float, time_pct: float) -> np.ndarray:
    """mel: (128, T) in [0,1]; mask with zeros"""
    m = mel.copy()
    F, T = m.shape
    # frequency masks
    for _ in range(max(0, freq_masks)):
        w = max(1, int(round(F * max(0.0, min(1.0, freq_pct)))))
        f0 = random.randint(0, max(0, F - w))
        m[f0:f0 + w, :] = 0.0
    # time masks
    for _ in range(max(0, time_masks)):
        w = max(1, int(round(T * max(0.0, min(1.0, time_pct)))))
        t0 = random.randint(0, max(0, T - w))
        m[:, t0:t0 + w] = 0.0
    return m

# [REST OF YOUR CODE UNCHANGED - CoverageWindowDataset, collate functions, etc.]
class CoverageWindowDataset(Dataset):
    def __init__(self, root: Path, classes: List[str],
                 seconds: float = TARGET_SECONDS,
                 stride_sec: float = 0.5,
                 split: str = "train",
                 aug_enable: bool = True):
        assert split in ("train", "val")
        self.root = root
        self.seconds = float(seconds)
        self.stride = float(stride_sec)
        self.split = split
        self.sr = SR
        self.aug_enable = aug_enable
        self.files: List[Tuple[Path, int, float]] = []
        class_to_idx = {c: i for i, c in enumerate(classes)}
        for c in classes:
            class_dir = root / c
            for ext in AUDIO_EXTS:
                for f in class_dir.rglob(f"*{ext}"):
                    dur = self._safe_duration(f)
                    if dur and dur > 0:
                        self.files.append((f, class_to_idx[c], dur))
        if not self.files:
            raise RuntimeError(f"No audio found under {root}")
        
        self.index: List[Tuple[int, float]] = []
        for i, (_, _, dur) in enumerate(self.files):
            if dur <= self.seconds:
                self.index.append((i, 0.0))
            else:
                start = 0.0
                jitter = 0.0
                if self.split == "train":
                    jitter = min(self.stride * 0.25, max(0.0, dur - self.seconds))
                while start <= (dur - self.seconds + 1e-6):
                    s = start
                    if self.split == "train" and jitter > 0.0:
                        s = max(0.0, min(dur - self.seconds,
                                         s + random.uniform(-jitter, jitter)))
                    self.index.append((i, s))
                    start += self.stride
        log.info(f"CoverageWindowDataset[{split}] files={len(self.files)} "
                 f"win={self.seconds}s stride={self.stride}s total_windows={len(self.index)}")
    
    def _safe_duration(self, path: Path) -> Optional[float]:
        try:
            info = sf.info(str(path))
            return float(info.frames) / float(info.samplerate)
        except Exception:
            try:
                return librosa.get_duration(path=str(path))
            except Exception:
                return None
    
    def __len__(self):
        return len(self.index)
    
    def __getitem__(self, idx: int):
        fidx, start_sec = self.index[idx]
        path, label, dur = self.files[fidx]
        try:
            y, sr = librosa.load(str(path), sr=self.sr, mono=True,
                                 offset=float(max(0.0, start_sec)),
                                 duration=float(self.seconds))
            tgt = int(self.sr * self.seconds)
            if len(y) < tgt:
                y = np.pad(y, (0, tgt - len(y)))
            elif len(y) > tgt:
                y = y[:tgt]
            mel = to_logmel(y)
            if self.split == "train" and self.aug_enable:
                mel = spec_augment_np(
                    mel,
                    freq_masks=AUG_FREQ_MASKS,
                    time_masks=AUG_TIME_MASKS,
                    freq_pct=AUG_FREQ_PCT,
                    time_pct=AUG_TIME_PCT,
                )
            m = mel[None]
            return torch.from_numpy(m), torch.tensor(label, dtype=torch.long)
        except Exception as e:
            log_exc(f"coverage_window_load_failed {path}", e)
            return None

def collate_skip_bad(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    Xs, Ys = zip(*batch)
    try:
        X = torch.stack(Xs, dim=0)
    except Exception as e:
        log_exc("collate_stack_failed", e)
        return None
    Y = torch.stack(Ys, dim=0)
    return X, Y

def compute_class_weights_from_windows(ds: CoverageWindowDataset, n_classes: int) -> torch.Tensor:
    counts = np.zeros(n_classes, dtype=np.float64)
    for fidx, _ in ds.index:
        label = ds.files[fidx][13]
        counts[label] += 1.0
    counts[counts == 0.0] = 1.0
    inv = 1.0 / counts
    weights = inv / inv.sum() * n_classes
    w = torch.tensor(weights, dtype=torch.float32)
    log.info(f"class window counts: {counts.tolist()}  -> loss weights: {weights.tolist()}")
    return w

def train_one_epoch(model, loader, optim, device, epoch_idx: int, loss_fn):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    skipped_batches = 0
    bad_batches_logged = 0
    for step, batch in enumerate(loader):
        if batch is None:
            skipped_batches += 1
            if bad_batches_logged < MAX_BAD_BATCHES_TO_LOG:
                log.warning(f"[E{epoch_idx}] Skipping bad batch at step={step} (batch=None)")
                bad_batches_logged += 1
            continue
        try:
            x, y = batch
            x, y = x.to(device), y.to(device)
            optim.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optim.step()
            total_loss += loss.item() * y.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            if step % 50 == 0:  # Less frequent logging
                log.info(f"[E{epoch_idx} S{step}] loss={loss.item():.4f} bs={y.size(0)} total={total}")
        except Exception as e:
            log_exc(f"train_step_error (epoch={epoch_idx}, step={step})", e)
            skipped_batches += 1
    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)
    log.info(f"[E{epoch_idx}] train_done | loss={avg_loss:.4f} acc={acc:.4f} total={total} skipped={skipped_batches}")
    return avg_loss, acc, skipped_batches

@torch.no_grad()
def evaluate(model, loader, device, epoch_idx: int, loss_fn):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    skipped_batches = 0
    for step, batch in enumerate(loader):
        if batch is None:
            skipped_batches += 1
            continue
        try:
            x, y = batch
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            total_loss += loss.item() * y.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
        except Exception as e:
            log_exc(f"eval_step_error (epoch={epoch_idx}, step={step})", e)
            skipped_batches += 1
    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)
    log.info(f"[E{epoch_idx}] eval_done  | loss={avg_loss:.4f} acc={acc:.4f} total={total} skipped={skipped_batches}")
    return avg_loss, acc, skipped_batches

def main():
    print_env_banner()
    
    try:
        download_s3_prefix(BUCKET, PREFIX, LOCAL_DIR)
    except Exception as e:
        log_exc("download_s3_prefix", e)
        raise
    
    classes = [
        "tires", "glass_shatter"
    ]
    
    train_ds = CoverageWindowDataset(
        LOCAL_DIR, classes,
        seconds=TARGET_SECONDS,
        stride_sec=COVERAGE_STRIDE_SEC,
        split="train",
        aug_enable=AUG_ENABLE
    )
    val_ds = CoverageWindowDataset(
        LOCAL_DIR, classes,
        seconds=TARGET_SECONDS,
        stride_sec=COVERAGE_STRIDE_SEC_VAL,
        split="val",
        aug_enable=False
    )
    
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, collate_fn=collate_skip_bad, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, collate_fn=collate_skip_bad, pin_memory=True
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AudioCNN(n_classes=len(classes)).to(device)
    
    # CHANGED: Simple loss without class weights (test first)
    loss_fn = nn.CrossEntropyLoss()
    log.info("Using simple CrossEntropyLoss")
    
    # Lower learning rate
    optim = torch.optim.Adam(model.parameters(), lr=1e-4)
    log.info("Using learning rate 1e-4")
    
    best_acc, best_path = 0.0, MODEL_DIR / "model.pt"
    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc, tr_skipped = train_one_epoch(model, train_loader, optim, device, epoch, loss_fn)
        va_loss, va_acc, va_skipped = evaluate(model, val_loader, device, epoch, loss_fn)
        log.info(f"[{epoch}/{EPOCHS}] "
                 f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} skipped={tr_skipped} | "
                 f"val_loss={va_loss:.4f} val_acc={va_acc:.4f} skipped={va_skipped}")
        
        if va_acc > best_acc:
            best_acc = va_acc
            try:
                torch.save(model.state_dict(), best_path)
                log.info(f"New best model saved: acc={best_acc:.4f} -> {best_path}")
            except Exception as e:
                log_exc("save_model", e)
    
    try:
        with open(MODEL_DIR / "classes.json", "w") as f:
            json.dump(classes, f)
    except Exception as e:
        log_exc("save_classes_json", e)
    
    log.info(f"Training complete. Best val_acc={best_acc:.4f}. Model saved to {best_path}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log_exc("FATAL", e)
        raise
