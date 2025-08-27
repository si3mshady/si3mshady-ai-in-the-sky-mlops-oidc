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
# Config via env
# -------------------------
BUCKET     = os.environ.get("S3_TRAIN_BUCKET", "urbansound-mlops-56423506")
PREFIX     = os.environ.get("S3_TRAIN_PREFIX", "training/").rstrip("/") + "/"
LOCAL_DIR  = Path("/opt/ml/input/data/training")
MODEL_DIR  = Path(os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
EPOCHS     = int(os.environ.get("EPOCHS", "5"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "32"))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "2"))   # set 0 for maximal tracebacks
MAX_BAD_BATCHES_TO_LOG = int(os.environ.get("MAX_BAD_BATCHES_TO_LOG", "50"))
MAX_BAD_FILES_BEFORE_WARN = int(os.environ.get("MAX_BAD_FILES_BEFORE_WARN", "50"))

LOCAL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
BAD_FILE_LOG = MODEL_DIR / "bad_audio_files.txt"

# -------------------------
# Audio params
# -------------------------
AUDIO_EXTS      = (".wav", ".mp3", ".flac", ".aif", ".aiff", ".m4a", ".ogg")
SR              = 22050
TARGET_SECONDS  = float(os.environ.get("TARGET_SECONDS", "2.0"))
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
    log.info(f"Python: {sys.version.splitlines()[0]}")
    log.info(f"PyTorch: {torch.__version__} | CUDA available: {torch.cuda.is_available()} | device_count={torch.cuda.device_count()}")
    log.info(f"NumPy: {np.__version__} | librosa: {librosa.__version__}")
    try:
        import soundfile as _sf
        log.info(f"soundfile: {_sf.__version__}")
    except Exception:
        log.info("soundfile: (unknown)")
    log.info(f"BUCKET={BUCKET}  PREFIX={PREFIX}  LOCAL_DIR={LOCAL_DIR}  MODEL_DIR={MODEL_DIR}")
    log.info(f"EPOCHS={EPOCHS}  BATCH_SIZE={BATCH_SIZE}  NUM_WORKERS={NUM_WORKERS}  TARGET_SECONDS={TARGET_SECONDS}")
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
    except Exception as e_sf:
        # fallback to librosa (will use audioread/ffmpeg if available)
        try:
            y, file_sr = librosa.load(str(path), sr=None, mono=False)
            return y, file_sr
        except Exception as e_lib:
            # Return None to signal failure; caller will log
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
        # librosa 0.10 API with keywords
        y = librosa.resample(y, orig_sr=file_sr, target_sr=sr)
    return y.astype(np.float32)

def to_logmel(y: np.ndarray) -> np.ndarray:
    if len(y) < TARGET_SAMPLES:
        y = np.pad(y, (0, TARGET_SAMPLES - len(y)))
    else:
        y = y[:TARGET_SAMPLES]
    S = librosa.feature.melspectrogram(
        y=y, sr=SR, n_mels=128, n_fft=2048, hop_length=512, power=2.0
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    S_norm = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-8)
    return S_norm.astype(np.float32)

# -------------------------
# Dataset with robust error handling
# -------------------------
class UrbanSoundDataset(Dataset):
    def __init__(self, root: Path, classes: List[str]):
        self.samples: List[Tuple[Path, int]] = []
        self.bad_files: List[str] = []
        class_to_idx = {c: i for i, c in enumerate(classes)}

        for c in classes:
            class_dir = root / c
            for ext in AUDIO_EXTS:
                for f in class_dir.rglob(f"*{ext}"):
                    self.samples.append((f, class_to_idx[c]))

        if not self.samples:
            raise RuntimeError(f"No audio found under {root}. Checked classes={classes}")

        random.shuffle(self.samples)
        log.info(f"Dataset initialized: {len(self.samples)} files across {len(classes)} classes.")
        counts = {c: 0 for c in classes}
        for p, idx in self.samples:
            for c, i in class_to_idx.items():
                if i == idx:
                    counts[c] += 1
                    break
        log.info(f"Per-class counts: {counts}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        try:
            y = load_audio_mono(path)
            m = to_logmel(y)[None]  # (1, 128, T)
            return torch.from_numpy(m), torch.tensor(label, dtype=torch.long)
        except Exception as e:
            # Log once per bad file and append to list
            msg = f"[BAD_AUDIO] {path} :: {type(e).__name__}: {e}"
            log.warning(msg)
            self.bad_files.append(str(path))
            # Return sentinel; collate_fn will drop it
            return None

def collate_skip_bad(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        # None batch (all bad) -> signal caller to skip
        return None
    Xs, Ys = zip(*batch)
    try:
        X = torch.stack(Xs, dim=0)
    except Exception as e:
        # Shape mismatch shouldn't happen given fixed pipeline, but log and drop
        log_exc("collate_stack_failed", e)
        return None
    Y = torch.stack(Ys, dim=0)
    return X, Y

# -------------------------
# Train & Eval with try/except per-batch
# -------------------------
def train_one_epoch(model, loader, optim, device, epoch_idx: int):
    model.train()
    loss_fn = nn.CrossEntropyLoss()
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

            if step % 20 == 0:
                log.info(f"[E{epoch_idx} S{step}] loss={loss.item():.4f} bs={y.size(0)} accumulated_total={total}")
        except Exception as e:
            log_exc(f"train_step_error (epoch={epoch_idx}, step={step})", e)
            skipped_batches += 1

    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)
    log.info(f"[E{epoch_idx}] train_done | loss={avg_loss:.4f} acc={acc:.4f} total={total} skipped_batches={skipped_batches}")
    return avg_loss, acc, skipped_batches

@torch.no_grad()
def evaluate(model, loader, device, epoch_idx: int):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
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
    log.info(f"[E{epoch_idx}] eval_done  | loss={avg_loss:.4f} acc={acc:.4f} total={total} skipped_batches={skipped_batches}")
    return avg_loss, acc, skipped_batches

# -------------------------
# Main
# -------------------------
def main():
    print_env_banner()

    # 1) Download data
    try:
        download_s3_prefix(BUCKET, PREFIX, LOCAL_DIR)
    except Exception as e:
        log_exc("download_s3_prefix", e)
        raise

    # 2) Prepare dataset
    classes = [
        "air_conditioner","car_horn","children_playing","dog_bark","drilling",
        "engine_idling","gun_shot","jackhammer","siren","street_music","alarms","crowd",
        "domestic","gunfire","police","grinding"
    ]

    try:
        ds = UrbanSoundDataset(LOCAL_DIR, classes)
    except Exception as e:
        log_exc("dataset_init", e)
        raise

    n = len(ds)
    train_size = int(n * 0.8)
    val_size = n - train_size
    log.info(f"Splitting dataset: total={n} -> train={train_size}, val={val_size}")
    train_ds, val_ds = torch.utils.data.random_split(ds, [train_size, val_size])

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, collate_fn=collate_skip_bad, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, collate_fn=collate_skip_bad, pin_memory=True
    )

    # 3) Model setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = AudioCNN(n_classes=len(classes)).to(device)
    optim  = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 4) Train loop (robust + chatty)
    best_acc, best_path = 0.0, MODEL_DIR / "model.pt"
    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc, tr_skipped = train_one_epoch(model, train_loader, optim, device, epoch)
        va_loss, va_acc, va_skipped = evaluate(model, val_loader, device, epoch)

        log.info(f"[{epoch}/{EPOCHS}] "
                 f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} skipped={tr_skipped} | "
                 f"val_loss={va_loss:.4f} val_acc={va_acc:.4f} skipped={va_skipped}")

        # Persist list of bad files each epoch (so you can grab from CloudWatch or model dir)
        try:
            if hasattr(train_ds.dataset, "bad_files") and train_ds.dataset.bad_files:
                with BAD_FILE_LOG.open("w") as f:
                    for p in sorted(set(train_ds.dataset.bad_files)):
                        f.write(f"{p}\n")
                if len(train_ds.dataset.bad_files) > MAX_BAD_FILES_BEFORE_WARN:
                    log.warning(f"Many bad files detected: {len(train_ds.dataset.bad_files)} "
                                f"(see {BAD_FILE_LOG})")
                else:
                    log.info(f"Bad files so far: {len(train_ds.dataset.bad_files)} (written to {BAD_FILE_LOG})")
        except Exception as e:
            log_exc("write_bad_file_log", e)

        if va_acc > best_acc:
            best_acc = va_acc
            try:
                torch.save(model.state_dict(), best_path)
                log.info(f"New best model saved: acc={best_acc:.4f} -> {best_path}")
            except Exception as e:
                log_exc("save_model", e)

    # 5) Save classes
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
        # Re-raise to mark job as failed (so CI/CD surfaces it), but with logs intact
        raise
