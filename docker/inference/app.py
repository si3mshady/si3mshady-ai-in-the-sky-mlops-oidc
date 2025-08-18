#!/usr/bin/env python3
import os, io, json, base64, traceback
from typing import Dict, Any, Tuple
from flask import Flask, request, jsonify, Response
import numpy as np
import librosa
import torch
import torch.nn as nn

# ---------- Config ----------
MODEL_DIR = os.environ.get("SM_MODEL_DIR", os.environ.get("MODEL_DIR", "/opt/ml/model"))
SAMPLE_RATE = int(os.environ.get("SAMPLE_RATE", "22050"))
DURATION    = float(os.environ.get("DURATION", "2.0"))
N_MELS      = int(os.environ.get("N_MELS", "128"))

# ---------- Model architecture (same as training) ----------
class AudioCNN(nn.Module):
    """
    Expects input (B, 1, 128, T).
    Uses adaptive pooling so T can vary.
    """
    def __init__(self, n_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.adapt = nn.AdaptiveAvgPool2d(output_size=(1, 16))  # -> (B,128,1,16)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.3),
            nn.Linear(2048, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.adapt(x)
        x = self.classifier(x)
        return x

# ---------- App ----------
app = Flask(__name__)
MODEL = None
CLASS_NAMES = [
    "air_conditioner","car_horn","children_playing","dog_bark","drilling",
    "engine_idling","gun_shot","jackhammer","siren","street_music",
]

def log(msg: str): print(msg, flush=True)

def _load_classes():
    global CLASS_NAMES
    p = os.path.join(MODEL_DIR, "classes.json")
    if os.path.exists(p):
        try:
            CLASS_NAMES = json.load(open(p))
            log(f"[startup] classes.json loaded ({len(CLASS_NAMES)} classes)")
        except Exception as e:
            log(f"[startup] failed to parse classes.json: {e}")

def _normalize_state_dict(sd: Dict[str, Any]) -> Dict[str, Any]:
    # Accept {"state_dict": {...}} or nested {"model": {...}}, strip common prefixes
    if "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    for k in ("model","module","model_state","model_state_dict"):
        if k in sd and isinstance(sd[k], dict):
            sd = sd[k]
            break
    # Strip leading prefixes like "model.", "module."
    def strip_prefix(k: str) -> str:
        for pref in ("model.", "module."):
            if k.startswith(pref): return k[len(pref):]
        return k
    return {strip_prefix(k): v for k, v in sd.items()}

def _load_model() -> None:
    global MODEL
    weights = None
    for name in ("model.pt", "model.pth"):
        p = os.path.join(MODEL_DIR, name)
        if os.path.exists(p):
            weights = p; break
    if not weights:
        log(f"[startup] No model.pt/pth under {MODEL_DIR}.")
        return
    sd = torch.load(weights, map_location="cpu")
    if not isinstance(sd, dict):
        log("[startup] Unexpected weights file format (not a dict)."); return
    sd = _normalize_state_dict(sd)
    net = AudioCNN(n_classes=len(CLASS_NAMES))
    missing, unexpected = net.load_state_dict(sd, strict=False)
    if missing:   log(f"[startup] load_state_dict missing keys: {list(missing)[:5]}...")
    if unexpected:log(f"[startup] load_state_dict unexpected keys: {list(unexpected)[:5]}...")
    net.eval()
    MODEL = net
    log(f"[startup] Loaded state-dict from {weights}")

def _preprocess(y: np.ndarray, sr: int) -> np.ndarray:
    if sr != SAMPLE_RATE:
        y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)
        sr = SAMPLE_RATE
    target = int(DURATION * SAMPLE_RATE)
    if len(y) < target:
        y = np.pad(y, (0, target - len(y)))
    else:
        y = y[:target]
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=N_MELS, power=2.0)
    logS = librosa.power_to_db(S + 1e-9)
    logS = (logS - logS.mean()) / (logS.std() + 1e-6)
    return logS[np.newaxis, np.newaxis, :, :].astype(np.float32)  # (1,1,128,T)

def _read_audio_from_request(req) -> Tuple[np.ndarray, int, str]:
    ctype = (req.headers.get("content-type") or "").lower()
    if "multipart/form-data" in ctype:
        up = req.files.get("file") or req.files.get("audio")
        if not up: raise ValueError("multipart/form-data requires field 'file' (or 'audio')")
        raw = up.read()
        y, sr = librosa.load(io.BytesIO(raw), sr=None, mono=True)
        return y, sr, "multipart/form-data"
    if "application/octet-stream" in ctype:
        raw = req.get_data()
        y, sr = librosa.load(io.BytesIO(raw), sr=None, mono=True)
        return y, sr, "application/octet-stream"
    if "application/json" in ctype:
        payload = req.get_json(force=True, silent=False)
        if "audio" in payload:
            raw = base64.b64decode(payload["audio"])
            y, sr = librosa.load(io.BytesIO(raw), sr=None, mono=True)
            return y, sr, "application/json"
        if "array" in payload:
            arr = np.array(payload["array"], dtype=np.float32)
            sr = int(payload.get("sr", SAMPLE_RATE))
            return arr, sr, "application/json"
        raise ValueError("JSON must include 'audio' (base64) or 'array'")
    raise ValueError(f"Unsupported Content-Type: {ctype}")

@app.get("/ping")
def ping() -> Response:
    # MUST be a 200 for SageMaker health checks. Keep it simple.
    return Response("OK", status=200, mimetype="text/plain")

@app.post("/invocations")
def invocations():
    try:
        if MODEL is None:
            return jsonify({"error": f"Model not loaded from {MODEL_DIR}/model.pt"}), 500
        y, sr, _ = _read_audio_from_request(request)
        x = _preprocess(y, sr)
        with torch.no_grad():
            logits = MODEL(torch.from_numpy(x))
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy().tolist()
        top_idx = int(np.argmax(probs))
        return jsonify({
            "label": CLASS_NAMES[top_idx],
            "confidence": float(probs[top_idx]),
            "classes": CLASS_NAMES,
            "probs": probs,
        })
    except Exception as e:
        log("[invocations] ERROR:\n" + traceback.format_exc())
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    log(f"[startup] MODEL_DIR={MODEL_DIR}")
    _load_classes()
    _load_model()
    log("[startup] Flask on 0.0.0.0:8080")
    from werkzeug.serving import run_simple
    run_simple("0.0.0.0", 8080, app, use_reloader=False)

