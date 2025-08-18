import os, io, json, base64, traceback
from typing import Tuple
from flask import Flask, request, jsonify
import numpy as np
import librosa

import torch
import torch.nn as nn

# -------- SageMaker-provided model path --------
MODEL_DIR = os.environ.get("SM_MODEL_DIR", os.environ.get("MODEL_DIR", "/opt/ml/model"))
WEIGHTS_CANDIDATES = ["model.ts", "model.pt", "model.pth"]  # we will use EXACTLY what's there

# -------- Model definition (same as training AudioCNN) --------
class AudioCNN(nn.Module):
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

# -------- App setup --------
app = Flask(__name__)

SAMPLE_RATE = int(os.environ.get("SAMPLE_RATE", "22050"))
DURATION    = float(os.environ.get("DURATION", "2.0"))
N_MELS      = int(os.environ.get("N_MELS", "128"))

CLASSES = None
MODEL: nn.Module | torch.jit.ScriptModule | None = None
IS_TS = False

def log(msg: str):  # simple stdout logger
    print(msg, flush=True)

def _load_classes():
    global CLASSES
    path = os.path.join(MODEL_DIR, "classes.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"classes.json not found in {MODEL_DIR}")
    with open(path, "r") as f:
        CLASSES = json.load(f)
    if not isinstance(CLASSES, list) or len(CLASSES) == 0:
        raise ValueError("classes.json must be a non-empty list of class names")
    log(f"[startup] Loaded classes.json with {len(CLASSES)} classes")

def _load_model():
    """Load the model ONLY from /opt/ml/model (SageMaker untar location)."""
    global MODEL, IS_TS

    # Prefer TorchScript if present, else load state_dict
    found = None
    for name in WEIGHTS_CANDIDATES:
        p = os.path.join(MODEL_DIR, name)
        if os.path.exists(p):
            found = p
            break
    if found is None:
        raise FileNotFoundError(f"No model file found in {MODEL_DIR} (looked for {WEIGHTS_CANDIDATES})")

    if found.endswith(".ts"):
        MODEL = torch.jit.load(found, map_location="cpu")
        MODEL.eval()
        IS_TS = True
        log(f"[startup] Loaded TorchScript model: {found}")
        return

    # state_dict path -> need architecture (this image contains AudioCNN)
    if CLASSES is None:
        raise RuntimeError("classes.json must be loaded before state_dict to size the classifier head")

    model = AudioCNN(n_classes=len(CLASSES))
    state = torch.load(found, map_location="cpu")

    # accommodate state dicts saved under various keys
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    for k in ("model", "net", "module", "model_state", "model_state_dict"):
        if isinstance(state, dict) and k in state and isinstance(state[k], dict):
            state = state[k]
            break

    model.load_state_dict(state, strict=False)
    model.eval()
    MODEL = model
    log(f"[startup] Loaded state_dict model: {found}")

def _read_audio_from_request(req) -> Tuple[np.ndarray, int]:
    ctype = (req.headers.get("content-type") or "").lower()

    if "multipart/form-data" in ctype:
        up = req.files.get("file") or req.files.get("audio")
        if not up:
            raise ValueError("multipart/form-data requires a 'file' (or 'audio') field")
        raw = up.read()
        y, sr = librosa.load(io.BytesIO(raw), sr=None, mono=True)
        return y, sr

    if "application/octet-stream" in ctype:
        raw = req.get_data()
        y, sr = librosa.load(io.BytesIO(raw), sr=None, mono=True)
        return y, sr

    if "application/json" in ctype:
        payload = req.get_json(force=True, silent=False)
        if "audio" in payload:  # base64 bytes
            raw = base64.b64decode(payload["audio"])
            y, sr = librosa.load(io.BytesIO(raw), sr=None, mono=True)
            return y, sr
        if "array" in payload:
            arr = np.array(payload["array"], dtype=np.float32)
            sr = int(payload.get("sr", SAMPLE_RATE))
            return arr, sr
        raise ValueError("JSON must include 'audio' (base64) or 'array'")

    raise ValueError(f"Unsupported Content-Type: {ctype}")

def _preprocess(y: np.ndarray, sr: int) -> np.ndarray:
    # resample
    if sr != SAMPLE_RATE:
        y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)
        sr = SAMPLE_RATE
    # pad/trim
    target = int(DURATION * SAMPLE_RATE)
    if len(y) < target:
        y = np.pad(y, (0, target - len(y)))
    else:
        y = y[:target]
    # mel-spec
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=N_MELS, power=2.0)
    logS = librosa.power_to_db(S + 1e-9)
    logS = (logS - logS.mean()) / (logS.std() + 1e-6)
    # shape (1,1,128,T)
    return logS[np.newaxis, np.newaxis, :, :].astype(np.float32)

# ------------------ Routes ------------------
@app.get("/ping")
def ping():
    # MUST return 200 for SageMaker health checks
    return ("OK", 200)

@app.post("/invocations")
def invocations():
    try:
        if MODEL is None or CLASSES is None:
            return jsonify({"error": "Model not loaded from /opt/ml/model"}), 500

        y, sr = _read_audio_from_request(request)
        x = _preprocess(y, sr)

        with torch.no_grad():
            tensor = torch.from_numpy(x)
            logits = MODEL(tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        top = int(np.argmax(probs))
        return jsonify({
            "label": CLASSES[top],
            "confidence": float(probs[top]),
            "classes": CLASSES,
            "probs": [float(p) for p in probs]
        })
    except Exception as e:
        log("[invocations] ERROR:\n" + traceback.format_exc())
        return jsonify({"error": str(e)}), 400

# ------------------ Startup ------------------
if __name__ == "__main__":
    log(f"[startup] MODEL_DIR={MODEL_DIR}")
    # Helpful for proving SageMaker untar worked
    try:
        for root, _, files in os.walk(MODEL_DIR):
            for f in files[:30]:
                log(f"[startup] found: {os.path.join(root, f)}")
    except Exception:
        pass

    _load_classes()    # required
    _load_model()      # required
    log("[startup] Flask serving on 0.0.0.0:8080")
    app.run(host="0.0.0.0", port=8080, debug=False, use_reloader=False)

