import os, io, json, base64, traceback
from typing import Tuple
from flask import Flask, request, jsonify
import numpy as np
import librosa
import torch

# SageMaker mounts your artifact here:
MODEL_DIR = os.environ.get("SM_MODEL_DIR", os.environ.get("MODEL_DIR", "/opt/ml/model"))
TS_PATH   = os.path.join(MODEL_DIR, "model.ts")      # required
CLASSES_P = os.path.join(MODEL_DIR, "classes.json")  # required

SAMPLE_RATE = int(os.environ.get("SAMPLE_RATE", "22050"))
DURATION    = float(os.environ.get("DURATION", "2.0"))
N_MELS      = int(os.environ.get("N_MELS", "128"))

app = Flask(__name__)
MODEL = None
CLASSES = None

def log(m): print(m, flush=True)

def _load_once():
    """Load ONLY what SageMaker untarred into /opt/ml/model."""
    global MODEL, CLASSES
    if MODEL is not None:  # already loaded
        return

    if not os.path.exists(TS_PATH):
        raise FileNotFoundError(f"Missing TorchScript model at {TS_PATH}. "
                                "Ensure your training job packaged model.ts in model.tar.gz.")
    if not os.path.exists(CLASSES_P):
        raise FileNotFoundError(f"Missing classes.json at {CLASSES_P}.")

    with open(CLASSES_P, "r") as f:
        CLASSES = json.load(f)
    if not isinstance(CLASSES, list) or len(CLASSES) == 0:
        raise ValueError("classes.json must be a non-empty list of class names")

    MODEL = torch.jit.load(TS_PATH, map_location="cpu").eval()
    log(f"[startup] Loaded TorchScript from {TS_PATH} with {len(CLASSES)} classes")

def _read_audio(req) -> Tuple[np.ndarray, int]:
    ctype = (req.headers.get("content-type") or "").lower()
    if "multipart/form-data" in ctype:
        up = req.files.get("file") or req.files.get("audio")
        if not up: raise ValueError("multipart/form-data needs 'file' or 'audio'")
        raw = up.read()
        y, sr = librosa.load(io.BytesIO(raw), sr=None, mono=True); return y, sr
    if "application/octet-stream" in ctype:
        raw = req.get_data()
        y, sr = librosa.load(io.BytesIO(raw), sr=None, mono=True); return y, sr
    if "application/json" in ctype:
        payload = req.get_json(force=True)
        if "audio" in payload:
            raw = base64.b64decode(payload["audio"])
            y, sr = librosa.load(io.BytesIO(raw), sr=None, mono=True); return y, sr
        if "array" in payload:
            arr = np.array(payload["array"], dtype=np.float32)
            sr  = int(payload.get("sr", SAMPLE_RATE))
            return arr, sr
        raise ValueError("JSON must include 'audio' (base64) or 'array'")
    raise ValueError(f"Unsupported Content-Type: {ctype}")

def _preprocess(y: np.ndarray, sr: int) -> np.ndarray:
    if sr != SAMPLE_RATE:
        y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)
        sr = SAMPLE_RATE
    target = int(DURATION * SAMPLE_RATE)
    if len(y) < target: y = np.pad(y, (0, target - len(y)))
    else:               y = y[:target]
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512,
                                       n_mels=N_MELS, power=2.0)
    logS = librosa.power_to_db(S + 1e-9)
    logS = (logS - logS.mean()) / (logS.std() + 1e-6)
    return logS[np.newaxis, np.newaxis, :, :].astype(np.float32)  # (1,1,128,T)

@app.get("/ping")
def ping():
    # keep endpoint stable; don’t force model load here
    return ("OK", 200)

@app.post("/invocations")
def invocations():
    try:
        _load_once()  # prove SageMaker mounted /opt/ml/model contents
        y, sr = _read_audio(request)
        x = _preprocess(y, sr)
        with torch.no_grad():
            logits = MODEL(torch.from_numpy(x))
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0].tolist()
        top = int(np.argmax(probs))
        return jsonify({"label": CLASSES[top], "confidence": float(probs[top]),
                        "classes": CLASSES, "probs": probs})
    except Exception as e:
        log("[invocations] ERROR:\n" + traceback.format_exc())
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    log(f"[startup] MODEL_DIR={MODEL_DIR}")
    try:
        for r, _, fs in os.walk(MODEL_DIR):
            for f in fs[:30]:
                log(f"[startup] found: {os.path.join(r,f)}")
    except Exception:
        pass
    app.run(host="0.0.0.0", port=8080, debug=False, use_reloader=False)

