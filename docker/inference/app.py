#!/usr/bin/env python3
import os, io, json, base64, traceback
from typing import Tuple, Dict, Any
from flask import Flask, request, jsonify, Response
import numpy as np
import librosa

MODEL_DIR   = os.getenv("MODEL_DIR", "/opt/ml/model")
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "22050"))
DURATION    = float(os.getenv("DURATION", "2.0"))
N_MELS      = int(os.getenv("N_MELS", "128"))

app = Flask(__name__)

MODEL = None          # optional TorchScript model if you ever drop one in
MODEL_META: Dict[str, Any] = {"present": False, "loaded": False, "path": None}
CLASS_NAMES = None

def log(msg: str): print(msg, flush=True)

def _try_import_torch() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False

def _load_classes() -> None:
    global CLASS_NAMES
    path = os.path.join(MODEL_DIR, "classes.json")
    if os.path.exists(path):
        try:
            with open(path) as f:
                CLASS_NAMES = json.load(f)
            log(f"[startup] classes.json loaded: {len(CLASS_NAMES)} classes")
        except Exception as e:
            log(f"[startup] classes.json present but failed to parse: {e}")

def _load_model() -> None:
    """Look for a TorchScript model (model.ts). If only state-dict exists, we run without a model."""
    global MODEL, MODEL_META
    ts_path = os.path.join(MODEL_DIR, "model.ts")
    if os.path.exists(ts_path):
        MODEL_META.update({"present": True, "path": ts_path, "type": "torchscript"})
        if _try_import_torch():
            try:
                import torch
                MODEL = torch.jit.load(ts_path, map_location="cpu")
                MODEL.eval()
                MODEL_META["loaded"] = True
                log(f"[startup] TorchScript model loaded from {ts_path}")
            except Exception as e:
                MODEL_META["error"] = str(e)
                log(f"[startup] TorchScript load failed: {e}")
        else:
            MODEL_META["note"] = "torch not installed; skipping load"
            log("[startup] torch not installed; skipping model load")
        return

    # State-dict present -> we don't reconstruct (by design), just run without a model.
    for name in ("model.pt", "model.pth"):
        p = os.path.join(MODEL_DIR, name)
        if os.path.exists(p):
            MODEL_META.update({"present": True, "path": p, "type": "state_dict",
                               "note": "state-dict detected; running without executing a model"})
            log(f"[startup] Detected weights at {p}. Serving without executing a model.")
            return

    log("[startup] No model files under /opt/ml/model (OK for /ping and stub inference)")

def _preprocess(y: np.ndarray, sr: int) -> np.ndarray:
    if sr != SAMPLE_RATE:
        y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)
        sr = SAMPLE_RATE
    target = int(DURATION * SAMPLE_RATE)
    if len(y) < target:
        y = np.pad(y, (0, target - len(y)))
    else:
        y = y[:target]
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512,
                                       n_mels=N_MELS, power=2.0)
    logS = librosa.power_to_db(S + 1e-9)
    logS = (logS - logS.mean()) / (logS.std() + 1e-6)
    return logS[np.newaxis, np.newaxis, :, :].astype(np.float32)  # (1,1,N_MELS,T)

def _read_audio_from_request(req) -> Tuple[np.ndarray, int]:
    ctype = (req.headers.get("content-type") or "").lower()

    if "multipart/form-data" in ctype:
        up = req.files.get("file") or req.files.get("audio")
        if not up: raise ValueError("multipart requires file field named 'file' (or 'audio')")
        raw = up.read()
        y, sr = librosa.load(io.BytesIO(raw), sr=None, mono=True)
        return y, sr

    if "application/octet-stream" in ctype:
        raw = req.get_data()
        y, sr = librosa.load(io.BytesIO(raw), sr=None, mono=True)
        return y, sr

    if "application/json" in ctype:
        payload = req.get_json(force=True, silent=False)
        if "audio" in payload:
            raw = base64.b64decode(payload["audio"])
            y, sr = librosa.load(io.BytesIO(raw), sr=None, mono=True)
            return y, sr
        if "array" in payload:
            arr = np.array(payload["array"], dtype=np.float32)
            sr = int(payload.get("sr", SAMPLE_RATE))
            return arr, sr
        raise ValueError("JSON must include 'audio' (base64) or 'array'")

    raise ValueError(f"Unsupported Content-Type: {ctype}")

# ---------- REQUIRED SAGEMAKER ENDPOINTS ----------
@app.get("/ping")
def ping():
    # FIX: must return a body + optional status; returning 200 alone triggers 500
    return Response("OK", status=200, mimetype="text/plain")

@app.post("/invocations")
def invocations():
    try:
        y, sr = _read_audio_from_request(request)
        x = _preprocess(y, sr)

        # If a TorchScript model is actually loaded, try to run it. Otherwise stub.
        if MODEL is not None and MODEL_META.get("loaded", False):
            import torch
            with torch.no_grad():
                out = MODEL(torch.from_numpy(x))
                scores = np.asarray(out).squeeze().tolist()
            return jsonify({
                "ok": True,
                "model_loaded": True,
                "scores": scores,
                "classes": CLASS_NAMES,
                "shape": list(x.shape),
            }), 200

        # Stub reply (works even without a model in the image)
        return jsonify({
            "ok": True,
            "model_loaded": False,
            "message": "No executable model available; returning debug info only.",
            "shape": list(x.shape),
            "sr": SAMPLE_RATE,
            "duration_sec": DURATION
        }), 200

    except Exception as e:
        log("[invocations] ERROR:\n" + traceback.format_exc())
        return jsonify({"ok": False, "error": str(e)}), 400

if __name__ == "__main__":
    log(f"[startup] MODEL_DIR={MODEL_DIR}")
    try:
        for root, _, files in os.walk(MODEL_DIR):
            for f in files[:20]:
                log(f"[startup] found in model dir: {os.path.join(root, f)}")
    except Exception:
        pass
    _load_classes()
    _load_model()
    log("[startup] Flask serving on 0.0.0.0:8080")
    app.run(host="0.0.0.0", port=8080, debug=False, use_reloader=False)

