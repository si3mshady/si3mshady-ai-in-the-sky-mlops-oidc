import base64
import io
import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from fastapi import FastAPI, Request, Response, HTTPException
from pydantic import BaseModel

# --- local imports from your repo
from src.models.audio_cnn import AudioCNN
from src.utils.audio import wav_to_melspec, normalize_spec

APP_PORT = int(os.getenv("PORT", "8080"))
MODEL_DIR = Path(os.getenv("MODEL_DIR", "/opt/ml/model"))

app = FastAPI(title="UrbanSound FastAPI Inference", version="1.0")

_model: Optional[torch.nn.Module] = None
_classes: Optional[list[str]] = None
_n_mels: int = 64


class JsonPayload(BaseModel):
    audio_base64: str


def _load_model():
    global _model, _classes, _n_mels
    model_path = MODEL_DIR / "model.pt"
    classes_path = MODEL_DIR / "classes.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Missing model file: {model_path}")
    if not classes_path.exists():
        raise FileNotFoundError(f"Missing classes file: {classes_path}")

    ckpt = torch.load(model_path, map_location="cpu")
    _n_mels = int(ckpt.get("n_mels", 64))
    num_classes = int(ckpt.get("num_classes", 10))

    m = AudioCNN(n_mels=_n_mels, num_classes=num_classes)
    m.load_state_dict(ckpt["state_dict"])
    m.eval()
    _model = m

    with open(classes_path, "r") as f:
        _classes = json.load(f)


def _ensure_loaded():
    if _model is None or _classes is None:
        _load_model()


def _predict_from_wave_bytes(wav_bytes: bytes) -> dict:
    """Decode audio (wav), make mel-spectrogram, run model, return JSON."""
    _ensure_loaded()
    y, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)  # mono
    spec = normalize_spec(wav_to_melspec(y, sr, n_mels=_n_mels))  # (M, T)
    x = torch.from_numpy(spec).unsqueeze(0).unsqueeze(0)  # (1,1,M,T)
    with torch.no_grad():
        probs = F.softmax(_model(x), dim=1)[0].cpu().numpy()
    idx = int(np.argmax(probs))
    return {
        "label": _classes[idx],
        "probabilities": { _classes[i]: float(probs[i]) for i in range(len(_classes)) }
    }


# ---------- SageMaker-compatible endpoints ----------

@app.get("/ping")
@app.head("/ping")
def ping():
    try:
        _ensure_loaded()
        return Response(content="OK", media_type="text/plain")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model not ready: {e}")


@app.post("/invocations")
async def invocations(request: Request):
    """
    Accepts:
      - application/json: {"audio_base64": "<...>"} (WAV bytes base64-encoded)
      - application/octet-stream or audio/wav: raw WAV bytes
    Returns:
      - application/json with {"label": "...", "probabilities": {...}}
    """
    ctype = request.headers.get("content-type", "").split(";")[0].strip().lower()

    try:
        if ctype == "application/json":
            payload = await request.json()
            data = JsonPayload(**payload)
            wav_bytes = base64.b64decode(data.audio_base64)
            result = _predict_from_wave_bytes(wav_bytes)
            return result

        elif ctype in ("application/octet-stream", "audio/wav"):
            wav_bytes = await request.body()
            result = _predict_from_wave_bytes(wav_bytes)
            return result

        else:
            raise HTTPException(status_code=415, detail=f"Unsupported content-type: {ctype}")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Inference failed: {e}")


if __name__ == "__main__":
    # Run Uvicorn without gunicorn (SageMaker starts a single process by default)
    import uvicorn
    # 0.0.0.0:8080 is required for SageMaker containers
    uvicorn.run(app, host="0.0.0.0", port=APP_PORT, log_level="info")

