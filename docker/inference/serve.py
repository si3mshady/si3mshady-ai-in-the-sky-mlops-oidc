# /opt/ml/code/serve.py
import io, os, sys, json, inspect
from typing import Dict, Any, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse
import uvicorn

CODE_DIR = "/opt/ml/code"
SRC_DIR = os.path.join(CODE_DIR, "src")
for p in (CODE_DIR, SRC_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
import numpy as np
import librosa

try:
    from models.audio_cnn import AudioCNN  # your repo layout
except ModuleNotFoundError:
    from src.models.audio_cnn import AudioCNN  # fallback if you move it later

app = FastAPI(title="UrbanSound Inference", version="1.0")

SAMPLE_RATE = 22050
DURATION = 4.0
N_MELS = 128
DEVICE = "cpu"

CLASS_NAMES = [
    "air_conditioner", "car_horn", "children_playing", "dog_bark", "drilling",
    "engine_idling", "gun_shot", "jackhammer", "siren", "street_music",
]

MODEL: Optional[torch.nn.Module] = None


def _instantiate_model() -> torch.nn.Module:
    """
    Instantiate AudioCNN no matter what its __init__ is named like.
    Tries common arg names: n_classes, num_classes, classes, out_dim, n_outputs.
    Falls back to no-arg constructor.
    """
    init = AudioCNN
    sig = inspect.signature(init)
    class_count = len(CLASS_NAMES)

    preferred_names = ("n_classes", "num_classes", "classes", "out_dim", "n_outputs")
    for name in preferred_names:
        if name in sig.parameters:
            try:
                return init(**{name: class_count})
            except TypeError:
                pass
    # try no-arg
    try:
        return init()
    except TypeError as e:
        raise RuntimeError(
            f"Cannot construct AudioCNN; __init__ signature is {sig}. "
            f"Tried {preferred_names} and no-arg."
        ) from e


def load_model() -> torch.nn.Module:
    model = _instantiate_model()
    model.to(DEVICE)
    model.eval()

    weights_path = "/opt/ml/model/model.pt"
    if os.path.exists(weights_path):
        state = torch.load(weights_path, map_location=DEVICE)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        try:
            model.load_state_dict(state, strict=False)
        except Exception as e:
            # Try loading under 'model' key, etc.
            if isinstance(state, dict):
                for k in ("model", "net", "module", "model_state", "model_state_dict"):
                    if k in state and isinstance(state[k], dict):
                        model.load_state_dict(state[k], strict=False)
                        break
                else:
                    raise e
            else:
                raise e
    return model


def preprocess_audio(wav: np.ndarray, sr: int) -> np.ndarray:
    target_len = int(DURATION * SAMPLE_RATE)
    if sr != SAMPLE_RATE:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=SAMPLE_RATE)
        sr = SAMPLE_RATE
    if len(wav) < target_len:
        wav = np.pad(wav, (0, target_len - len(wav)))
    else:
        wav = wav[:target_len]

    S = librosa.feature.melspectrogram(
        y=wav, sr=sr, n_fft=2048, hop_length=512, n_mels=N_MELS, power=2.0
    )
    logS = librosa.power_to_db(S + 1e-9)
    logS = (logS - logS.mean()) / (logS.std() + 1e-6)
    return logS[np.newaxis, np.newaxis, :, :].astype(np.float32)  # (1,1,N_MELS,T)


def predict_ndarray(x: np.ndarray) -> Dict[str, Any]:
    with torch.no_grad():
        tensor = torch.from_numpy(x).to(DEVICE)
        logits = MODEL(tensor)  # type: ignore
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    top_idx = int(np.argmax(probs))
    return {
        "label": CLASS_NAMES[top_idx],
        "confidence": float(probs[top_idx]),
        "probs": {CLASS_NAMES[i]: float(p) for i, p in enumerate(probs)},
    }


@app.get("/ping")
def ping() -> PlainTextResponse:
    return PlainTextResponse("OK", status_code=200 if MODEL is not None else 503)


@app.post("/invocations")
async def invocations(request: Request, file: UploadFile = File(None)):
    ctype = request.headers.get("content-type", "")

    try:
        if "multipart/form-data" in ctype:
            if file is None:
                raise HTTPException(status_code=400, detail="No file provided")
            raw = await file.read()
            wav, sr = librosa.load(io.BytesIO(raw), sr=None, mono=True)
            x = preprocess_audio(wav, sr)
            return JSONResponse(predict_ndarray(x))

        elif "application/json" in ctype:
            payload = await request.json()
            if "array" in payload:
                arr = np.array(payload["array"], dtype=np.float32)
                sr = int(payload.get("sr", SAMPLE_RATE))
                x = preprocess_audio(arr, sr)
                return JSONResponse(predict_ndarray(x))
            elif "s3_uri" in payload:
                raise HTTPException(status_code=501, detail="s3_uri handling not implemented")
            else:
                raise HTTPException(status_code=400, detail="Unsupported JSON payload")

        else:
            raise HTTPException(status_code=415, detail=f"Unsupported Content-Type: {ctype}")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Lifespan (replaces deprecated on_event)
@app.on_event("startup")
def _load_once():
    global MODEL
    MODEL = load_model()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)

