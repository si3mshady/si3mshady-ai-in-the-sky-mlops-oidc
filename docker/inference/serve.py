import io, os, sys, json, inspect, logging
from typing import Dict, Any, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse

# import path
CODE_DIR = "/opt/ml/code"
SRC_DIR = os.path.join(CODE_DIR, "src")
for p in (CODE_DIR, SRC_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
import numpy as np
import librosa

# try local first, fallback to src layout
try:
    from models.audio_cnn import AudioCNN
except ModuleNotFoundError:
    from src.models.audio_cnn import AudioCNN  # type: ignore

log = logging.getLogger("inference")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

app = FastAPI(title="UrbanSound Inference", version="1.0")

SAMPLE_RATE = 22050
DURATION = 4.0
N_MELS = 128
DEVICE = "cpu"




# 🔒 Hard-coded 17 classes; do NOT auto-load from classes.json
CLASS_NAMES = [ "siren","alarms",
        "domestic","gunfire","police","forced_entry"]

MODEL: Optional[torch.nn.Module] = None

def _instantiate_model(n_classes: int) -> torch.nn.Module:
    init = AudioCNN
    sig = inspect.signature(init)
    for name in ("n_classes","num_classes","classes","out_dim","n_outputs"):
        if name in sig.parameters:
            try:
                return init(**{name: n_classes})
            except TypeError:
                pass
    return init()  # type: ignore

def load_model() -> torch.nn.Module:
    # NOTE: do not load/override CLASS_NAMES from JSON
    weights_path = "/opt/ml/model/model.pt"

    model = _instantiate_model(n_classes=len(CLASS_NAMES))
    model.to(DEVICE)
    model.eval()

    if os.path.exists(weights_path):
        state = torch.load(weights_path, map_location=DEVICE)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        try:
            model.load_state_dict(state, strict=False)
            log.info("Loaded weights from %s", weights_path)
        except Exception:
            for k in ("model","net","module","model_state","model_state_dict"):
                if isinstance(state, dict) and k in state and isinstance(state[k], dict):
                    model.load_state_dict(state[k], strict=False)
                    log.info("Loaded nested weights from key '%s' in %s", k, weights_path)
                    break
    else:
        log.warning("No weights found at %s; model will output random-ish logits", weights_path)

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
    return logS[np.newaxis, np.newaxis, :, :].astype(np.float32)

def predict_ndarray(x: np.ndarray) -> Dict[str, Any]:
    with torch.no_grad():
        tensor = torch.from_numpy(x).to(DEVICE)
        logits = MODEL(tensor)  # type: ignore
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    top_idx = int(np.argmax(probs))
    return {
        "label": CLASS_NAMES[top_idx],
        "confidence": float(probs[top_idx]),
        "classes": CLASS_NAMES,
        "probs": [float(p) for p in probs],
    }

@app.get("/ping")
def ping() -> PlainTextResponse:
    return PlainTextResponse("OK", status_code=200)

@app.post("/invocations")
async def invocations(request: Request, file: UploadFile = File(None)):
    ctype = (request.headers.get("content-type") or "").lower()
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
            if "audio" in payload:
                import base64
                raw = base64.b64decode(payload["audio"])
                wav, sr = librosa.load(io.BytesIO(raw), sr=None, mono=True)
                x = preprocess_audio(wav, sr)
                return JSONResponse(predict_ndarray(x))
            elif "array" in payload:
                arr = np.array(payload["array"], dtype=np.float32)
                sr = int(payload.get("sr", SAMPLE_RATE))
                x = preprocess_audio(arr, sr)
                return JSONResponse(predict_ndarray(x))
            else:
                raise HTTPException(status_code=400, detail="Unsupported JSON payload")

        elif "application/octet-stream" in ctype:
            raw = await request.body()
            wav, sr = librosa.load(io.BytesIO(raw), sr=None, mono=True)
            x = preprocess_audio(wav, sr)
            return JSONResponse(predict_ndarray(x))

        else:
            raise HTTPException(status_code=415, detail=f"Unsupported Content-Type: {ctype}")

    except HTTPException:
        raise
    except Exception as e:
        log.exception("inference_error")
        return JSONResponse({"error": str(e)}, status_code=500)

@app.on_event("startup")
def _load_once():
    global MODEL
    MODEL = load_model()
    log.info("Inference server ready. classes=%s", CLASS_NAMES)

