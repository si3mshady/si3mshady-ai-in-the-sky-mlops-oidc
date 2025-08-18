import os, io, json, base64, traceback
from typing import Tuple
from flask import Flask, request, jsonify
import numpy as np
import librosa

MODEL_DIR   = os.getenv("MODEL_DIR", "/opt/ml/model")
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "22050"))
DURATION    = float(os.getenv("DURATION", "2.0"))
N_MELS      = int(os.getenv("N_MELS", "128"))

app = Flask(__name__)

def _preprocess(y: np.ndarray, sr: int) -> np.ndarray:
    if sr != SAMPLE_RATE:
        y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)
    target = int(DURATION * SAMPLE_RATE)
    if len(y) < target:
        y = np.pad(y, (0, target - len(y)))
    else:
        y = y[:target]
    S = librosa.feature.melspectrogram(y=y, sr=SAMPLE_RATE, n_fft=2048, hop_length=512, n_mels=N_MELS, power=2.0)
    logS = librosa.power_to_db(S + 1e-9)
    logS = (logS - logS.mean()) / (logS.std() + 1e-6)
    return logS[np.newaxis, np.newaxis, :, :].astype(np.float32)  # (1,1,128,T)

def _read_audio_from_request(req) -> Tuple[np.ndarray, int]:
    ctype = (req.headers.get("content-type") or "").lower()
    if "multipart/form-data" in ctype:
        up = req.files.get("file") or req.files.get("audio")
        if not up: raise ValueError("multipart/form-data requires 'file' (or 'audio')")
        raw = up.read()
        y, sr = librosa.load(io.BytesIO(raw), sr=None, mono=True)
        return y, sr
    if "application/octet-stream" in ctype:
        raw = req.get_data()
        y, sr = librosa.load(io.BytesIO(raw), sr=None, mono=True)
        return y, sr
    if "application/json" in ctype:
        payload = req.get_json(force=True)
        if "audio" in payload:  # base64
            raw = base64.b64decode(payload["audio"])
            y, sr = librosa.load(io.BytesIO(raw), sr=None, mono=True)
            return y, sr
        if "array" in payload:
            arr = np.array(payload["array"], dtype=np.float32)
            sr = int(payload.get("sr", SAMPLE_RATE))
            return arr, sr
        raise ValueError("JSON must include 'audio' (base64) or 'array'")
    raise ValueError(f"Unsupported Content-Type: {ctype}")

@app.route("/ping", methods=["GET"])
def ping():
    # Keep this dirt simple so SageMaker health check always returns 200.
    return ("OK", 200)

@app.route("/invocations", methods=["POST"])
def invocations():
    try:
        y, sr = _read_audio_from_request(request)
        x = _preprocess(y, sr)

        # If you later export a TorchScript model to /opt/ml/model/model.ts, this will run it.
        # Otherwise we return a stub so your client-upstream path still works.
        try:
            import torch  # optional
            ts_path = os.path.join(MODEL_DIR, "model.ts")
            if os.path.exists(ts_path):
                m = torch.jit.load(ts_path, map_location="cpu").eval()
                with torch.no_grad():
                    out = m(torch.from_numpy(x))
                scores = out.squeeze().detach().cpu().numpy().tolist()
                return jsonify({"model_loaded": True, "scores": scores, "shape": list(x.shape)})
        except Exception as e:
            # Fall through to stub
            print(f"[infer] model exec skipped/failed: {e}", flush=True)

        # Stub response if no executable model is present
        return jsonify({"model_loaded": False, "shape": list(x.shape), "sr": SAMPLE_RATE, "duration_sec": DURATION})
    except Exception as e:
        print("[invocations] ERROR:\n" + traceback.format_exc(), flush=True)
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False, use_reloader=False)

