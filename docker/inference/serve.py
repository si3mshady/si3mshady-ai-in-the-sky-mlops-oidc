#!/usr/bin/env python3
import os, io, json, base64, traceback
from flask import Flask, request, jsonify, Response
import numpy as np
import librosa
import torch
import torch.nn as nn

MODEL_DIR = os.environ.get("SM_MODEL_DIR", os.environ.get("MODEL_DIR", "/opt/ml/model"))
SAMPLE_RATE = 22050
DURATION = 3.0  # Match training
N_MELS = 128
DEVICE = torch.device("cpu")

class AudioCNN(nn.Module):
    def __init__(self, n_classes: int = 5):  # Updated for 5 classes
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
        self.adapt = nn.AdaptiveAvgPool2d(output_size=(1, 16))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.3),
            nn.Linear(2048, n_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.adapt(x)
        x = self.classifier(x)
        return x

app = Flask(__name__)
MODEL = None
CLASS_NAMES = ["alarms", "domestic", "forced_entry", "gunfire", "police", "siren"]

def _to_window(y: np.ndarray, sr: int) -> np.ndarray:
    if sr != SAMPLE_RATE:
        y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)
    
    target = int(DURATION * SAMPLE_RATE)
    if len(y) < target:
        y = np.pad(y, (0, target - len(y)))
    else:
        y = y[:target]
    
    S = librosa.feature.melspectrogram(y=y, sr=SAMPLE_RATE, n_mels=N_MELS)
    S_db = librosa.power_to_db(S, ref=np.max)
    
    # SAME normalization as training
    S_norm = (S_db - (-80.0)) / (0.0 - (-80.0))
    S_norm = np.clip(S_norm, 0.0, 1.0)
    
    return S_norm[np.newaxis, np.newaxis, :, :].astype(np.float32)

def _load_model():
    global MODEL
    model_path = os.path.join(MODEL_DIR, "model.pt")
    if os.path.exists(model_path):
        MODEL = AudioCNN(n_classes=len(CLASS_NAMES))
        MODEL.load_state_dict(torch.load(model_path, map_location="cpu"))
        MODEL.eval()
        print(f"✅ Model loaded: {len(CLASS_NAMES)} classes")

def _read_audio(req):
    if "multipart/form-data" in req.headers.get("content-type", ""):
        file = req.files.get("file")
        if file:
            y, sr = librosa.load(io.BytesIO(file.read()), sr=None, mono=True)
            return y, sr
    raise ValueError("Upload audio file via multipart/form-data")

@app.get("/ping")
def ping():
    return Response("OK", status=200)

@app.post("/invocations")
def invocations():
    try:
        if MODEL is None:
            return jsonify({"error": "Model not loaded"}), 500
        
        y, sr = _read_audio(request)
        x = _to_window(y, sr)
        
        with torch.no_grad():
            logits = MODEL(torch.from_numpy(x))
            probs = torch.softmax(logits, dim=1).numpy()[0]
        
        top = int(np.argmax(probs))
        
        result = {
            "label": CLASS_NAMES[top],
            "confidence": float(probs[top]),
            "all_probabilities": {cls: float(prob) for cls, prob in zip(CLASS_NAMES, probs)}
        }
        
        print(f"🎯 Prediction: {result}")
        return jsonify(result)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    _load_model()
    from werkzeug.serving import run_simple
    run_simple("0.0.0.0", 8080, app)

