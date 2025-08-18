# docker/training/Dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/ml/code

COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --upgrade pip \
 && pip install --no-cache-dir -r /tmp/requirements.txt \
 && python - <<'PY'
import sys, librosa, torch, soundfile
print("[build-check] python:", sys.version.split()[0])
print("[build-check] librosa:", librosa.__version__)
print("[build-check] torch:", torch.__version__)
print("[build-check] soundfile:", soundfile.__version__)
PY

COPY entryPoint.py ./train.py
COPY audio_cnn.py ./models/audio_cnn.py

ARG GIT_SHA=dev
ENV GIT_SHA=$GIT_SHA
ENV PYTHONPATH=/opt/ml/code
ENV SAGEMAKER_PROGRAM=train.py

ENTRYPOINT ["python", "train.py"]

