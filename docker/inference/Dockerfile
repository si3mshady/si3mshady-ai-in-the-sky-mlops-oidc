FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/opt/ml/code

# Audio libs for librosa/soundfile + ffmpeg decoding
RUN apt-get update && apt-get install -y --no-install-recommends \
      libsndfile1 ffmpeg curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/ml/code

# Install runtime deps (FastAPI + Uvicorn are in your requirements.txt already)
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# App code
COPY src ./src
COPY src/inference/serve.py ./serve.py

# Healthcheck hits our FastAPI /ping
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD curl -fsS http://localhost:8080/ping || exit 1

# Run FastAPI app on 0.0.0.0:8080 (SageMaker expects 8080)
CMD ["python", "serve.py"]

