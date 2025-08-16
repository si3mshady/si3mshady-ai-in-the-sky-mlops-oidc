FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/opt/ml/code

# audio libs + curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
      libsndfile1 ffmpeg curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/ml/code

# requirements file lives at docker/inference/requirements.txt
COPY docker/inference/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# copy the FastAPI server and the minimal code it needs
COPY docker/inference/serve.py ./serve.py
COPY models ./models
COPY src/utils ./src/utils
COPY configs/audio_model.yaml ./configs/audio_model.yaml

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD curl -fsS http://localhost:8080/ping || exit 1

CMD ["python", "serve.py"]

