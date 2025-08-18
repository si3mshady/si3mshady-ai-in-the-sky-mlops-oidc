FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_DIR=/opt/ml/model

# Audio libs + curl for healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends \
      libsndfile1 ffmpeg curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/ml/code

# Python deps
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# App
COPY app.py ./app.py

EXPOSE 8080
# If app is up, this returns 200 and lets SageMaker mark container healthy
HEALTHCHECK --interval=30s --timeout=5s --retries=20 CMD curl -fsS http://localhost:8080/ping || exit 1

# No wrapper scripts, no uvicorn: just run the Flask app
CMD ["python", "app.py"]

