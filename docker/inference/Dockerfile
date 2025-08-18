FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_DIR=/opt/ml/model

# System libs for audio + curl for quick checks
RUN apt-get update && apt-get install -y --no-install-recommends \
      libsndfile1 ffmpeg curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/ml/code

# Minimal deps + PyTorch CPU (no CUDA)
RUN pip install --no-cache-dir \
      flask numpy librosa soundfile \
  && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch

# App & starter
COPY app.py /opt/ml/code/app.py
COPY start.sh /usr/local/bin/start.sh
RUN chmod +x /usr/local/bin/start.sh

EXPOSE 8080

ENTRYPOINT ["/usr/local/bin/start.sh"]
CMD ["serve"]

