FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/opt/ml/code \
    # keep torch CPU only, silence CUDA probing noise
    CUDA_VISIBLE_DEVICES=""

# audio libs + curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
      libsndfile1 ffmpeg curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/ml/code

# requirements file lives at docker/inference/requirements.txt
COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --upgrade pip \
 && pip install --no-cache-dir -r /tmp/requirements.txt

# app
COPY serve.py ./serve.py
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --retries=10 CMD curl -fsS http://localhost:8080/ping || exit 1

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["serve"]

