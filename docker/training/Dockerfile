# docker/training/Dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# libs for librosa/soundfile
RUN apt-get update && apt-get install -y --no-install-recommends \
      libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/ml/code

# 1) Install Python deps
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# 2) Copy training script and model code
#    You have audio_cnn.py in the build context alongside entryPoint.py.
#    We place it under /opt/ml/code/models/audio_cnn.py so "from models.audio_cnn import AudioCNN" works.
COPY entryPoint.py ./train.py
COPY audio_cnn.py ./models/audio_cnn.py

# Make sure /opt/ml/code is on sys.path
ENV PYTHONPATH=/opt/ml/code

# SageMaker expects this, but ENTRYPOINT also runs it locally
ENV SAGEMAKER_PROGRAM=train.py
ENTRYPOINT ["python", "train.py"]

