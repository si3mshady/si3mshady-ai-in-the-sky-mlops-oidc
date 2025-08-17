FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 ffmpeg && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/ml/code

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY entryPoint.py ./train.py

ENV SAGEMAKER_PROGRAM=train.py

ENTRYPOINT ["python","train.py"]

