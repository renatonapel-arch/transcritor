FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    DATA_DIR=/data \
    PORT=8830 \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py index.html ./

RUN mkdir -p /data
EXPOSE 8830

CMD ["python", "app.py"]
