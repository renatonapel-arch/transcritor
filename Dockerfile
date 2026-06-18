FROM python:3.12-slim

# faster-whisper usa PyAV (ffmpeg embutido na wheel) e yt-dlp baixa um stream de
# áudio único -> não precisa de ffmpeg do sistema. Imagem slim basta.
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
