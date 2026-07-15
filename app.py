#!/usr/bin/env python3
"""
Transcritor de Vídeos — serviço (Clavis Renato)
-----------------------------------------------
FastAPI: baixa o áudio de um vídeo (TikTok/YouTube/Instagram/...) com yt-dlp e
transcreve 100% local com faster-whisper. Custo zero de API.

Modo assíncrono: POST /api/transcrever cria um job e devolve job_id; o front
faz polling em GET /api/job/{id}. Assim vídeos longos não estouram o timeout do
proxy. Histórico e modelos ficam em DATA_DIR (volume persistente na VPS).
"""

import json
import os
import tempfile
import time
import glob
import threading
import uuid
from datetime import datetime

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_HTML = os.path.join(BASE_DIR, "index.html")
DATA_DIR = os.environ.get("DATA_DIR", BASE_DIR)
os.makedirs(DATA_DIR, exist_ok=True)
HISTORY_FILE = os.path.join(DATA_DIR, "history.json")

# modelos do whisper no volume persistente (não rebaixar a cada deploy)
os.environ.setdefault("HF_HOME", os.path.join(DATA_DIR, "hf"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

app = FastAPI(title="Transcritor de Vídeos")

_models = {}
_model_lock = threading.Lock()
_jobs = {}
_jobs_lock = threading.Lock()
# 1 transcrição por vez: protege a CPU compartilhada da VPS
_work_sema = threading.Semaphore(1)


def get_model(nome: str):
    from faster_whisper import WhisperModel
    with _model_lock:
        if nome not in _models:
            _models[nome] = WhisperModel(nome, device="cpu", compute_type="int8",
                                         download_root=os.path.join(DATA_DIR, "models"))
        return _models[nome]


def _is_tiktok(url: str) -> bool:
    return any(h in url for h in ("tiktok.com", "tiktok."))


def _tiktok_fallback(url: str, pasta_tmp: str):
    import urllib.request, urllib.parse
    api = "https://www.tikwm.com/api/"
    data = urllib.parse.urlencode({"url": url, "hd": 1}).encode()
    req = urllib.request.Request(api, data=data, headers={
        "User-Agent": "Mozilla/5.0", "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        result = json.loads(resp.read())
    if result.get("code") != 0 or not result.get("data"):
        raise RuntimeError(result.get("msg", "tikwm API error"))
    d = result["data"]
    # "play" = vídeo original com o áudio de fato falado; "music" é só a trilha
    # sonora de fundo (às vezes um som viral reaproveitado, sem relação com a fala)
    audio_url = d.get("play") or d.get("music")
    if not audio_url:
        raise RuntimeError("Nenhum áudio retornado pela API alternativa")
    ext = "mp4" if audio_url == d.get("play") else "mp3"
    out_path = os.path.join(pasta_tmp, f"audio.{ext}")
    req2 = urllib.request.Request(audio_url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req2, timeout=60) as r, open(out_path, "wb") as f:
        f.write(r.read())
    info = {"title": d.get("title", ""), "duration": d.get("duration", 0)}
    return out_path, info


def baixar_audio(url: str, pasta_tmp: str):
    from yt_dlp import YoutubeDL
    saida = os.path.join(pasta_tmp, "audio.%(ext)s")
    cookies_path = os.path.join(DATA_DIR, "cookies.txt")
    opcoes = {
        "format": "bestaudio/best[acodec!=none]/best",
        "outtmpl": saida,
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
    }
    if os.path.exists(cookies_path):
        opcoes["cookiefile"] = cookies_path
    try:
        with YoutubeDL(opcoes) as ydl:
            info = ydl.extract_info(url, download=True)
        arquivos = glob.glob(os.path.join(pasta_tmp, "audio.*"))
        if not arquivos:
            raise FileNotFoundError("Áudio não baixado")
        return arquivos[0], info
    except Exception as e:
        if _is_tiktok(url):
            return _tiktok_fallback(url, pasta_tmp)
        raise


def ler_historico():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def salvar_no_historico(item: dict):
    hist = ler_historico()
    hist.insert(0, item)
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(hist[:500], f, ensure_ascii=False, indent=2)


def set_job(jid, **kw):
    with _jobs_lock:
        _jobs.setdefault(jid, {}).update(kw)


def _processar(jid, url, modelo, idioma, fonte):
    with _work_sema:
        t0 = time.time()
        with tempfile.TemporaryDirectory() as tmp:
            try:
                set_job(jid, status="baixando")
                caminho, info_dl = baixar_audio(url, tmp)
            except Exception as e:
                set_job(jid, status="erro", erro=f"Falha no download: {e}")
                return
            try:
                set_job(jid, status="transcrevendo", titulo=info_dl.get("title") or "")
                model = get_model(modelo)
                lang = None if idioma in ("auto", "", None) else idioma
                segmentos, info = model.transcribe(caminho, language=lang, vad_filter=True)
                texto = " ".join(s.text.strip() for s in segmentos).strip()
            except Exception as e:
                set_job(jid, status="erro", erro=f"Falha na transcrição: {e}")
                return
        dur = float(getattr(info, "duration", 0) or info_dl.get("duration") or 0)
        detected_lang = getattr(info, "language", lang or "?")
        detected_prob = getattr(info, "language_probability", 0) or 0
        item = {
            "url": url, "fonte": fonte, "modelo": modelo,
            "idioma": detected_lang, "confianca": round(float(detected_prob) * 100),
            "duracao_audio": round(dur), "palavras": len(texto.split()),
            "tempo_processo": round(time.time() - t0, 1),
            "titulo": info_dl.get("title") or "", "texto": texto,
            "data": datetime.now().strftime("%d/%m %H:%M"),
        }
        salvar_no_historico(item)
        set_job(jid, status="ok", resultado=item)


class TranscReq(BaseModel):
    url: str
    modelo: str = "small"
    idioma: str = "auto"
    fonte: str = "Outro"


@app.get("/", response_class=HTMLResponse)
def home():
    with open(INDEX_HTML, "r", encoding="utf-8") as f:
        return f.read()


@app.get("/api/health")
def health():
    cookies_path = os.path.join(DATA_DIR, "cookies.txt")
    return {"ok": True, "cookies": os.path.exists(cookies_path)}


@app.post("/api/cookies")
async def upload_cookies(request: Request):
    body = await request.body()
    cookies_path = os.path.join(DATA_DIR, "cookies.txt")
    with open(cookies_path, "wb") as f:
        f.write(body)
    return {"ok": True, "size": len(body)}


@app.get("/api/history")
def api_history():
    return ler_historico()


@app.delete("/api/history")
def api_clear_history():
    if os.path.exists(HISTORY_FILE):
        os.remove(HISTORY_FILE)
    return {"ok": True}


@app.post("/api/transcrever")
def api_transcrever(req: TranscReq):
    if not (req.url or "").strip():
        return JSONResponse({"erro": "Informe o link do vídeo."}, status_code=400)
    jid = uuid.uuid4().hex[:12]
    set_job(jid, status="fila", criado=time.time())
    threading.Thread(target=_processar,
                     args=(jid, req.url.strip(), req.modelo, req.idioma, req.fonte),
                     daemon=True).start()
    return {"job_id": jid}


@app.get("/api/job/{jid}")
def api_job(jid: str):
    with _jobs_lock:
        job = _jobs.get(jid)
    if not job:
        return JSONResponse({"erro": "job não encontrado"}, status_code=404)
    return job


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8830")), log_level="info")
