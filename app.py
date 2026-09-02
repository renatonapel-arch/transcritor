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
import re
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
CODIGO_FILE = os.path.join(DATA_DIR, "contador_codigo.json")

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


def _tem_audio(caminho: str) -> bool:
    import subprocess
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
             "stream=index", "-of", "csv=p=0", caminho],
            capture_output=True, text=True, timeout=15)
        return bool(r.stdout.strip())
    except Exception:
        return True  # ffprobe indisponível: não bloqueia o fluxo normal


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

    if d.get("images"):
        # carrossel de fotos: não existe fala pra transcrever. O "play"/"music"
        # aqui é só a trilha sonora do post — baixar e mandar pro Whisper
        # transcrevia LETRA DE MÚSICA como se fosse o vídeo falando (bug real,
        # já gerou veredito errado no item #0088 antes desta correção).
        info = {"title": d.get("title", ""), "duration": 0,
                "tipo_post": "carrossel", "imagens_n": len(d["images"]),
                "_tikwm": d}
        return None, info

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
    info = {"title": d.get("title", ""), "duration": d.get("duration", 0), "_tikwm": d}
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
        # TikTok às vezes reporta acodec=aac mas entrega um arquivo só de vídeo
        if not _tem_audio(arquivos[0]):
            raise RuntimeError("Arquivo baixado não tem faixa de áudio")
        return arquivos[0], info
    except Exception:
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


def _ler_contador_codigo():
    if os.path.exists(CODIGO_FILE):
        try:
            with open(CODIGO_FILE, "r", encoding="utf-8") as f:
                return json.load(f).get("proximo", 1)
        except Exception:
            pass
    return 1


def _salvar_contador_codigo(n: int):
    with open(CODIGO_FILE, "w", encoding="utf-8") as f:
        json.dump({"proximo": n}, f)


def proximo_codigo() -> str:
    n = _ler_contador_codigo()
    _salvar_contador_codigo(n + 1)
    return f"#{n:04d}"


def migrar_codigos_existentes():
    """Dá #NNNN pra item do histórico que ainda não tem. Roda 1x no start,
    idempotente — item que já tem codigo nunca é tocado de novo. Não depende
    do campo 'data' (string 'dd/mm HH:MM' sem ano, não ordena bem sozinha);
    usa a ordem da própria lista, que salvar_no_historico já mantém do mais
    novo pro mais antigo (insert(0, ...)) — então o mais antigo sem código
    recebe o próximo número disponível."""
    hist = ler_historico()
    if not hist or all(h.get("codigo") for h in hist):
        return
    n = _ler_contador_codigo()
    mudou = False
    for h in reversed(hist):
        if not h.get("codigo"):
            h["codigo"] = f"#{n:04d}"
            n += 1
            mudou = True
    if mudou:
        _salvar_contador_codigo(n)
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(hist, f, ensure_ascii=False, indent=2)


migrar_codigos_existentes()


def set_job(jid, **kw):
    with _jobs_lock:
        _jobs.setdefault(jid, {}).update(kw)


def extrair_metadados(info_dl: dict) -> dict:
    """Metadados do post, venha do yt-dlp (campos nativos de info_dl) ou do
    fallback tikwm (dict bruto guardado em info_dl['_tikwm']). Cada campo
    tenta as duas fontes, na ordem que a sondagem confirmou existir.
    NUNCA grava URL de mídia (play/music/cover/avatar) — são links
    assinados que expiram em 24-48h e viram link morto no histórico."""
    tk = info_dl.get("_tikwm") or {}
    music = tk.get("music_info") or {}
    author = tk.get("author") or {}

    titulo = info_dl.get("title") or ""
    if not titulo and tk.get("content_desc"):
        titulo = "\n".join(tk["content_desc"])
    if not titulo:
        titulo = info_dl.get("description") or ""

    return {
        "titulo": titulo,
        "autor": info_dl.get("uploader") or author.get("unique_id") or "",
        "data_publicacao": info_dl.get("timestamp") or tk.get("create_time"),
        "views": info_dl.get("view_count", tk.get("play_count")),
        "likes": info_dl.get("like_count", tk.get("digg_count")),
        "salvos": info_dl.get("save_count", tk.get("collect_count") or tk.get("download_count")),
        "som_titulo": info_dl.get("track") or music.get("title") or "",
        "som_autor": info_dl.get("artist") or music.get("author") or "",
        # is_ad só existe quando a fonte foi o fallback tikwm; yt-dlp não expõe
        # esse campo pro extrator de TikTok — fica None nesse caso, não False
        "is_ad": tk.get("is_ad"),
        "hashtags": re.findall(r"#(\w+)", titulo),
    }


def ocr_carrossel(image_urls: list) -> str:
    """Manda as fotos do carrossel pro Gemini ler o texto de cada uma
    (item 4 do roadmap de 01/09/2026 — ligado só pra carrossel, não pra
    vídeo, que teve OCR partido entre modelos no teste). Falha SEMPRE em
    silêncio: sem chave configurada, erro de rede ou de API → volta "" e o
    julgamento cai pra legenda, que já funcionava antes desta função
    existir. Custo medido em teste real: ~US$0,024 por carrossel de 12
    fotos (gemini-3.7-flash, tabela de preço de 01/09/2026)."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key or not image_urls:
        return ""
    import base64
    import urllib.request

    parts = [{"text": (
        "Isto e um carrossel de fotos de rede social, com "
        f"{len(image_urls)} imagens, na ordem em que aparecem. Para CADA "
        "imagem, leia todo texto visivel nela (titulo, numeros, legenda, "
        "qualquer coisa escrita). Responda em texto simples, uma linha por "
        "imagem, no formato 'Imagem N: <texto lido>' — ou 'Imagem N: (sem "
        "texto legivel)' se não houver nada escrito. Não invente texto que "
        "não esteja de fato visível na imagem."
    )}]
    for img_url in image_urls:
        try:
            req = urllib.request.Request(img_url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=20) as r:
                img = r.read()
            parts.append({"inline_data": {"mime_type": "image/jpeg",
                                          "data": base64.b64encode(img).decode()}})
        except Exception:
            continue  # 1 foto que falhou não derruba as outras 11
    if len(parts) <= 1:
        return ""

    try:
        body = json.dumps({"contents": [{"parts": parts}]}).encode()
        req = urllib.request.Request(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.7-flash:generateContent",
            data=body, headers={"Content-Type": "application/json", "x-goog-api-key": api_key})
        with urllib.request.urlopen(req, timeout=90) as resp:
            r = json.loads(resp.read())
        u = r.get("usageMetadata", {})
        print(f"[ocr_carrossel] {len(parts) - 1} fotos · tokens entrada="
              f"{u.get('promptTokenCount')} saida={u.get('candidatesTokenCount')} "
              f"pensamento={u.get('thoughtsTokenCount')}", flush=True)
        return r["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception as e:
        print(f"[ocr_carrossel] falhou, seguindo só com a legenda: {e}", flush=True)
        return ""


def buscar_comentarios(url: str, count: int = 20) -> dict:
    """1 página de comentários via tikwm (item 3 do roadmap, 01/09/2026).
    Falha SEMPRE em silêncio: sem comentário aqui NUNCA pode virar "post
    sem reação" no julgamento — é só "a busca não trouxe nada dessa vez"
    (o tikwm pode estar fora do ar, sem fonte reserva pra isso).
    Coleta é sabidamente PARCIAL (medido: total 26 → só 19 devolvidos) —
    daí o campo comentarios_incompleto, pra nunca tratar como lista fechada.
    NUNCA grava identidade de quem comentou (sec_uid, unique_id, avatar,
    região) — só o texto e os números. Quem comentou não pediu pra
    aparecer na triagem de vídeo do Renato; LGPD."""
    import urllib.parse
    import urllib.request
    try:
        body = urllib.parse.urlencode({"url": url, "count": count, "cursor": 0}).encode()
        req = urllib.request.Request(
            "https://www.tikwm.com/api/comment/list", data=body,
            headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            r = json.loads(resp.read())
        if r.get("code") != 0:
            raise RuntimeError(r.get("msg", "tikwm comment API error"))
        data = r.get("data") or {}
        brutos = data.get("comments") or []
        # ordem que a API devolveu, NUNCA reordenar por curtida — o
        # comentário mais útil visto num teste real tinha 2 curtidas, e o
        # mais curtido (107) era piada
        comentarios = [{
            "texto": (c.get("text") or "")[:200],
            "likes": c.get("digg_count"),
            "respostas": c.get("reply_total"),
            "data": c.get("create_time"),
        } for c in brutos]
        total = data.get("total")
        return {
            "comentarios": comentarios,
            "comentarios_total": total,
            "comentarios_incompleto": bool(total and total > len(comentarios)),
        }
    except Exception as e:
        print(f"[buscar_comentarios] falhou, seguindo sem comentário: {e}", flush=True)
        return {"comentarios": [], "comentarios_total": None, "comentarios_incompleto": None}


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

            meta = extrair_metadados(info_dl)
            carrossel = info_dl.get("tipo_post") == "carrossel"
            texto_imagens = ""
            try:
                set_job(jid, status="transcrevendo", titulo=meta["titulo"])
                if carrossel:
                    # sem áudio de fala pra transcrever — ver _tiktok_fallback
                    texto, dur, detected_lang, detected_prob = "", 0, "", 0
                    imagens = (info_dl.get("_tikwm") or {}).get("images") or []
                    if imagens:
                        set_job(jid, status="lendo_fotos")
                        texto_imagens = ocr_carrossel(imagens)
                else:
                    model = get_model(modelo)
                    lang = None if idioma in ("auto", "", None) else idioma
                    segmentos, info = model.transcribe(caminho, language=lang, vad_filter=True)
                    texto = " ".join(s.text.strip() for s in segmentos).strip()
                    dur = float(getattr(info, "duration", 0) or info_dl.get("duration") or 0)
                    detected_lang = getattr(info, "language", lang or "?")
                    detected_prob = getattr(info, "language_probability", 0) or 0
            except Exception as e:
                set_job(jid, status="erro", erro=f"Falha na transcrição: {e}")
                return

            set_job(jid, status="lendo_comentarios")
            if info_dl.get("_tikwm"):
                time.sleep(1.1)  # tikwm: 1 req/s — já chamamos ele acima pra baixar a mídia
            comentarios_info = buscar_comentarios(url)
        item = {
            "codigo": proximo_codigo(),
            "url": url, "fonte": fonte, "modelo": modelo,
            "idioma": detected_lang, "confianca": round(float(detected_prob) * 100),
            "duracao_audio": round(dur), "palavras": len(texto.split()),
            "tempo_processo": round(time.time() - t0, 1),
            "texto": texto,
            "texto_imagens": texto_imagens,
            "tipo_post": "carrossel" if carrossel else "video",
            "imagens_n": info_dl.get("imagens_n") if carrossel else None,
            **meta,
            **comentarios_info,
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
