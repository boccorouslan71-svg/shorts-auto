#!/usr/bin/env python3
"""shorts-auto : vidéo longue 16:9 -> 3 shorts 9:16 sous-titrés + pub, upload Drive, callback Make.

Entrées (variables d'environnement) :
  VIDEO_URL, JOB_ID, MODE (blur|crop), TITLE, SKIP_UPLOAD (test)
Secrets : GROQ_API_KEY, RCLONE_CONFIG_B64, DRIVE_FOLDER_ID, MAKE_CALLBACK_URL
Option : --fallback  => envoie seulement un callback d'erreur (si le script a planté avant).
"""
import base64, json, os, re, shutil, subprocess, sys, time, glob, tempfile
from pathlib import Path

import requests

W, H, FPS = 1080, 1920, 30
NB_PARTS = 3
MAX_PART_SEC = 180
CHUNK_SEC = 600  # découpe audio en morceaux de 10 min
FONT_DIR = "/usr/share/fonts/truetype/dejavu"
FONT_BOLD = FONT_DIR + "/DejaVuSans-Bold.ttf"
ROOT = Path(__file__).resolve().parent
WORK = Path(os.environ.get("WORK_DIR", tempfile.gettempdir())) / "shorts_work"
SENT_FLAG = Path(tempfile.gettempdir()) / "callback_sent"


class StepError(Exception):
    """Erreur avec message clair en français, renvoyée dans le callback."""


def log(*a):
    print(*a, flush=True)


def run(cmd, **kw):
    """Exécute une commande ; ne jamais logguer la ligne complète (peut contenir des secrets)."""
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


# ---------------------------------------------------------------- 1. Téléchargement
def download(url: str) -> Path:
    WORK.mkdir(parents=True, exist_ok=True)
    out = WORK / "source.mp4"
    low = url.lower()
    try:
        if "drive.google.com" in low or "docs.google.com" in low:
            log("Téléchargement : Google Drive (gdown)")
            r = run(["gdown", "--fuzzy", url, "-O", str(out)])
        elif re.search(r"\.(mp4|mov|mkv|webm|avi|m4v)(\?|$)", low):
            log("Téléchargement : lien direct (curl)")
            r = run(["curl", "-L", "--fail", "--retry", "3", "-o", str(out), url])
        else:
            log("Téléchargement : yt-dlp")
            r = run(["yt-dlp", "-f", "bv*[height<=1080]+ba/b[height<=1080]/b", "--merge-output-format", "mp4",
                     "--no-playlist", "-o", str(out), url])
    except FileNotFoundError as e:
        raise StepError(f"téléchargement bloqué : outil manquant ({e.filename})")
    if r.returncode != 0 or not out.exists() or out.stat().st_size < 10_000:
        tail = (r.stderr or "").strip().splitlines()[-1:] or [""]
        log("Détail technique :", tail[0][:200])
        raise StepError("téléchargement bloqué : impossible de récupérer la vidéo (lien invalide, privé ou protégé).")
    # Vérifie que c'est bien une vidéo
    dur = probe_duration(out, strict=False)
    if not dur:
        raise StepError("téléchargement bloqué : le fichier récupéré n'est pas une vidéo lisible.")
    return out


def probe_duration(path: Path, strict=True):
    r = run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)])
    try:
        d = float(json.loads(r.stdout)["format"]["duration"])
        return d if d > 0 else None
    except Exception:
        if strict:
            raise StepError(f"fichier illisible : {path.name}")
        return None


def probe_video(path: Path):
    r = run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
             "-of", "json", str(path)])
    s = json.loads(r.stdout)["streams"][0]
    return int(s["width"]), int(s["height"])


def has_audio(path: Path) -> bool:
    r = run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index", "-of", "csv=p=0", str(path)])
    return bool(r.stdout.strip())


# ---------------------------------------------------------------- 3. Transcription
def extract_chunks(src: Path, total: float):
    chunks = []
    n = int(total // CHUNK_SEC) + 1
    for i in range(n):
        start = i * CHUNK_SEC
        if start >= total:
            break
        f = WORK / f"audio_{i}.mp3"
        r = run(["ffmpeg", "-y", "-v", "error", "-ss", str(start), "-t", str(CHUNK_SEC), "-i", str(src),
                 "-vn", "-ac", "1", "-ar", "16000", "-b:a", "32k", str(f)])
        if r.returncode != 0 or not f.exists():
            raise StepError("extraction audio impossible (la vidéo n'a peut-être pas de son).")
        chunks.append((start, f))
    return chunks


def groq_transcribe(path: Path, key: str):
    """Retourne la liste de mots [{word,start,end}] pour un morceau. Retry si limite de débit."""
    for attempt in range(5):
        with open(path, "rb") as fh:
            resp = requests.post(
                "https://api.groq.com/openai/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {key}"},
                files={"file": (path.name, fh, "audio/mpeg")},
                data=[("model", "whisper-large-v3"), ("language", "fr"), ("response_format", "verbose_json"),
                      ("timestamp_granularities[]", "word")],
                timeout=300,
            )
        if resp.status_code == 429 or resp.status_code >= 500:
            wait = min(float(resp.headers.get("retry-after", 15 * (attempt + 1))), 120)
            log(f"Groq limite/erreur {resp.status_code}, attente {wait:.0f}s")
            time.sleep(wait)
            continue
        if resp.status_code != 200:
            raise RuntimeError(f"Groq HTTP {resp.status_code}")
        return resp.json().get("words", [])
    raise RuntimeError("Groq : trop de tentatives")


def local_transcribe(path: Path):
    """Fallback : faster-whisper small sur le runner."""
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        log("Installation de faster-whisper…")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "faster-whisper"], check=True)
        from faster_whisper import WhisperModel
    model = WhisperModel("small", device="cpu", compute_type="int8")
    segs, _ = model.transcribe(str(path), language="fr", word_timestamps=True)
    return [{"word": w.word.strip(), "start": w.start, "end": w.end} for s in segs for w in (s.words or [])]


def transcribe(src: Path, total: float, warnings: list):
    chunks = extract_chunks(src, total)
    key = os.environ.get("GROQ_API_KEY", "").strip()
    words = []
    use_groq = bool(key)
    if not use_groq:
        warnings.append("GROQ_API_KEY absent : transcription locale (faster-whisper).")
    for start, f in chunks:
        part_words = None
        if use_groq:
            try:
                part_words = groq_transcribe(f, key)
            except Exception as e:
                log(f"Groq a échoué ({type(e).__name__}: {e}) -> bascule sur faster-whisper")
                warnings.append("Groq indisponible : transcription locale (faster-whisper) utilisée.")
                use_groq = False
        if part_words is None:
            try:
                part_words = local_transcribe(f)
            except Exception as e:
                raise StepError(f"transcription impossible (Groq et faster-whisper ont échoué) : {type(e).__name__}")
        for w in part_words:
            words.append({"word": str(w["word"]).strip(), "start": float(w["start"]) + start,
                          "end": float(w["end"]) + start})
    words = [w for w in words if w["word"]]
    if not words:
        warnings.append("Aucune parole détectée : vidéo sans sous-titres.")
    return words


# ---------------------------------------------------------------- Sous-titres ASS
def ts(t: float) -> str:
    t = max(t, 0)
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h)}:{int(m):02d}:{s:05.2f}"


def make_ass(words, t0, t1, path: Path):
    """Sous-titres de la partie [t0,t1) : groupes de 3 à 5 mots, blanc gras contour noir, tiers bas."""
    ws = [w for w in words if t0 <= w["start"] < t1]
    groups, cur = [], []
    for w in ws:
        if cur and (len(cur) >= 4 or w["start"] - cur[-1]["end"] > 0.8):
            groups.append(cur)
            cur = []
        cur.append(w)
    if cur:
        groups.append(cur)
    # évite un groupe final de 1-2 mots isolés : fusionne avec le précédent si <=5 mots au total
    if len(groups) >= 2 and len(groups[-1]) < 3 and len(groups[-2]) + len(groups[-1]) <= 5 \
            and groups[-1][0]["start"] - groups[-2][-1]["end"] <= 0.8:
        groups[-2] += groups.pop()
    head = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {W}
PlayResY: {H}
WrapStyle: 0

[V4+ Styles]
Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding
Style: Default,DejaVu Sans,72,&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,6,0,2,90,90,430,1

[Events]
Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text
"""
    lines = []
    for i, g in enumerate(groups):
        s = g[0]["start"] - t0
        e = g[-1]["end"] - t0 + 0.25
        if i + 1 < len(groups):
            e = min(e, groups[i + 1][0]["start"] - t0 - 0.02)
        e = max(e, s + 0.3)
        e = min(e, t1 - t0)
        text = " ".join(x["word"] for x in g).replace("{", "(").replace("}", ")")
        lines.append(f"Dialogue: 0,{ts(s)},{ts(e)},Default,,0,0,0,,{text}")
    path.write_text(head + "\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- 4-6. Montage
ENC = ["-c:v", "libx264", "-crf", "23", "-preset", "veryfast", "-pix_fmt", "yuv420p", "-r", str(FPS),
       "-c:a", "aac", "-b:a", "128k", "-ar", "48000", "-ac", "2", "-movflags", "+faststart"]
LOUD = "loudnorm=I=-16:TP=-1.5:LRA=11"


def video_chain(mode: str, badge: str | None, ass: str | None) -> str:
    if mode == "crop":
        f = f"[0:v]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},setsar=1"
    else:
        f = (f"[0:v]split[a][b];[a]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},"
             f"boxblur=40:5[bg];[b]scale={W}:-2[fg];[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1")
    f += f",fps={FPS}"
    if ass:
        f += f",subtitles={ass}:fontsdir={FONT_DIR}"
    if badge:
        f += (f",drawtext=fontfile={FONT_BOLD}:text='{badge}':fontcolor=white:fontsize=48:"
              f"box=1:boxcolor=black@0.55:boxborderw=18:x=(w-text_w)/2:y=110")
    return f + "[v]"


def encode_part(src: Path, t0: float, dur: float, mode: str, badge: str, ass: str, out: Path):
    # Entrée en -ss avant -i (rapide) ; le re-encodage garantit la précision.
    fc = video_chain(mode, badge, ass)
    cmd = ["ffmpeg", "-y", "-v", "error", "-ss", f"{t0:.3f}", "-t", f"{dur:.3f}", "-i", str(src)]
    if has_audio(src):
        cmd += ["-filter_complex", fc + f";[0:a]{LOUD}[a]", "-map", "[v]", "-map", "[a]"]
    else:
        cmd += ["-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo", "-filter_complex", fc,
                "-map", "[v]", "-map", "1:a", "-shortest"]
    cmd += ENC + [str(out)]
    r = run(cmd, cwd=WORK)
    if r.returncode != 0:
        log("ffmpeg:", r.stderr[-600:])
        raise StepError("erreur FFmpeg pendant le montage d'une partie.")


def normalize_ad(ad: Path, out: Path):
    w, h = probe_video(ad)
    vf = (f"scale={W}:{H}:force_original_aspect_ratio=decrease,pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:black,"
          f"setsar=1,fps={FPS}")
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(ad)]
    if has_audio(ad):
        cmd += ["-vf", vf, "-af", LOUD]
    else:
        cmd += ["-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo", "-vf", vf, "-map", "0:v", "-map", "1:a", "-shortest"]
    cmd += ENC + [str(out)]
    r = run(cmd)
    if r.returncode != 0:
        log("ffmpeg pub:", r.stderr[-600:])
        raise StepError("erreur FFmpeg pendant la normalisation de la pub.")


def concat(a: Path, b: Path, out: Path):
    lst = WORK / "concat.txt"
    lst.write_text(f"file '{a}'\nfile '{b}'\n")
    r = run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy",
             "-movflags", "+faststart", str(out)])
    if r.returncode != 0:
        raise StepError("erreur FFmpeg pendant l'ajout de la pub.")


def verify_output(p: Path) -> float:
    if not p.exists() or p.stat().st_size < 10_000:
        raise StepError(f"fichier final vide ou absent : {p.name}")
    dur = probe_duration(p)
    w, h = probe_video(p)
    if (w, h) != (W, H):
        raise StepError(f"format final incorrect pour {p.name} : {w}x{h}")
    return round(dur, 1)


# ---------------------------------------------------------------- 7. Drive
def upload_drive(files, job_id, folder_id):
    b64 = os.environ.get("RCLONE_CONFIG_B64", "").strip()
    if not b64 or not folder_id:
        raise StepError("Google Drive non configuré : secrets RCLONE_CONFIG_B64 et/ou DRIVE_FOLDER_ID manquants.")
    cfg = WORK / "rclone.conf"
    try:
        cfg.write_bytes(base64.b64decode(b64))
    except Exception:
        raise StepError("RCLONE_CONFIG_B64 invalide (ce n'est pas du base64 correct).")
    cfg.chmod(0o600)
    m = re.search(r"^\[([^\]]+)\]", cfg.read_text(), re.M)
    if not m:
        raise StepError("RCLONE_CONFIG_B64 invalide (aucun remote trouvé).")
    remote = m.group(1)
    base = ["rclone", "--config", str(cfg), "--drive-root-folder-id", folder_id]
    ids = {}
    for f in files:
        for attempt in range(3):
            r = run(base + ["copyto", str(f), f"{remote}:{job_id}/{f.name}"])
            if r.returncode == 0:
                break
            time.sleep(5 * (attempt + 1))
        else:
            log("rclone:", re.sub(r"[A-Za-z0-9_\-]{25,}", "***", r.stderr)[-300:])
            raise StepError("échec de l'envoi vers Google Drive (autorisation expirée ou dossier introuvable ?).")
    r = run(base + ["lsjson", f"{remote}:{job_id}"])
    if r.returncode != 0:
        raise StepError("impossible de relire les identifiants des fichiers sur Drive.")
    for item in json.loads(r.stdout):
        ids[item["Name"]] = item["ID"]
    for f in files:
        if f.name not in ids:
            raise StepError(f"fichier absent de Drive après envoi : {f.name}")
    return ids


# ---------------------------------------------------------------- 8. Callback
def callback(payload: dict):
    url = os.environ.get("MAKE_CALLBACK_URL", "").strip()
    if not url:
        log("MAKE_CALLBACK_URL vide : callback ignoré.")
        SENT_FLAG.write_text("skipped")
        return
    for attempt in range(3):
        try:
            r = requests.post(url, json=payload, timeout=30)
            if r.status_code < 300:
                log(f"Callback envoyé (HTTP {r.status_code}).")
                SENT_FLAG.write_text("ok")
                return
            log(f"Callback HTTP {r.status_code}")
        except requests.RequestException as e:
            log(f"Callback erreur réseau : {type(e).__name__}")
        time.sleep(5 * (attempt + 1) ** 2)
    log("Callback : 3 échecs.")


# ---------------------------------------------------------------- Principal
def process(env, warnings):
    src = download(env["video_url"])
    total = probe_duration(src)
    log(f"Durée source : {total:.1f}s")
    if total < 9:
        raise StepError("vidéo trop courte pour être découpée en 3 parties.")
    words = transcribe(src, total, warnings)
    seg = total / NB_PARTS
    ad_raw = ROOT / "assets" / "pub.mp4"
    ad = None
    if ad_raw.exists() and ad_raw.stat().st_size > 0:
        ad = WORK / "pub_norm.mp4"
        normalize_ad(ad_raw, ad)
    else:
        warnings.append("assets/pub.mp4 absent : pub non ajoutée.")
    finals = []
    for i in range(NB_PARTS):
        t0, t1 = i * seg, (i + 1) * seg
        if seg > MAX_PART_SEC:
            if i == 0:
                warnings.append(f"Chaque partie dure {seg:.0f}s (> {MAX_PART_SEC}s).")
        ass = WORK / f"p{i+1}.ass"
        make_ass(words, t0, t1, ass)
        name = f"{env['job_id']}_partie{i+1}.mp4"
        body = WORK / f"body{i+1}.mp4"
        encode_part(src, t0, seg, env["mode"], f"Partie {i+1}/{NB_PARTS}", ass.name, body)
        final = WORK / name
        if ad:
            concat(body, ad, final)
        else:
            shutil.copy(body, final)
        dur = verify_output(final)
        log(f"{name} : OK ({dur}s, {W}x{H})")
        finals.append((i + 1, final, dur))
    if env["skip_upload"]:
        warnings.append("Mode test : upload Drive ignoré.")
        ids = {f.name: "TEST_NO_UPLOAD" for _, f, _ in finals}
    else:
        ids = upload_drive([f for _, f, _ in finals], env["job_id"], os.environ.get("DRIVE_FOLDER_ID", "").strip())
    return [{"index": i, "drive_file_id": ids[f.name], "filename": f.name, "duration_sec": d} for i, f, d in finals]


def main():
    job_id = os.environ.get("JOB_ID", "").strip()
    payload = {"job_id": job_id, "status": "error", "source_url": os.environ.get("VIDEO_URL", ""),
               "title": os.environ.get("TITLE", ""), "parts": [], "warnings": [], "error": None}
    if "--fallback" in sys.argv:
        if not SENT_FLAG.exists():
            payload["error"] = "Le traitement a planté avant la fin (voir les logs GitHub Actions)."
            callback(payload)
        return
    warnings = payload["warnings"]
    code = 0
    try:
        mode = os.environ.get("MODE", "blur").strip().lower()
        if mode not in ("blur", "crop"):
            warnings.append(f"mode '{mode}' inconnu, 'blur' utilisé.")
            mode = "blur"
        if not job_id or not payload["source_url"]:
            raise StepError("paramètres manquants : video_url et job_id sont obligatoires.")
        env = {"video_url": payload["source_url"], "job_id": job_id, "mode": mode,
               "skip_upload": os.environ.get("SKIP_UPLOAD", "").lower() == "true"}
        payload["parts"] = process(env, warnings)
        payload["status"] = "success"
    except StepError as e:
        payload["error"] = f"Erreur : {e}"
        payload["parts"] = []
        code = 1
    except Exception as e:  # filet de sécurité global
        payload["error"] = f"Erreur inattendue ({type(e).__name__}) pendant le traitement."
        payload["parts"] = []
        code = 1
        log("Exception:", repr(e)[:300])
    finally:
        shutil.rmtree(WORK, ignore_errors=True)  # nettoyage (inclut la config rclone)
    callback(payload)
    log("Résultat :", json.dumps({k: v for k, v in payload.items() if k != "source_url"}, ensure_ascii=False))
    sys.exit(code)


if __name__ == "__main__":
    main()
