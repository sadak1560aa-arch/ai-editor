# ai_editor_app.py

import os

# ⚠️ يجب ضبط التوكن قبل استيراد replicate
os.environ.setdefault(
    "REPLICATE_API_TOKEN",
    os.getenv("REPLICATE_API_TOKEN", "")
)

import uuid
import io
import subprocess
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from contextlib import asynccontextmanager, contextmanager
import sqlite3
import asyncio

import cv2
import numpy as np
import requests
import replicate
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn

# ==========================================================
# المسارات والإعدادات
# ==========================================================
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_IMAGE_TYPES = {"image/png", "image/jpeg", "image/webp"}
ALLOWED_VIDEO_TYPES = {"video/mp4", "video/quicktime", "video/x-msvideo"}
MAX_IMAGE_BYTES = 25 * 1024 * 1024   # 25 MB
MAX_VIDEO_BYTES = 200 * 1024 * 1024  # 200 MB

MODEL_REMBG = "cjwbw/rembg:fb8af171cfa1616ddcf1242c093f9c46bcada5ad4cf6f2fbe8b81b330ec5c003"
MODEL_SDXL = "stability-ai/sdxl:7762fd07cf82c948538e41f63f77d685e02b063e37e496e96eefd46c929f9bdc"
MODEL_DEOLDIFY = "arielreplicate/deoldify_image:4eb8cb11d135a066eb7dee32e4858b0246c00f50a0a1c7ce571b3a3faea61ebe"

# ==========================================================
# قاعدة البيانات
# ==========================================================
DB_PATH = BASE_DIR / "ai_editor.db"

@contextmanager
def db_conn():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()

def init_db() -> None:
    with db_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_type TEXT NOT NULL,
                file_id TEXT NOT NULL,
                input_path TEXT,
                output_path TEXT,
                output_url TEXT,
                prompt TEXT,
                status TEXT NOT NULL DEFAULT 'success',
                error TEXT,
                created_at TEXT NOT NULL
            )
        """)

def log_job(job_type: str, file_id: str, output_url: str = "", *, input_path: Path | None = None, output_path: Path | None = None, prompt: str | None = None, status: str = "success", error: str | None = None) -> None:
    try:
        with db_conn() as conn:
            conn.execute(
                "INSERT INTO jobs (job_type, file_id, input_path, output_path, output_url, prompt, status, error, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (job_type, file_id, str(input_path) if input_path else None, str(output_path) if output_path else None, output_url, prompt, status, error, datetime.utcnow().isoformat(timespec="seconds")),
            )
    except Exception as exc:
        print(f"[DB] فشل تسجيل المهمة: {exc}")

# ==========================================================
# التنظيف
# ==========================================================
CLEANUP_MAX_AGE_HOURS = 24
CLEANUP_INTERVAL_SECONDS = 3600

def cleanup_old_files(max_age_hours: int = CLEANUP_MAX_AGE_HOURS) -> dict:
    cutoff = datetime.utcnow() - timedelta(hours=max_age_hours)
    cutoff_ts = cutoff.timestamp()
    deleted = {"uploads": 0, "outputs": 0, "db_rows": 0}
    for directory, key in ((UPLOAD_DIR, "uploads"), (OUTPUT_DIR, "outputs")):
        if not directory.exists(): continue
        for f in directory.iterdir():
            if f.is_file() and f.stat().st_mtime < cutoff_ts:
                try:
                    f.unlink()
                    deleted[key] += 1
                except OSError: pass
    with db_conn() as conn:
        cur = conn.execute("DELETE FROM jobs WHERE created_at < ?", (cutoff.isoformat(timespec="seconds"),))
        deleted["db_rows"] = cur.rowcount
    return deleted

async def cleanup_loop() -> None:
    while True:
        try:
            await asyncio.sleep(CLEANUP_INTERVAL_SECONDS)
            result = await asyncio.to_thread(cleanup_old_files)
            print(f"[CLEANUP] {result}")
        except asyncio.CancelledError: break
        except Exception as exc: print(f"[CLEANUP] خطأ: {exc}")

_cleanup_task: asyncio.Task | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _cleanup_task
    init_db()
    _cleanup_task = asyncio.create_task(cleanup_loop())
    try: yield
    finally:
        if _cleanup_task:
            _cleanup_task.cancel()
            try: await _cleanup_task
            except asyncio.CancelledError: pass

# ==========================================================
# التطبيق
# ==========================================================
app = FastAPI(title="AI Editor", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")

# ==========================================================
# دوال مساعدة
# ==========================================================
def _save_upload(file: UploadFile, allowed_types: set, max_bytes: int) -> tuple[str, Path]:
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail=f"نوع الملف غير مدعوم: {file.content_type}")
    data = file.file.read()
    if len(data) > max_bytes:
        raise HTTPException(status_code=413, detail=f"حجم الملف يتجاوز الحد المسموح.")
    file_id = str(uuid.uuid4())
    ext = Path(file.filename or "upload").suffix or ".bin"
    path = UPLOAD_DIR / f"{file_id}{ext}"
    path.write_bytes(data)
    return file_id, path

def _replicate_output_to_pil(output) -> Image.Image:
    if isinstance(output, list):
        if not output: raise HTTPException(502, "لم يُرجع النموذج أي مخرجات.")
        output = output[0]
    if hasattr(output, "read"):
        data = output.read()
        if isinstance(data, bytes): return Image.open(io.BytesIO(data))
        return Image.open(data)
    if isinstance(output, str):
        resp = requests.get(output, timeout=120)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content))
    if isinstance(output, Image.Image): return output
    raise HTTPException(502, f"نوع مخرجات غير متوقع: {type(output)}")

def _run_ffmpeg(cmd: list[str]) -> None:
    if shutil.which("ffmpeg") is None:
        raise HTTPException(500, "ffmpeg غير مثبّت على الخادم.")
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        raise HTTPException(500, f"فشل ffmpeg: {exc.stderr or exc.stdout or exc}") from exc

# ==========================================================
# نقاط النهاية
# ==========================================================
@app.post("/api/enhance")
def enhance_image(file: UploadFile = File(...)):
    file_id, path = _save_upload(file, ALLOWED_IMAGE_TYPES, MAX_IMAGE_BYTES)
    img = cv2.imread(str(path))
    if img is None: raise HTTPException(400, "تعذّر قراءة الصورة.")
    img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    enhanced = cv2.resize(enhanced, None, fx=2, fy=2, interpolation=cv2.INTER_LANCZOS4)
    out_path = OUTPUT_DIR / f"{file_id}_enhanced.png"
    if not cv2.imwrite(str(out_path), enhanced): raise HTTPException(500, "فشل حفظ الصورة.")
    output_url = f"/outputs/{out_path.name}"
    log_job("enhance", file_id, output_url, input_path=path, output_path=out_path)
    return {"success": True, "url": output_url}

@app.post("/api/remove-background")
def remove_background(file: UploadFile = File(...)):
    file_id, path = _save_upload(file, ALLOWED_IMAGE_TYPES, MAX_IMAGE_BYTES)
    try:
        with open(path, "rb") as fh:
            output = replicate.run(MODEL_REMBG, input={"image": fh})
    except Exception as exc:
        log_job("remove-background", file_id, input_path=path, status="error", error=str(exc))
        raise HTTPException(502, f"فشل النموذج: {exc}")
    img = _replicate_output_to_pil(output)
    out_path = OUTPUT_DIR / f"{file_id}_nobg.png"
    img.save(out_path, format="PNG")
    output_url = f"/outputs/{out_path.name}"
    log_job("remove-background", file_id, output_url, input_path=path, output_path=out_path)
    return {"success": True, "url": output_url}

@app.post("/api/generate")
def generate_image(prompt: str = Form(...)):
    if not prompt.strip(): raise HTTPException(400, "الوصف النصي مطلوب.")
    try:
        output = replicate.run(MODEL_SDXL, input={"prompt": prompt.strip(), "width": 1024, "height": 1024, "num_outputs": 1, "num_inference_steps": 30, "guidance_scale": 7.5}, timeout=3600)
    except Exception as exc:
        log_job("generate", "", prompt=prompt.strip(), status="error", error=str(exc))
        raise HTTPException(502, f"فشل النموذج: {exc}")
    img = _replicate_output_to_pil(output)
    file_id = str(uuid.uuid4())
    out_path = OUTPUT_DIR / f"{file_id}.png"
    img.save(out_path, format="PNG")
    output_url = f"/outputs/{out_path.name}"
    log_job("generate", file_id, output_url, output_path=out_path, prompt=prompt.strip())
    return {"success": True, "url": output_url}

@app.post("/api/colorize")
def colorize_image(file: UploadFile = File(...)):
    file_id, path = _save_upload(file, ALLOWED_IMAGE_TYPES, MAX_IMAGE_BYTES)
    try:
        with open(path, "rb") as fh:
            output = replicate.run(MODEL_DEOLDIFY, input={"image": fh})
    except Exception as exc:
        log_job("colorize", file_id, input_path=path, status="error", error=str(exc))
        raise HTTPException(502, f"فشل النموذج: {exc}")
    img = _replicate_output_to_pil(output)
    out_path = OUTPUT_DIR / f"{file_id}_color.png"
    img.save(out_path, format="PNG")
    output_url = f"/outputs/{out_path.name}"
    log_job("colorize", file_id, output_url, input_path=path, output_path=out_path)
    return {"success": True, "url": output_url}

@app.post("/api/video-upscale")
def video_upscale(file: UploadFile = File(...)):
    file_id, in_path = _save_upload(file, ALLOWED_VIDEO_TYPES, MAX_VIDEO_BYTES)
    silent_out = OUTPUT_DIR / f"{file_id}_silent.mp4"
    final_out = OUTPUT_DIR / f"{file_id}_hq.mp4"
    _run_ffmpeg(["ffmpeg", "-y", "-i", str(in_path), "-vf", "scale=iw*2:ih*2:flags=lanczos,hqdn3d=1.5:1.5:6:6", "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "20", str(silent_out)])
    _run_ffmpeg(["ffmpeg", "-y", "-i", str(silent_out), "-i", str(in_path), "-c:v", "copy", "-c:a", "aac", "-map", "0:v:0", "-map", "1:a:0?", "-shortest", str(final_out)])
    silent_out.unlink(missing_ok=True)
    output_url = f"/outputs/{final_out.name}"
    log_job("video-upscale", file_id, output_url, input_path=in_path, output_path=final_out)
    return {"success": True, "url": output_url}

@app.get("/api/history")
def history(limit: int = 20, job_type: str | None = None):
    limit = max(1, min(limit, 100))
    with db_conn() as conn:
        if job_type: rows = conn.execute("SELECT * FROM jobs WHERE job_type = ? ORDER BY id DESC LIMIT ?", (job_type, limit)).fetchall()
        else: rows = conn.execute("SELECT * FROM jobs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return {"success": True, "count": len(rows), "jobs": [dict(r) for r in rows]}

@app.get("/api/status")
def status():
    def _count(directory: Path) -> int: return sum(1 for f in directory.iterdir() if f.is_file()) if directory.exists() else 0
    with db_conn() as conn:
        jobs_total = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
        jobs_today = conn.execute("SELECT COUNT(*) FROM jobs WHERE date(created_at) = date('now')").fetchone()[0]
        jobs_errors = conn.execute("SELECT COUNT(*) FROM jobs WHERE status = 'error'").fetchone()[0]
    return {"success": True, "uploads_files": _count(UPLOAD_DIR), "outputs_files": _count(OUTPUT_DIR), "jobs_total": jobs_total, "jobs_today": jobs_today, "jobs_errors": jobs_errors}

@app.post("/api/cleanup")
def manual_cleanup(max_age_hours: int = CLEANUP_MAX_AGE_HOURS):
    if max_age_hours < 1: raise HTTPException(400, "max_age_hours يجب أن يكون 1 على الأقل.")
    return {"success": True, "deleted": cleanup_old_files(max_age_hours)}

HTML_PAGE = """
<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>AI Studio</title>
  <style>
    :root { color-scheme: dark; }
    * { box-sizing: border-box; }
    body { font-family: 'Segoe UI', Tahoma, sans-serif; background: #0f0f1a; color: #e0e0e0; margin: 0; padding: 1.5rem; max-width: 960px; margin-inline: auto; }
    .header { text-align: center; margin-bottom: 1.5rem; font-size: 1.6rem; }
    .tabs { display: flex; gap: .5rem; flex-wrap: wrap; margin-bottom: 1rem; }
    .tab-btn { padding: .6rem 1rem; border: 2px solid #6c5ce7; background: transparent; color: #a99cf0; border-radius: 8px; cursor: pointer; transition: .2s; }
    .tab-btn.active { background: #6c5ce7; color: #fff; }
    .panel { background: #1a1a2e; border-radius: 12px; padding: 1.25rem; display: flex; flex-direction: column; gap: .9rem; }
    .panel.hidden { display: none; }
    input[type=file], input[type=text] { width: 100%; padding: .6rem; border-radius: 8px; border: 1px solid #333; background: #111; color: #eee; }
    button.primary { padding: .7rem 1.2rem; border: none; border-radius: 8px; background: #6c5ce7; color: #fff; font-size: 1rem; cursor: pointer; }
    button.primary:disabled { opacity: .5; cursor: wait; }
    .result { margin-top: 1.5rem; text-align: center; }
    .result img, .result video { max-width: 100%; border-radius: 10px; }
    .status { font-size: .95rem; color: #a99cf0; min-height: 1.2em; }
    .error { color: #ff6b6b; }
  </style>
</head>
<body>
  <div class="header">🎨 AI Studio</div>
  <nav class="tabs">
    <button class="tab-btn active" data-tab="image">📷 صورة</button>
    <button class="tab-btn" data-tab="video">🎬 فيديو</button>
    <button class="tab-btn" data-tab="generate">✨ توليد</button>
    <button class="tab-btn" data-tab="history">📜 السجل</button>
  </nav>
  <section id="panel-image" class="panel">
    <input type="file" id="imageFile" accept="image/*" />
    <div style="display:flex; gap:.6rem; flex-wrap:wrap;">
      <button class="primary" onclick="doEnhance()">تحسين الجودة</button>
      <button class="primary" onclick="doRemoveBg()">إزالة الخلفية</button>
      <button class="primary" onclick="doColorize()">تلوين</button>
    </div>
  </section>
  <section id="panel-video" class="panel hidden">
    <input type="file" id="videoFile" accept="video/*" />
    <button class="primary" onclick="doVideoUpscale()">تحسين دقة الفيديو 2x</button>
  </section>
  <section id="panel-generate" class="panel hidden">
    <input type="text" id="promptInput" placeholder="اكتب وصف الصورة..." />
    <button class="primary" onclick="doGenerate()">توليد الصورة</button>
  </section>
  <section id="panel-history" class="panel hidden">
    <div style="display:flex; gap:.6rem; flex-wrap:wrap;">
      <button class="primary" onclick="loadHistory()">تحديث السجل</button>
      <button class="primary" onclick="loadStatus()">حالة الخادم</button>
      <button class="primary" onclick="doCleanup()">تنظيف يدوي</button>
    </div>
    <div id="historyBox" style="margin-top:1rem; overflow-x:auto;"></div>
  </section>
  <div id="status" class="status"></div>
  <div id="result" class="result"></div>
  <script>
    const statusEl = document.getElementById('status');
    const resultEl = document.getElementById('result');
    document.querySelectorAll('.tab-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        document.querySelectorAll('.panel').forEach(p => p.classList.add('hidden'));
        document.getElementById('panel-' + btn.dataset.tab).classList.remove('hidden');
      });
    });
    function setStatus(msg, isError = false) { statusEl.textContent = msg; statusEl.className = 'status' + (isError ? ' error' : ''); }
    function showResult(url, isVideo = false) { resultEl.innerHTML = isVideo ? `<video src="${url}" controls autoplay></video>` : `<img src="${url}" alt="result" />`; }
    async function send(endpoint, formData, isVideo = false) {
      setStatus('جارٍ المعالجة...'); resultEl.innerHTML = '';
      try {
        const res = await fetch(endpoint, { method: 'POST', body: formData });
        const data = await res.json();
        if (!res.ok || !data.success) throw new Error(data.detail || data.message || 'فشل الطلب');
        showResult(data.url, isVideo); setStatus('تم بنجاح ✅');
      } catch (err) { setStatus('خطأ: ' + err.message, true); }
    }
    function fdWithFile(id, field = 'file') {
      const f = document.getElementById(id).files[0];
      if (!f) { setStatus('اختر ملفًا أولًا.', true); return null; }
      const fd = new FormData(); fd.append(field, f); return fd;
    }
    function doEnhance() { const fd = fdWithFile('imageFile'); if (fd) send('/api/enhance', fd); }
    function doRemoveBg() { const fd = fdWithFile('imageFile'); if (fd) send('/api/remove-background', fd); }
    function doColorize() { const fd = fdWithFile('imageFile'); if (fd) send('/api/colorize', fd); }
    function doVideoUpscale() { const fd = fdWithFile('videoFile'); if (fd) send('/api/video-upscale', fd, true); }
    function doGenerate() {
      const prompt = document.getElementById('promptInput').value.trim();
      if (!prompt) { setStatus('اكتب وصفًا أولًا.', true); return; }
      const fd = new FormData(); fd.append('prompt', prompt); send('/api/generate', fd);
    }
    async function loadHistory() {
      setStatus('جارٍ تحميل السجل...');
      try {
        const res = await fetch('/api/history?limit=20'); const data = await res.json();
        if (!data.success) throw new Error('فشل الجلب');
        const rows = data.jobs.map(j => `<tr><td>${j.job_type}</td><td style="color:${j.status === 'error' ? '#ff6b6b' : '#7ee787'}">${j.status}</td><td>${j.prompt || '-'}</td><td>${j.output_url ? `<a href="${j.output_url}" target="_blank">فتح</a>` : '-'}</td><td>${j.created_at}</td></tr>`).join('');
        document.getElementById('historyBox').innerHTML = `<table style="width:100%; border-collapse:collapse; font-size:.88rem;"><thead><tr style="background:#222;"><th style="padding:.5rem; text-align:right;">النوع</th><th style="padding:.5rem;">الحالة</th><th style="padding:.5rem;">الوصف</th><th style="padding:.5rem;">المخرج</th><th style="padding:.5rem;">التاريخ</th></tr></thead><tbody>${rows || '<tr><td colspan="5" style="padding:1rem; text-align:center;">لا يوجد سجل بعد.</td></tr>'}</tbody></table>`;
        setStatus(`تم تحميل ${data.count} عملية ✅`);
      } catch (e) { setStatus('خطأ: ' + e.message, true); }
    }
    async function loadStatus() {
      try {
        const res = await fetch('/api/status'); const data = await res.json();
        setStatus(`📁 uploads=${data.uploads_files} | outputs=${data.outputs_files} | 📋 jobs=${data.jobs_total} (اليوم=${data.jobs_today}, أخطاء=${data.jobs_errors})`);
      } catch (e) { setStatus('خطأ: ' + e.message, true); }
    }
    async function doCleanup() {
      if (!confirm('سيتم حذف الملفات الأقدم من 24 ساعة. متابعة؟')) return;
      setStatus('جارٍ التنظيف...');
      try {
        const res = await fetch('/api/cleanup', { method: 'POST' }); const data = await res.json();
        if (!data.success) throw new Error(data.detail || 'فشل');
        const d = data.deleted; setStatus(`✅ حُذف: uploads=${d.uploads}, outputs=${d.outputs}, سجلات=${d.db_rows}`); loadHistory();
      } catch (e) { setStatus('خطأ: ' + e.message, true); }
    }
  </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def home(): return HTML_PAGE

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
