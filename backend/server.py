"""
CrispAI local processing server.
Photoshop plugin creates a session, web UI handles preview/apply.
"""

import io
import os
import uuid
import base64
import logging
import subprocess
import webbrowser
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


def find_chromium_browser():
    """Find Chrome or Edge for --app launch (no URL bar, no tabs)."""
    candidates = [
        os.path.expandvars(r"%ProgramFiles%\Google\Chrome\Application\chrome.exe"),
        os.path.expandvars(r"%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe"),
        os.path.expandvars(r"%LocalAppData%\Google\Chrome\Application\chrome.exe"),
        os.path.expandvars(r"%ProgramFiles(x86)%\Microsoft\Edge\Application\msedge.exe"),
        os.path.expandvars(r"%ProgramFiles%\Microsoft\Edge\Application\msedge.exe"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def launch_app_window(url: str):
    """Open URL as a borderless app window (Chrome/Edge --app mode)."""
    browser = find_chromium_browser()
    if browser:
        user_data = os.path.join(os.environ.get("LOCALAPPDATA", os.path.expanduser("~")), "CrispAI", "browser")
        os.makedirs(user_data, exist_ok=True)
        subprocess.Popen([
            browser,
            f"--app={url}",
            f"--user-data-dir={user_data}",
            "--window-size=1280,820",
        ])
        logger.info(f"Launched app window via: {browser}")
    else:
        logger.warning("Chrome/Edge not found, falling back to default browser")
        webbrowser.open(url)

from models.denoise import DenoiseModel
from models.sharpen import SharpenModel
from models.deblur  import DeblurModel

# Log to both console (visible when run via py.exe) and a file (so pyw.exe
# silent launches still leave a trail when something goes wrong).
LOG_FILE = os.path.join(os.path.dirname(__file__), "crispai.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 256 * 1024 * 1024   # 256 MB upload limit
CORS(app)

denoise_model = None
sharpen_model = None
deblur_model  = None
sessions = {}   # sid -> { pil, width, height, filename, result_pil, status }

PREVIEW_MAX = 1200   # px — preview is scaled to this, full-res apply is not


def get_denoise_model():
    global denoise_model
    if denoise_model is None:
        logger.info("Loading denoise model...")
        denoise_model = DenoiseModel()
    return denoise_model


def get_sharpen_model():
    global sharpen_model
    if sharpen_model is None:
        logger.info("Loading sharpen model...")
        sharpen_model = SharpenModel()
    return sharpen_model


def get_deblur_model():
    global deblur_model
    if deblur_model is None:
        logger.info("Loading deblur model...")
        deblur_model = DeblurModel()
    return deblur_model


# ── Image helpers ──────────────────────────────────────────────────────────────

def rgba8_to_pil(b64: str, width: int, height: int) -> Image.Image:
    raw = base64.b64decode(b64)
    return Image.frombytes("RGBA", (width, height), raw).convert("RGB")


def pil_to_png_b64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def pil_to_rgba_b64(image: Image.Image) -> str:
    """Raw RGBA bytes — used by UXP imaging.putPixels()."""
    return base64.b64encode(image.convert("RGBA").tobytes()).decode("utf-8")


def process_image(image: Image.Image, params: dict) -> Image.Image:
    """Three-stage pipeline (Topaz-style):  Denoise -> Deblur -> Sharpen.

    params shape:
      {
        denoise: { enabled: bool, model: str, strength: 0..1 },
        deblur:  { enabled: bool, mode: str,  strength: 0..1, angle: 0..180 },
        sharpen: { enabled: bool, model: str, strength: 0..1 },
      }
    """
    result = image

    d = params.get("denoise", {})
    if d.get("enabled"):
        result = get_denoise_model().process(
            result,
            strength=float(d.get("strength", 0.5)),
            model=d.get("model", "normal"),
        )

    b = params.get("deblur", {})
    if b.get("enabled"):
        result = get_deblur_model().process(
            result,
            mode=b.get("mode", "motion_shake"),
            strength=float(b.get("strength", 0.5)),
            motion_angle=float(b.get("angle", 0.0)),
        )

    s = params.get("sharpen", {})
    if s.get("enabled"):
        result = get_sharpen_model().process(
            result,
            strength=float(s.get("strength", 0.5)),
            model=s.get("model", "standard"),
        )

    return result


def _parse_params(data: dict) -> dict:
    """Accept either the new nested shape or the old flat one (back-compat)."""
    if any(k in data for k in ("denoise", "deblur", "sharpen")):
        return data
    # Legacy flat keys
    return {
        "denoise": {
            "enabled":  float(data.get("denoise_strength", 0)) > 0.01,
            "model":    "normal",
            "strength": float(data.get("denoise_strength", 0.5)),
        },
        "sharpen": {
            "enabled":  float(data.get("sharpen_strength", 0)) > 0.01,
            "model":    "standard",
            "strength": float(data.get("sharpen_strength", 0.5)),
        },
        "deblur": {"enabled": False},
    }


# ── Static / UI ────────────────────────────────────────────────────────────────

@app.route("/")
@app.route("/ui")
def ui():
    return send_from_directory(STATIC_DIR, "ui.html")


# ── Health ─────────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "version": "0.3.0"})


# ── Session API ────────────────────────────────────────────────────────────────

def _new_session(pil: Image.Image, filename: str = "untitled.png") -> str:
    sid = str(uuid.uuid4())
    sessions[sid] = {
        "pil":         pil,
        "width":       pil.size[0],
        "height":      pil.size[1],
        "filename":    filename,
        "result_pil":  None,
        "result_rgba": None,
        "status":      "pending",
    }
    logger.info(f"Session {sid[:8]}…  {pil.size[0]}x{pil.size[1]}  ({filename})")
    return sid


@app.route("/session/create", methods=["POST"])
def session_create():
    """PS plugin: upload raw RGBA pixels, get session URL."""
    data = request.json
    if not data or "image" not in data:
        return jsonify({"error": "missing image"}), 400
    try:
        width  = int(data["width"])
        height = int(data["height"])
        pil = rgba8_to_pil(data["image"], width, height)
        sid = _new_session(pil, "photoshop_layer.png")
        url = f"http://localhost:7788/ui?session={sid}"
        launch_app_window(url)
        return jsonify({"session_id": sid, "url": url})
    except Exception as e:
        logger.error(f"session_create error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/session/upload", methods=["POST"])
def session_upload():
    """Standalone web UI: upload an image file directly."""
    if "file" not in request.files:
        return jsonify({"error": "no file"}), 400
    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "empty filename"}), 400
    try:
        pil = Image.open(f.stream).convert("RGB")
        sid = _new_session(pil, f.filename)
        return jsonify({"session_id": sid, "width": pil.size[0], "height": pil.size[1]})
    except Exception as e:
        logger.error(f"upload error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/session/<sid>/original", methods=["GET"])
def session_original(sid):
    """Original image for display. Downscaled to PREVIEW_MAX so it matches
    the preview pipeline's resolution — otherwise the two compare panels
    would go through different resampling (browser-scale vs LANCZOS) and
    look subtly different even when strength is 0."""
    s = sessions.get(sid)
    if not s:
        return jsonify({"error": "session not found"}), 404
    img = s["pil"]
    w, h = img.size
    if max(w, h) > PREVIEW_MAX:
        scale = PREVIEW_MAX / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return jsonify({
        "image":  pil_to_png_b64(img),
        "width":  s["width"],    # original (true) dimensions, for UI layout
        "height": s["height"],
    })


@app.route("/session/<sid>/preview", methods=["POST"])
def session_preview(sid):
    """Web UI calls this on every slider change (debounced).
    Scales down to PREVIEW_MAX for speed."""
    s = sessions.get(sid)
    if not s:
        return jsonify({"error": "session not found"}), 404
    data = request.json or {}
    try:
        img = s["pil"].copy()
        w, h = img.size
        if max(w, h) > PREVIEW_MAX:
            scale = PREVIEW_MAX / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        result = process_image(img, _parse_params(data))
        return jsonify({"image": pil_to_png_b64(result)})
    except Exception as e:
        logger.error(f"preview error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/session/<sid>/apply", methods=["POST"])
def session_apply(sid):
    """Process at full resolution, store result. Web UI follows up with download."""
    s = sessions.get(sid)
    if not s:
        return jsonify({"error": "session not found"}), 404
    data = request.json or {}
    try:
        result = process_image(s["pil"].copy(), _parse_params(data))
        s["result_pil"]  = result
        s["result_rgba"] = pil_to_rgba_b64(result)   # for PS plugin
        s["status"]      = "ready"
        logger.info(f"Session {sid[:8]}… ready")
        return jsonify({"status": "ready"})
    except Exception as e:
        logger.error(f"apply error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/session/<sid>/download", methods=["GET"])
def session_download(sid):
    """Stream the processed result as a downloadable file."""
    s = sessions.get(sid)
    if not s or s["result_pil"] is None:
        return jsonify({"error": "result not ready"}), 404
    fmt = (request.args.get("format") or "png").lower()
    if fmt not in ("png", "jpg", "jpeg", "tiff", "tif"):
        return jsonify({"error": f"unsupported format: {fmt}"}), 400
    if fmt == "jpg": fmt = "jpeg"
    if fmt == "tif": fmt = "tiff"

    buf = io.BytesIO()
    save_kwargs = {"format": fmt.upper()}
    if fmt == "jpeg":
        save_kwargs["quality"] = 95
        save_kwargs["subsampling"] = 0
    s["result_pil"].save(buf, **save_kwargs)
    buf.seek(0)

    base = os.path.splitext(os.path.basename(s["filename"]))[0] or "image"
    ext  = {"jpeg": "jpg", "tiff": "tif"}.get(fmt, fmt)
    download_name = f"{base}_crispai.{ext}"

    mime = {"png": "image/png", "jpeg": "image/jpeg", "tiff": "image/tiff"}[fmt]
    from flask import send_file
    return send_file(buf, mimetype=mime, as_attachment=True, download_name=download_name)


@app.route("/session/<sid>/result", methods=["GET"])
def session_result(sid):
    """Plugin polls this until status == ready, then places result in PS."""
    s = sessions.get(sid)
    if not s:
        return jsonify({"status": "not_found"}), 404
    if s["status"] == "ready":
        resp = {
            "status":    "ready",
            "raw_rgba":  s["result_rgba"],
            "width":     s["width"],
            "height":    s["height"],
        }
        del sessions[sid]   # free memory once collected
        return jsonify(resp)
    return jsonify({"status": s["status"]})


@app.route("/session/<sid>/cancel", methods=["POST"])
def session_cancel(sid):
    s = sessions.get(sid)
    if s:
        s["status"] = "cancelled"
    return jsonify({"status": "cancelled"})


if __name__ == "__main__":
    logger.info("CrispAI server starting on http://localhost:7788")
    app.run(host="127.0.0.1", port=7788, debug=False)
