"""
CrispAI local processing server.
Photoshop plugin creates a session, web UI handles preview/apply.
"""

import io
import os
import uuid
import base64
import logging
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

from models.denoise import DenoiseModel
from models.sharpen import SharpenModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder="static")
CORS(app)

denoise_model = None
sharpen_model = None
sessions = {}   # sid -> { original_rgb, width, height, result_rgba, status }

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


def process_image(image: Image.Image, denoise_strength: float,
                  sharpen_strength: float, sharpen_mode: str) -> Image.Image:
    result = get_denoise_model().process(image, strength=denoise_strength)
    result = get_sharpen_model().process(result, mode=sharpen_mode, strength=sharpen_strength)
    return result


# ── Static / UI ────────────────────────────────────────────────────────────────

@app.route("/ui")
def ui():
    return send_from_directory(STATIC_DIR, "ui.html")


# ── Health ─────────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "version": "0.3.0"})


# ── Session API ────────────────────────────────────────────────────────────────

@app.route("/session/create", methods=["POST"])
def session_create():
    """Plugin calls this to upload pixels and get a session URL."""
    data = request.json
    if not data or "image" not in data:
        return jsonify({"error": "missing image"}), 400
    try:
        sid = str(uuid.uuid4())
        width  = int(data["width"])
        height = int(data["height"])
        # Store as PIL RGB to avoid repeated decode
        pil = rgba8_to_pil(data["image"], width, height)
        sessions[sid] = {
            "pil":        pil,
            "width":      width,
            "height":     height,
            "result_rgba": None,
            "status":     "pending",
        }
        url = f"http://localhost:7788/ui?session={sid}"
        logger.info(f"Session created: {sid}  {width}x{height}")
        return jsonify({"session_id": sid, "url": url})
    except Exception as e:
        logger.error(f"session_create error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/session/<sid>/original", methods=["GET"])
def session_original(sid):
    """Web UI fetches the original image for display."""
    s = sessions.get(sid)
    if not s:
        return jsonify({"error": "session not found"}), 404
    return jsonify({
        "image":  pil_to_png_b64(s["pil"]),
        "width":  s["width"],
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
        # Scale for fast preview
        w, h = img.size
        if max(w, h) > PREVIEW_MAX:
            scale = PREVIEW_MAX / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        result = process_image(
            img,
            float(data.get("denoise_strength", 0.5)),
            float(data.get("sharpen_strength", 0.5)),
            data.get("sharpen_mode", "auto"),
        )
        return jsonify({"image": pil_to_png_b64(result)})
    except Exception as e:
        logger.error(f"preview error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/session/<sid>/apply", methods=["POST"])
def session_apply(sid):
    """Web UI calls Apply — processes at full resolution, stores result."""
    s = sessions.get(sid)
    if not s:
        return jsonify({"error": "session not found"}), 404
    data = request.json or {}
    try:
        result = process_image(
            s["pil"].copy(),
            float(data.get("denoise_strength", 0.5)),
            float(data.get("sharpen_strength", 0.5)),
            data.get("sharpen_mode", "auto"),
        )
        s["result_rgba"] = pil_to_rgba_b64(result)
        s["status"] = "ready"
        logger.info(f"Session {sid} ready")
        return jsonify({"status": "ready"})
    except Exception as e:
        logger.error(f"apply error: {e}")
        return jsonify({"error": str(e)}), 500


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
