"""
CrispAI local processing server.
Photoshop plugin sends images here, gets back processed results.
"""

import io
import base64
import logging
from flask import Flask, request, jsonify
from PIL import Image

from models.denoise import DenoiseModel
from models.sharpen import SharpenModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

denoise_model = None
sharpen_model = None


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


def decode_image(b64_string: str) -> Image.Image:
    data = base64.b64decode(b64_string)
    return Image.open(io.BytesIO(data)).convert("RGB")


def encode_image(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "version": "0.1.0"})


@app.route("/denoise", methods=["POST"])
def denoise():
    data = request.json
    if not data or "image" not in data:
        return jsonify({"error": "missing image"}), 400

    strength = float(data.get("strength", 0.5))

    try:
        image = decode_image(data["image"])
        model = get_denoise_model()
        result = model.process(image, strength=strength)
        return jsonify({"image": encode_image(result)})
    except Exception as e:
        logger.error(f"Denoise error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/sharpen", methods=["POST"])
def sharpen():
    data = request.json
    if not data or "image" not in data:
        return jsonify({"error": "missing image"}), 400

    mode = data.get("mode", "auto")  # auto, motion_blur, focus_blur
    strength = float(data.get("strength", 0.5))

    try:
        image = decode_image(data["image"])
        model = get_sharpen_model()
        result = model.process(image, mode=mode, strength=strength)
        return jsonify({"image": encode_image(result)})
    except Exception as e:
        logger.error(f"Sharpen error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/enhance", methods=["POST"])
def enhance():
    """Run both denoise and sharpen in one call."""
    data = request.json
    if not data or "image" not in data:
        return jsonify({"error": "missing image"}), 400

    denoise_strength = float(data.get("denoise_strength", 0.5))
    sharpen_strength = float(data.get("sharpen_strength", 0.5))
    sharpen_mode = data.get("sharpen_mode", "auto")

    try:
        image = decode_image(data["image"])
        image = get_denoise_model().process(image, strength=denoise_strength)
        image = get_sharpen_model().process(image, mode=sharpen_mode, strength=sharpen_strength)
        return jsonify({"image": encode_image(image)})
    except Exception as e:
        logger.error(f"Enhance error: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    logger.info("CrispAI server starting on http://localhost:7788")
    app.run(host="127.0.0.1", port=7788, debug=False)
