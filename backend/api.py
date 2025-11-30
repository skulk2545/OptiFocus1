# app.py
import io
import base64
import traceback

import cv2
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image

# --- YOUR ONLY REQUIRED MODULES ---
from backend_spectacle import measure_spectacle_frame
from eyedetector2_6 import AdvancedEyeSpectacleBackend
# ----------------------------------

app = Flask(__name__)
CORS(app)

# persistent instance of PD detector
pd_detector = AdvancedEyeSpectacleBackend()

# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def decode_b64_to_cv2(b64str):
    try:
        data = base64.b64decode(b64str)
        img = Image.open(io.BytesIO(data)).convert("RGB")
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    except Exception:
        return None


def encode_cv2_to_b64(img_bgr, quality=90):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    buff = io.BytesIO()
    pil.save(buff, format="JPEG", quality=quality)
    return base64.b64encode(buff.getvalue()).decode("ascii")


def get_float(x):
    try:
        return float(x)
    except:
        return 0.0


# ---------------------------------------------------------
# ROUTES
# ---------------------------------------------------------

@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"status": "ok"})


@app.route("/process", methods=["POST"])
def process_route():
    """
    Input:
        { "image_b64": "<base64 jpeg>" }

    Output:
        PD values + Frame measurements (A, B, DBL)
    """
    try:
        data = request.get_json(force=True)

        if "image_b64" not in data:
            return jsonify({"error": "missing_image"}), 400

        # -------------------------------
        # Decode B64 → BGR image
        # -------------------------------
        img = decode_b64_to_cv2(data["image_b64"])
        if img is None:
            return jsonify({"error": "invalid_image"}), 400

        # =====================================================
        # 1. PD DETECTION  (eyedetector2_6)
        # =====================================================
        pd_result = pd_detector.process_bgr(img)

        pd_left_mm = get_float(pd_result.get("pd_left_mm"))
        pd_right_mm = get_float(pd_result.get("pd_right_mm"))
        pd_total_mm = pd_left_mm + pd_right_mm

        left_center = pd_result.get("left_center", [0, 0])
        right_center = pd_result.get("right_center", [0, 0])

        scale_mm_per_px_PD = get_float(pd_result.get("scale_mm_per_px"))

        # =====================================================
        # 1b. Crop to eye region including eyebrows
        # =====================================================
        # =====================================================
# 1b. Crop to eye region including eyebrows and side portions
# =====================================================
        if left_center != [0, 0] and right_center != [0, 0]:
            # Horizontal padding increased to include side portions
            x1 = int(min(left_center[0], right_center[0]) - 80)  # left extra
            x2 = int(max(left_center[0], right_center[0]) + 80)  # right extra

            # Vertical padding to cover eyebrows
            y1 = int(min(left_center[1], right_center[1]) - 90)  # above eyes
            y2 = int(max(left_center[1], right_center[1]) + 60)  # below eyes

            # Clamp to image size
            x1, y1 = max(x1, 0), max(y1, 0)
            x2, y2 = min(x2, img.shape[1]), min(y2, img.shape[0])

            img_cropped = img[y1:y2, x1:x2].copy()
        else:
            img_cropped = img.copy()  # fallback to full image
  # fallback to full image

        # =====================================================
        # 2. FRAME MEASUREMENT  (backend_spectacle)
        # =====================================================
        img_bytes = cv2.imencode(".jpg", img_cropped)[1].tobytes()
        frame_result = measure_spectacle_frame(img_bytes)

        frame_detected = bool(frame_result.get("frame_detected", False))
        A_mm = get_float(frame_result.get("A_mm"))
        B_mm = get_float(frame_result.get("B_mm"))
        DBL_mm = get_float(frame_result.get("DBL_mm"))
        px_per_mm = get_float(frame_result.get("px_per_mm"))

        # =====================================================
        # Prepare output image
        # =====================================================
        out_b64 = encode_cv2_to_b64(img_cropped)

        # =====================================================
        # Final response
        # =====================================================
        return jsonify({
            # PD
            "pd_left_mm": pd_left_mm,
            "pd_right_mm": pd_right_mm,
            "pd_mm": pd_total_mm,
            "left_center_px": left_center,
            "right_center_px": right_center,
            "mm_per_px_pd": scale_mm_per_px_PD,

            # Frame
            "frame_detected": frame_detected,
            "A_mm": A_mm,
            "B_mm": B_mm,
            "DBL_mm": DBL_mm,
            "px_per_mm": px_per_mm,

            # Image
            "image_b64": out_b64,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "processing_failed", "detail": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
