import base64, io, os
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from PIL import Image
import cv2, numpy as np
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from integrated_system import IntegratedEyeSpectacleSystem

app = Flask(__name__)
CORS(app)
system = IntegratedEyeSpectacleSystem()

# ----------- IMAGE HELPERS -----------
def b64_to_bgr(b64str):
    data = base64.b64decode(b64str)
    image = Image.open(io.BytesIO(data)).convert("RGB")
    return np.array(image)[:, :, ::-1]

def bgr_to_b64_jpeg(bgr_img, quality=90):
    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ----------- ROUTES -----------
@app.route("/ping")
def ping():
    return jsonify({"status": "ok"})


@app.route("/process", methods=["POST"])
def process():
    try:
        data = request.get_json(force=True)
        frame = b64_to_bgr(data["image_b64"])
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = system.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return jsonify({
                "error": "no_face_detected",
                "annotated_image_b64": bgr_to_b64_jpeg(frame)
            })

        landmarks = results.multi_face_landmarks[0]
        system.process_pd_measurement(landmarks, w, h)
        system.calculate_ipd_calibration(landmarks, w)
        system.detect_frame_contours(frame, landmarks)
        system.frame_detected = system.is_frame_stable_detected()

        if system.frame_detected:
            system.calculate_measurements()

        system.draw_combined_interface(frame)

        # Extract A/B/DBL
        A_px = B_px = DBL_px = 0
        for attr in dir(system):
            if attr.startswith("__"):
                continue
            val = getattr(system, attr)
            if isinstance(val, dict):
                if "A" in val and val["A"] != 0: A_px = val["A"]
                if "B" in val and val["B"] != 0: B_px = val["B"]
                if "DBL" in val and val["DBL"] != 0: DBL_px = val["DBL"]

        px_per_mm = getattr(system, "px_per_mm", 0) or 1e-6
        A_mm = round(A_px / px_per_mm, 1)
        B_mm = round(B_px / px_per_mm, 1)
        DBL_mm = round(DBL_px / px_per_mm, 1)

        pd_val = int(round(system.latest_ipd_mm_avg)) if system.latest_ipd_mm_avg else 0
        head_tilt = round(getattr(system, "latest_head_tilt_deg", 0.0), 1)
        annotated_b64 = bgr_to_b64_jpeg(frame)

        print(f"✅ A={A_mm}mm, B={B_mm}mm, DBL={DBL_mm}mm, PD={pd_val}mm")

        return jsonify({
            "pd_mm": pd_val,
            "A_mm": A_mm,
            "B_mm": B_mm,
            "DBL_mm": DBL_mm,
            "head_tilt_deg": head_tilt,
            "frame_detected": bool(system.frame_detected),
            "annotated_image_b64": annotated_b64
        })

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": "processing_failed", "detail": str(e)}), 500


@app.route("/download_pdf", methods=["POST"])
def download_pdf():
    """Generate and send an order PDF"""
    try:
        data = request.get_json(force=True)

        # Prepare file path
        file_name = f"order_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        file_path = os.path.join(os.getcwd(), file_name)
        c = canvas.Canvas(file_path, pagesize=A4)
        width, height = A4

        # ---- HEADER ----
        optician_name = data.get("optician_name", "Optician Name")
        c.setFont("Helvetica-Bold", 16)
        c.drawString(30, height - 60, optician_name)

        # Optional logo
        if data.get("logo_b64"):
            try:
                imgdata = base64.b64decode(data["logo_b64"])
                img = Image.open(BytesIO(imgdata))
                logo_path = "temp_logo.png"
                img.save(logo_path)
                c.drawImage(logo_path, width - 120, height - 100, 60, 60)
                os.remove(logo_path)
            except Exception as e:
                print("⚠️ Logo load error:", e)

        c.setFont("Helvetica", 11)
        c.drawString(30, height - 80, f"Order Date: {datetime.now().strftime('%d-%m-%Y')}")
        c.drawString(30, height - 95, f"Order / Bill No: {data.get('order_no', '---')}")
        c.drawString(30, height - 110, f"Customer Name: {data.get('customer_name', '---')}")

        # ---- ORDER DETAILS ----
        y = height - 140
        c.setFont("Helvetica-Bold", 12)
        c.drawString(30, y, "ORDER DETAILS")
        y -= 15
        c.setFont("Helvetica", 10)
        c.drawString(40, y, f"INDEX: {data.get('material','')}")
        c.drawString(150, y, f"COATING: {data.get('coating','')}")
        y -= 15
        c.drawString(40, y, f"TYPE: {data.get('type','')}")
        c.drawString(150, y, f"DIA: {data.get('dia','')}")

        # ---- PRESCRIPTION ----
        y -= 30
        c.setFont("Helvetica-Bold", 12)
        c.drawString(30, y, "EYE DETAILS")
        y -= 15
        c.setFont("Helvetica", 10)
        c.drawString(40, y, "EYE    SPH     CYL     AXIS     ADD")
        y -= 15
        c.drawString(40, y, f"RE     {data.get('sph_re','')}     {data.get('cyl_re','')}     {data.get('axis_re','')}     {data.get('add_re','')}")
        y -= 15
        c.drawString(40, y, f"LE     {data.get('sph_le','')}     {data.get('cyl_le','')}     {data.get('axis_le','')}     {data.get('add_le','')}")

        # ---- ADDITIONAL INFO ----
        y -= 30
        c.setFont("Helvetica-Bold", 12)
        c.drawString(30, y, "ADDITIONAL INFO")
        y -= 15
        c.setFont("Helvetica", 10)
        c.drawString(40, y, data.get("additional_info", ""))

        # ---- FRAME PARAMETERS ----
        y -= 30
        c.setFont("Helvetica-Bold", 12)
        c.drawString(30, y, "FRAME PARAMETERS")
        y -= 15
        c.setFont("Helvetica", 10)
        c.drawString(40, y, f"A = {data.get('A_mm',0)} mm,  B = {data.get('B_mm',0)} mm,  DBL = {data.get('DBL_mm',0)} mm")
        y -= 15
        c.drawString(40, y, f"PD = {data.get('pd_mm',0)} mm")

        c.showPage()
        c.save()

        return send_file(file_path, as_attachment=True, download_name=file_name)

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": "pdf_generation_failed", "detail": str(e)}), 500


# ----------- RUN SERVER -----------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
