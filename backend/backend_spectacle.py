import cv2
import mediapipe as mp
import numpy as np
from collections import deque

ASSUMED_IPD_MM = 62.0


# -----------------------------
# CORE SYSTEM (unchanged logic)
# -----------------------------
class SpectacleFrameDetectionSystem:
    def __init__(self):
        self.px_per_mm = 10.0
        self.frame_detected = False
        self.left_box = None
        self.right_box = None

        self.detection_history = deque(maxlen=5)
        self.edge_strength_history = deque(maxlen=10)

        self.A_history = deque(maxlen=60)
        self.B_history = deque(maxlen=60)
        self.stable_A = None
        self.stable_B = None

    def detect_spectacle_frame(self, frame, landmarks, w, h):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        left_reg = self._get_extended_frame_region(landmarks, w, h, "left")
        right_reg = self._get_extended_frame_region(landmarks, w, h, "right")

        left_has, left_box = self._detect_frame_rectangle(enhanced, left_reg)
        right_has, right_box = self._detect_frame_rectangle(enhanced, right_reg)

        edge_strength = self._calculate_edge_strength(enhanced, left_reg, right_reg)
        self.edge_strength_history.append(edge_strength)

        frame_now = left_has and right_has and edge_strength > 0.06
        self.detection_history.append(frame_now)

        # Hysteresis ON
        if not self.frame_detected and sum(self.detection_history) >= 4:
            self.frame_detected = True
            self.left_box = left_box
            self.right_box = right_box

        # Hysteresis OFF
        elif self.frame_detected and sum(self.detection_history) <= 1:
            self.frame_detected = False
            self.left_box = None
            self.right_box = None

        return self.frame_detected

    def _get_extended_frame_region(self, landmarks, w, h, side):
        idxs = (
            [33, 133, 160, 159, 158, 157, 173, 144, 145, 153, 154, 155]
            if side == "left"
            else [362, 263, 387, 386, 385, 384, 398, 373, 374, 380, 381, 382]
        )
        pts = [(int(landmarks.landmark[i].x * w), int(landmarks.landmark[i].y * h)) for i in idxs]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]

        pad_x = 30
        pad_y = 25

        return [
            max(0, min(xs) - pad_x),
            max(0, min(ys) - pad_y),
            min(w, max(xs) + pad_x),
            min(h, max(ys) + pad_y),
        ]

    def _detect_frame_rectangle(self, gray, region):
        x1, y1, x2, y2 = region
        roi = gray[y1:y2, x1:x2]
        if roi.size == 0 or roi.shape[0] < 20 or roi.shape[1] < 20:
            return False, None

        edges = cv2.Canny(cv2.GaussianBlur(roi, (5, 5), 0), 40, 100)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.dilate(edges, kernel)
        edges = cv2.erode(edges, kernel)

        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30, minLineLength=25, maxLineGap=15)
        if lines is None or len(lines) < 3:
            return False, None

        horiz, vert = [], []
        for line in lines:
            x1_l, y1_l, x2_l, y2_l = line[0]
            angle = abs(np.degrees(np.arctan2(y2_l - y1_l, x2_l - x1_l)))
            if angle < 30 or angle > 150:
                horiz.append(line[0])
            elif 60 < angle < 120:
                vert.append(line[0])

        if not horiz or not vert:
            return False, None

        all_x = []
        all_y = []
        for x1_l, y1_l, x2_l, y2_l in horiz + vert:
            all_x += [x1_l, x2_l]
            all_y += [y1_l, y2_l]

        box_x = min(all_x)
        box_y = min(all_y)
        box_w = max(all_x) - box_x
        box_h = max(all_y) - box_y

        if box_w < 50 or box_h < 30 or box_w > 160 or box_h > 110:
            return False, None

        ar = box_w / float(box_h)
        if ar < 0.9 or ar > 2.8:
            return False, None

        return True, [x1 + box_x, y1 + box_y, box_w, box_h]

    def stabilize_measurements(self, A_raw, B_raw):
        self.A_history.append(A_raw)
        self.B_history.append(B_raw)

        if len(self.A_history) == self.A_history.maxlen:
            self.stable_A = np.median(self.A_history)
            self.stable_B = np.median(self.B_history)
        else:
            self.stable_A = A_raw
            self.stable_B = B_raw

        return self.stable_A, self.stable_B

    def _calculate_edge_strength(self, gray, regL, regR):
        tot = 0
        for x1, y1, x2, y2 in [regL, regR]:
            roi = gray[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            edges = np.sum(cv2.Canny(roi, 40, 120) > 0)
            tot += edges / roi.size
        return tot / 2.0


# -----------------------------
# CALIBRATION & FACE CROP
# -----------------------------
def calibrate_px_per_mm_ipd(landmarks, w, h, assumed_ipd_mm=ASSUMED_IPD_MM):
    left = landmarks.landmark[468]
    right = landmarks.landmark[473]
    L = np.array([left.x * w, left.y * h])
    R = np.array([right.x * w, right.y * h])
    return np.linalg.norm(L - R) / assumed_ipd_mm


def crop_face_region(frame, landmarks, w, h):
    IDS = [33, 133, 159, 145, 160, 161, 246, 263, 362, 386, 374, 387, 388, 466,
           70, 63, 105, 66, 107, 300, 293, 334, 296, 336, 6, 197, 195, 5, 4]
    xs, ys = [], []

    for idx in IDS:
        lm = landmarks.landmark[idx]
        xs.append(int(lm.x * w))
        ys.append(int(lm.y * h))

    x_min, x_max = max(min(xs), 0), min(max(xs), w)
    y_min, y_max = max(min(ys), 0), min(max(ys), h)

    if x_max <= x_min or y_max <= y_min:
        return None, 0, 0

    pad_x = int((x_max - x_min) * 0.30)
    pad_y1 = int((y_max - y_min) * 0.25)
    pad_y2 = int((y_max - y_min) * 0.30)

    x0 = max(x_min - pad_x, 0)
    x1 = min(x_max + pad_x, w)
    y0 = max(y_min - pad_y1, 0)
    y1 = min(y_max + pad_y2, h)

    crop = frame[y0:y1, x0:x1].copy()
    if crop.size == 0:
        return None, 0, 0

    return crop, x0, y0


# ---------------------------------
# MAIN BACKEND API FUNCTION (YOU IMPORT THIS)
# ---------------------------------
def measure_spectacle_frame(image_bytes):
    """Runs detection on a single uploaded image."""

    mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # Convert bytes → BGR image
    arr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        return {"error": "Invalid image"}

    h, w = frame.shape[:2]

    # Landmarks
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = mp_face_mesh.process(rgb)
    if not results.multi_face_landmarks:
        return {"frame_detected": False, "error": "no_face"}

    landmarks = results.multi_face_landmarks[0]

    system = SpectacleFrameDetectionSystem()

    # Calibration
    pxmm = calibrate_px_per_mm_ipd(landmarks, w, h)
    system.px_per_mm = pxmm

    # Frame detection
    detected = system.detect_spectacle_frame(frame, landmarks, w, h)
    if not detected:
        return {"frame_detected": False}

    # Must have left/right boxes
    if not system.left_box or not system.right_box:
        return {"frame_detected": False, "error": "box_missing"}

    L = system.left_box
    R = system.right_box

    A_raw = (L[2] + R[2]) / 2
    B_raw = (L[3] + R[3]) / 2

    A_s, B_s = system.stabilize_measurements(A_raw, B_raw)

    return {
        "frame_detected": True,
        "A_mm": round(A_s / pxmm, 1),
        "B_mm": round((B_s * 1.26) / pxmm, 1),
        "px_per_mm": pxmm
    }
