import cv2
import mediapipe as mp
import numpy as np
import os
import math
import time
from datetime import datetime
from collections import deque

class IntegratedEyeSpectacleSystem:
    def __init__(self):
        # MediaPipe initialization
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            refine_landmarks=True,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.save_dir = "captures"
        os.makedirs(self.save_dir, exist_ok=True)
        
        # === PD MEASUREMENT SETTINGS ===
        self.iris_real_mm = 11.7
        self.HISTORY_LEN = 15
        self.HEAD_TILT_LIMIT_DEG = 6.0
        self.ALIGNMENT_IRIS_PX_MIN = 6
        self.ALIGNMENT_IRIS_PX_MAX = 45
        self.FOCAL_LENGTH_PX = 850
        
        # PD measurement state
        self.ipd_history = deque(maxlen=self.HISTORY_LEN)
        self.left_nose_history = deque(maxlen=self.HISTORY_LEN)
        self.right_nose_history = deque(maxlen=self.HISTORY_LEN)
        self.nose_line_history = deque(maxlen=self.HISTORY_LEN)
        self.scale_history = deque(maxlen=self.HISTORY_LEN)
        self.head_tilt_history = deque(maxlen=self.HISTORY_LEN)
        
        self.manual_left_pupil = None
        self.manual_right_pupil = None
        self.selected_eye = None
        self.auto_left_pupil = None
        self.auto_right_pupil = None
        self.initial_auto_left_pupil = None
        self.initial_auto_right_pupil = None
        
        # === SPECTACLE MEASUREMENT SETTINGS ===
        self.px_per_mm = 3.5
        self.calibration_samples = []
        self.calibration_stable = False
        
        # Valid ranges (internal validation)
        self.A_MIN = 40.0
        self.A_MAX = 62.0
        self.B_MIN = 25.0
        self.B_MAX = 50.0
        self.DBL_MIN = 14.0
        self.DBL_MAX = 24.0
        
        # Frame detection
        self.frame_detected = False
        self.last_detection_state = False
        self.left_frame_box = None
        self.right_frame_box = None
        self.left_box_history = deque(maxlen=10)
        self.right_box_history = deque(maxlen=10)
        self.detection_confidence_buffer = deque(maxlen=10)
        self.min_confidence_threshold = 0.5
        
        # Measurements
        self.measurements = {"A": 0, "B": 0, "DBL": 0}
        self.stable_measurements = {"A": 0, "B": 0, "DBL": 0}
        
        # Manual adjustments
        self.manual_adjustments_active = False
        self.left_size_adjustment = {"w": 0, "h": 0}
        self.right_size_adjustment = {"w": 0, "h": 0}
        self.left_box_offset = {"x": 0, "y": 0}
        self.right_box_offset = {"x": 0, "y": 0}
        
        # Mouse interaction
        self.dragging_box = None
        self.dragging_edge = None
        self.drag_start_pos = None
        self.left_lens_rect = None
        self.right_lens_rect = None
        
        # Freeze state
        self.freeze_frame = None
        self.freeze_landmarks = None
        
        # Edge detection
        self.canny_low = 20
        self.canny_high = 60
        self.min_area = 200
        
        # Store latest measurements
        self.latest_ipd_mm_avg = None
        self.latest_left_nose_avg = None
        self.latest_right_nose_avg = None
        self.latest_head_tilt_deg = 0.0
        self.latest_left_open = False
        self.latest_right_open = False
    
    # ==================== UTILITY FUNCTIONS ====================
    
    def dist(self, a, b):
        """Calculate distance between two points"""
        return math.hypot(a[0] - b[0], a[1] - b[1]) if (a and b) else 0.0
    
    def angle_between_points_deg(self, p1, p2):
        """Calculate angle between two points"""
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        return math.degrees(math.atan2(dx, dy))
    
    def calculate_head_tilt(self, face_landmarks, w, h):
        """Calculate head tilt angle"""
        try:
            forehead = (int(face_landmarks.landmark[10].x * w), int(face_landmarks.landmark[10].y * h))
            chin = (int(face_landmarks.landmark[152].x * w), int(face_landmarks.landmark[152].y * h))
            vert_angle = self.angle_between_points_deg(forehead, chin)
            overall_tilt = abs(vert_angle)
            self.head_tilt_history.append(overall_tilt)
            smoothed_tilt = sum(self.head_tilt_history) / len(self.head_tilt_history)
            return smoothed_tilt
        except Exception:
            return 0.0
    
    def draw_sniper_cross(self, img, pos, size=20, color=(255, 180, 0), thickness=2):
        """Draw crosshair marker"""
        if pos is None:
            return
        x, y = int(pos[0]), int(pos[1])
        cv2.line(img, (x-size, y), (x+size, y), color, thickness)
        cv2.line(img, (x, y-size), (x, y+size), color, thickness)
        cv2.circle(img, (x, y), max(2, size//6), color, -1)
    
    def eye_aspect_ratio(self, landmarks, eye_points, w, h):
        """Calculate Eye Aspect Ratio for blink detection"""
        pts = [(int(landmarks[p].x * w), int(landmarks[p].y * h)) for p in eye_points]
        if len(pts) != 6:
            return 0.0
        A = self.dist(pts[1], pts[5])
        B = self.dist(pts[2], pts[4])
        C = self.dist(pts[0], pts[3])
        return (A + B) / (2.0 * C) if C > 0 else 0.0
    
    # ==================== PD MEASUREMENT FUNCTIONS ====================
    
    def process_pd_measurement(self, landmarks, w, h):
        """Process PD measurement and store results"""
        
        ipd_mm_avg = left_nose_avg = right_nose_avg = None
        head_tilt_deg = 0.0
        VIEWER_LEFT_OPEN = VIEWER_RIGHT_OPEN = False
        
        try:
            auto_left = (int(landmarks.landmark[468].x * w), int(landmarks.landmark[468].y * h))
            auto_right = (int(landmarks.landmark[473].x * w), int(landmarks.landmark[473].y * h))
            nose_center = (int(landmarks.landmark[1].x * w), int(landmarks.landmark[1].y * h))
            
            self.auto_left_pupil = auto_left
            self.auto_right_pupil = auto_right
            if self.initial_auto_left_pupil is None:
                self.initial_auto_left_pupil = auto_left
            if self.initial_auto_right_pupil is None:
                self.initial_auto_right_pupil = auto_right
            
            final_left = self.manual_left_pupil if self.manual_left_pupil is not None else self.auto_left_pupil
            final_right = self.manual_right_pupil if self.manual_right_pupil is not None else self.auto_right_pupil
            
            # Eye open detection
            left_eye_idx = [362, 385, 387, 263, 373, 386]
            right_eye_idx = [33, 160, 158, 133, 153, 159]
            ear_left = self.eye_aspect_ratio(landmarks.landmark, left_eye_idx, w, h)
            ear_right = self.eye_aspect_ratio(landmarks.landmark, right_eye_idx, w, h)
            VIEWER_LEFT_OPEN = ear_left > 0.20
            VIEWER_RIGHT_OPEN = ear_right > 0.20
            
            # Iris size for scale
            left_iris_px = self.dist((landmarks.landmark[469].x * w, landmarks.landmark[469].y * h),
                                     (landmarks.landmark[471].x * w, landmarks.landmark[471].y * h))
            right_iris_px = self.dist((landmarks.landmark[474].x * w, landmarks.landmark[474].y * h),
                                      (landmarks.landmark[476].x * w, landmarks.landmark[476].y * h))
            iris_candidates = [v for v in [left_iris_px, right_iris_px] if v and v > 0]
            if iris_candidates:
                iris_px = sum(iris_candidates) / len(iris_candidates)
                current_scale = self.iris_real_mm / iris_px if iris_px > 0 else 0.1
                if current_scale > 0:
                    self.scale_history.append(current_scale)
            
            smoothed_scale = (sum(self.scale_history) / len(self.scale_history)) if len(self.scale_history) else None
            scale_to_use = smoothed_scale if smoothed_scale and smoothed_scale > 0 else (
                self.iris_real_mm / iris_px if iris_px > 0 else 0.1)
            
            head_tilt_deg = self.calculate_head_tilt(landmarks, w, h)
            
            # Nose line calculation
            if final_left and final_right:
                eye_line_y = (final_left[1] + final_right[1]) / 2.0
            else:
                eye_line_y = h // 2
            
            raw_nose_line_point = (nose_center[0], int(eye_line_y))
            self.nose_line_history.append(raw_nose_line_point)
            avg_nose_line_point = (int(sum(pt[0] for pt in self.nose_line_history) / len(self.nose_line_history)),
                                   int(sum(pt[1] for pt in self.nose_line_history) / len(self.nose_line_history)))
            
            # Left/Right to nose measurements
            if VIEWER_LEFT_OPEN and final_left:
                left_to_nose_px = self.dist(final_left, avg_nose_line_point)
                left_nose_val = left_to_nose_px * scale_to_use
                self.left_nose_history.append(left_nose_val)
                left_nose_avg = sum(self.left_nose_history) / len(self.left_nose_history)
            
            if VIEWER_RIGHT_OPEN and final_right:
                right_to_nose_px = self.dist(final_right, avg_nose_line_point)
                right_nose_val = right_to_nose_px * scale_to_use
                self.right_nose_history.append(right_nose_val)
                right_nose_avg = sum(self.right_nose_history) / len(self.right_nose_history)
            
            # PD calculation
            if VIEWER_LEFT_OPEN and VIEWER_RIGHT_OPEN and final_left and final_right:
                ipd_px = self.dist(final_left, final_right)
                ipd_mm = ipd_px * scale_to_use
                self.ipd_history.append(ipd_mm)
                ipd_mm_avg = sum(self.ipd_history) / len(self.ipd_history)
            elif VIEWER_LEFT_OPEN and not VIEWER_RIGHT_OPEN:
                ipd_mm_avg = left_nose_avg
            elif VIEWER_RIGHT_OPEN and not VIEWER_LEFT_OPEN:
                ipd_mm_avg = right_nose_avg
            
        except Exception:
            pass
        
        # Store latest values
        self.latest_ipd_mm_avg = ipd_mm_avg
        self.latest_left_nose_avg = left_nose_avg
        self.latest_right_nose_avg = right_nose_avg
        self.latest_head_tilt_deg = head_tilt_deg
        self.latest_left_open = VIEWER_LEFT_OPEN
        self.latest_right_open = VIEWER_RIGHT_OPEN
    
    # ==================== SPECTACLE MEASUREMENT FUNCTIONS ====================
    
    def detect_frame_contours(self, image, landmarks):
        """Detect spectacle frame contours"""
        h, w, _ = image.shape
        try:
            left_outer = landmarks.landmark[33]
            left_inner = landmarks.landmark[133]
            right_inner = landmarks.landmark[362]
            right_outer = landmarks.landmark[263]
            left_top = landmarks.landmark[159]
            left_bottom = landmarks.landmark[145]
            right_top = landmarks.landmark[386]
            right_bottom = landmarks.landmark[374]
            
            left_x1 = max(0, int(left_outer.x * w) - 25)
            left_x2 = min(w, int(left_inner.x * w) + 10)
            left_y1 = max(0, int(left_top.y * h) - 20)
            left_y2 = min(h, int(left_bottom.y * h) + 20)
            
            right_x1 = max(0, int(right_inner.x * w) - 10)
            right_x2 = min(w, int(right_outer.x * w) + 25)
            right_y1 = max(0, int(right_top.y * h) - 20)
            right_y2 = min(h, int(right_bottom.y * h) + 20)
            
            left_frame = self.detect_single_lens_frame(
                image[left_y1:left_y2, left_x1:left_x2],
                offset=(left_x1, left_y1),
                landmarks=landmarks,
                is_left=True,
                frame_shape=(w, h)
            )
            
            right_frame = self.detect_single_lens_frame(
                image[right_y1:right_y2, right_x1:right_x2],
                offset=(right_x1, right_y1),
                landmarks=landmarks,
                is_left=False,
                frame_shape=(w, h)
            )
            
            if left_frame and right_frame:
                self.left_box_history.append(left_frame)
                self.right_box_history.append(right_frame)
                self.detection_confidence_buffer.append(1)
                return True
            else:
                self.detection_confidence_buffer.append(0)
                return False
                
        except Exception:
            self.detection_confidence_buffer.append(0)
            return False
    
    def detect_single_lens_frame(self, roi, offset=(0, 0), landmarks=None, is_left=True, frame_shape=(640, 480)):
        """Multi-method frame detection"""
        if roi.size == 0 or roi.shape[0] < 20 or roi.shape[1] < 20:
            return None
        
        try:
            edge_result = self.detect_by_edges(roi, offset)
            adaptive_result = self.detect_by_adaptive_threshold(roi, offset)
            color_result = self.detect_by_color_contrast(roi, offset)
            
            if landmarks and not any([edge_result, adaptive_result, color_result]):
                landmark_result = self.estimate_from_landmarks(landmarks, is_left, frame_shape)
                return landmark_result
            
            results = [r for r in [edge_result, adaptive_result, color_result] if r is not None]
            
            if not results:
                if landmarks:
                    return self.estimate_from_landmarks(landmarks, is_left, frame_shape)
                return None
            
            best = max(results, key=lambda r: r[2] * r[3])
            
            x, y, w, h = best
            shrink_factor = 0.95
            center_x = x + w // 2
            center_y = y + h // 2
            new_w = int(w * shrink_factor)
            new_h = int(h * shrink_factor)
            new_x = center_x - new_w // 2
            new_y = center_y - new_h // 2
            
            return (new_x, new_y, new_w, new_h)
            
        except Exception:
            return None
    
    def detect_by_edges(self, roi, offset):
        """Edge-based detection"""
        try:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, self.canny_low, self.canny_high)
            
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            edges = cv2.dilate(edges, kernel, iterations=2)
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
            
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            return self.find_best_contour(contours, roi.shape, offset)
        except:
            return None
    
    def detect_by_adaptive_threshold(self, roi, offset):
        """Adaptive threshold detection"""
        try:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            thresh1 = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY, 11, 2)
            thresh2 = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                           cv2.THRESH_BINARY, 11, 2)
            thresh = cv2.bitwise_or(thresh1, thresh2)
            
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
            
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            return self.find_best_contour(contours, roi.shape, offset)
        except:
            return None
    
    def detect_by_color_contrast(self, roi, offset):
        """Color contrast-based detection"""
        try:
            lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
            l_channel = lab[:,:,0]
            
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            enhanced = clahe.apply(l_channel)
            
            _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
            
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            return self.find_best_contour(contours, roi.shape, offset)
        except:
            return None
    
    def find_best_contour(self, contours, roi_shape, offset):
        """Find best matching spectacle frame contour"""
        if not contours:
            return None
        
        valid_contours = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area:
                continue
            
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / h if h > 0 else 0
            
            if 0.5 < aspect_ratio < 3.0 and w > 30 and h > 20:
                hull = cv2.convexHull(cnt)
                hull_area = cv2.contourArea(hull)
                solidity = area / hull_area if hull_area > 0 else 0
                
                if solidity > 0.3:
                    score = area * solidity
                    valid_contours.append((x, y, w, h, score))
        
        if not valid_contours:
            return None
        
        best = max(valid_contours, key=lambda x: x[4])
        x, y, w, h, _ = best
        
        return (x + offset[0], y + offset[1], w, h)
    
    def estimate_from_landmarks(self, landmarks, is_left, frame_shape):
        """Fallback: estimate frame position from landmarks"""
        w, h = frame_shape
        
        if is_left:
            outer = landmarks.landmark[33]
            inner = landmarks.landmark[133]
            top = landmarks.landmark[159]
            bottom = landmarks.landmark[145]
        else:
            inner = landmarks.landmark[362]
            outer = landmarks.landmark[263]
            top = landmarks.landmark[386]
            bottom = landmarks.landmark[374]
        
        x1 = int(outer.x * w) - 8
        x2 = int(inner.x * w) + 3
        y1 = int(top.y * h) - 5
        y2 = int(bottom.y * h) + 5
        
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        frame_w = x2 - x1
        frame_h = y2 - y1
        
        if frame_w > 30 and frame_h > 20:
            return (x1, y1, frame_w, frame_h)
        
        return None
    
    def is_frame_stable_detected(self):
        """Check if frame detection is stable"""
        if len(self.detection_confidence_buffer) < 3:
            return False
        detection_rate = sum(self.detection_confidence_buffer) / len(self.detection_confidence_buffer)
        return detection_rate >= self.min_confidence_threshold
    
    def get_stable_box(self, history):
        """Average box positions for smooth tracking"""
        if not history:
            return None
        x_vals = [box[0] for box in history]
        y_vals = [box[1] for box in history]
        w_vals = [box[2] for box in history]
        h_vals = [box[3] for box in history]
        return (
            int(np.median(x_vals)),
            int(np.median(y_vals)),
            int(np.median(w_vals)),
            int(np.median(h_vals))
        )
    
    def calculate_measurements(self):
        """Calculate measurements from detected frames"""
        left_box = self.get_stable_box(self.left_box_history)
        right_box = self.get_stable_box(self.right_box_history)
        
        if not left_box or not right_box:
            return
        
        self.left_frame_box = left_box
        self.right_frame_box = right_box
        
        left_w = left_box[2] + self.left_size_adjustment["w"]
        left_h = left_box[3] + self.left_size_adjustment["h"]
        right_w = right_box[2] + self.right_size_adjustment["w"]
        right_h = right_box[3] + self.right_size_adjustment["h"]
        
        left_inner_edge = left_box[0] + self.left_box_offset["x"] + left_w
        right_inner_edge = right_box[0] + self.right_box_offset["x"]
        dbl_px = abs(right_inner_edge - left_inner_edge)
        
        A_px = (left_w + right_w) / 2
        B_px = (left_h + right_h) / 2
        
        self.measurements["A"] = A_px
        self.measurements["B"] = B_px
        self.measurements["DBL"] = dbl_px
        
        self.stable_measurements["A"] = A_px
        self.stable_measurements["B"] = B_px
        self.stable_measurements["DBL"] = dbl_px

    
    def update_measurements_from_boxes(self):
        """Recompute measurements from current boxes"""
        left_box = self.get_stable_box(self.left_box_history)
        right_box = self.get_stable_box(self.right_box_history)
        
        if not left_box and self.left_lens_rect:
            lx1, ly1, lx2, ly2 = self.left_lens_rect
            left_box = (lx1, ly1, lx2 - lx1, ly2 - ly1)
        if not right_box and self.right_lens_rect:
            rx1, ry1, rx2, ry2 = self.right_lens_rect
            right_box = (rx1, ry1, rx2 - rx1, ry2 - ry1)
        
        if not left_box or not right_box:
            return
        
        self.left_frame_box = left_box
        self.right_frame_box = right_box
        
        left_w = left_box[2] + self.left_size_adjustment.get("w", 0)
        left_h = left_box[3] + self.left_size_adjustment.get("h", 0)
        right_w = right_box[2] + self.right_size_adjustment.get("w", 0)
        right_h = right_box[3] + self.right_size_adjustment.get("h", 0)
        
        left_inner_edge = left_box[0] + self.left_box_offset.get("x", 0) + left_w
        right_inner_edge = right_box[0] + self.right_box_offset.get("x", 0)
        dbl_px = abs(right_inner_edge - left_inner_edge)
        
        A_px = (left_w + right_w) / 2
        B_px = (left_h + right_h) / 2
        
        self.stable_measurements["A"] = A_px
        self.stable_measurements["B"] = B_px
        self.stable_measurements["DBL"] = dbl_px
    
    def calculate_ipd_calibration(self, landmarks, frame_width):
        """IPD-based calibration"""
        try:
            left_iris_x = landmarks.landmark[468].x * frame_width
            right_iris_x = landmarks.landmark[473].x * frame_width
            ipd_px = abs(right_iris_x - left_iris_x)
            cal_ipd = ipd_px / 63.0 if ipd_px > 0 else None
            
            left_outer_x = landmarks.landmark[33].x * frame_width
            right_outer_x = landmarks.landmark[263].x * frame_width
            eye_span_px = abs(right_outer_x - left_outer_x)
            cal_span = eye_span_px / 95.0 if eye_span_px > 0 else None
            
            candidates = [c for c in (cal_ipd, cal_span) if c is not None]
            if not candidates:
                return
            new_cal = (cal_ipd * 0.9 + cal_span * 0.1) if (cal_ipd and cal_span) else (cal_ipd or cal_span)
            
            if 1.0 < new_cal < 7.0:
                self.calibration_samples.append(new_cal)
                if len(self.calibration_samples) > 20:
                    self.calibration_samples.pop(0)
                if len(self.calibration_samples) >= 6:
                    self.px_per_mm = float(np.median(self.calibration_samples))
                    self.calibration_stable = True
                else:
                    self.px_per_mm = new_cal
        except Exception:
            pass
    
    def draw_combined_interface(self, frame):
        frame = np.ascontiguousarray(frame, dtype=np.uint8)  # ✅ ensure OpenCV compatibility
        cv2.drawMarker(frame, self.auto_left_pupil, (0,255,0), cv2.MARKER_CROSS, 8, 1)
        h, w, _ = frame.shape
        
        # Draw pupil markers
        if self.auto_left_pupil:
            cv2.drawMarker(frame, self.auto_left_pupil, (0,255,0), cv2.MARKER_CROSS, 8, 1)
        if self.auto_right_pupil:
            cv2.drawMarker(frame, self.auto_right_pupil, (0,255,0), cv2.MARKER_CROSS, 8, 1)
        
        # Draw spectacle frame boxes
        left_box = self.get_stable_box(self.left_box_history)
        right_box = self.get_stable_box(self.right_box_history)
        
        if left_box and right_box:
            A_mm = round(self.stable_measurements["A"] / self.px_per_mm, 1)
            B_mm = round(self.stable_measurements["B"] / self.px_per_mm, 1)
            DBL_mm = round(self.stable_measurements["DBL"] / self.px_per_mm, 1)
            
            box_color = (255, 120, 0)  # Orange for spectacle boxes
            text_color = (255, 120, 0)
            thickness = 2
            line_type = cv2.LINE_AA
            
            left_x = left_box[0] + self.left_box_offset["x"]
            left_y = left_box[1] + self.left_box_offset["y"]
            left_w = left_box[2] + self.left_size_adjustment["w"]
            left_h = left_box[3] + self.left_size_adjustment["h"]
            
            right_x = right_box[0] + self.right_box_offset["x"]
            right_y = right_box[1] + self.right_box_offset["y"]
            right_w = right_box[2] + self.right_size_adjustment["w"]
            right_h = right_box[3] + self.right_size_adjustment["h"]
            
            left_x1, left_y1 = int(left_x), int(left_y)
            left_x2, left_y2 = int(left_x + left_w), int(left_y + left_h)
            cv2.rectangle(frame, (left_x1, left_y1), (left_x2, left_y2), box_color, thickness, line_type)
            self.left_lens_rect = (left_x1, left_y1, left_x2, left_y2)
            
            right_x1, right_y1 = int(right_x), int(right_y)
            right_x2, right_y2 = int(right_x + right_w), int(right_y + right_h)
            cv2.rectangle(frame, (right_x1, right_y1), (right_x2, right_y2), box_color, thickness, line_type)
            self.right_lens_rect = (right_x1, right_y1, right_x2, right_y2)
            
            # Spectacle measurements on left side
            cv2.putText(frame, f"A: {A_mm}mm", (left_x1, left_y1 - 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, text_color, 2, line_type)
            cv2.putText(frame, f"B: {B_mm}mm", (left_x1, left_y1 - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, text_color, 2, line_type)
            
            center_x = int((left_x2 + right_x1) / 2)
            center_y = int((left_y1 + right_y1) / 2)
            cv2.putText(frame, f"DBL: {DBL_mm}mm", (center_x - 45, center_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, text_color, 2, line_type)
        
        # Draw PD measurements panel (top-left)
        panel_bg = frame.copy()
        cv2.rectangle(panel_bg, (10, 10), (350, 180), (0, 0, 0), -1)
        cv2.addWeighted(panel_bg, 0.7, frame, 0.3, 0, frame)
        
        y0 = 35
        cv2.putText(frame, "=== PD MEASUREMENTS ===", (20, y0), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        y0 += 30
        
        if self.latest_ipd_mm_avg:
            cv2.putText(frame, f"PD: {int(round(self.latest_ipd_mm_avg))} mm",
                        (20, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:
            cv2.putText(frame, "PD: -- mm", (20, y0), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        y0 += 28
        
        if self.latest_left_open and self.latest_left_nose_avg is not None:
            cv2.putText(frame, f"Left->Nose: {int(round(self.latest_left_nose_avg))} mm",
                        (20, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 255, 100), 1)
        else:
            cv2.putText(frame, "Left eye: CLOSED", (20, y0),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 1)
        y0 += 25
        
        if self.latest_right_open and self.latest_right_nose_avg is not None:
            cv2.putText(frame, f"Right->Nose: {int(round(self.latest_right_nose_avg))} mm",
                        (20, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 255, 100), 1)
        else:
            cv2.putText(frame, "Right eye: CLOSED", (20, y0),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 1)
        y0 += 25
        
        cv2.putText(frame, f"Head tilt: {int(round(self.latest_head_tilt_deg))} deg", (20, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                    (0,200,200) if abs(self.latest_head_tilt_deg) <= self.HEAD_TILT_LIMIT_DEG else (0,0,255), 1)
        
        # Draw spectacle info panel (top-right)
        if self.frame_detected:
            panel_bg2 = frame.copy()
            cv2.rectangle(panel_bg2, (w - 360, 10), (w - 10, 130), (0, 0, 0), -1)
            cv2.addWeighted(panel_bg2, 0.7, frame, 0.3, 0, frame)
            
            y0 = 35
            cv2.putText(frame, "=== SPECTACLE FRAME ===", (w - 350, y0), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 120, 0), 2)
            y0 += 30
            
            status = "[MANUAL]" if self.manual_adjustments_active else "[AUTO]"
            cv2.putText(frame, status, (w - 350, y0), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 0), 1)
            y0 += 25
            
            cal_status = "Cal: OK" if self.calibration_stable else f"Cal: {len(self.calibration_samples)}/6"
            cv2.putText(frame, cal_status, (w - 350, y0), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            y0 += 22
            
            cv2.putText(frame, f"PX/MM: {self.px_per_mm:.2f}", (w - 350, y0), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Controls at bottom
        cv2.putText(frame, "P: Capture PD | S: Capture Spectacle | R: Reset | Drag boxes/pupils | ESC: Exit",
                    (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    
    def save_capture(self, frame, kind, landmarks):
        """Save capture with measurements"""
        h, w, _ = frame.shape
        
        left_eye = (int(landmarks.landmark[33].x * w), int(landmarks.landmark[33].y * h))
        right_eye = (int(landmarks.landmark[263].x * w), int(landmarks.landmark[263].y * h))
        
        pad = 80
        x1 = max(0, left_eye[0] - pad)
        x2 = min(w, right_eye[0] + pad)
        y1 = max(0, min(left_eye[1], right_eye[1]) - pad)
        y2 = min(h, max(left_eye[1], right_eye[1]) + pad)
        crop = frame[y1:y2, x1:x2]
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.save_dir}/{kind}_{timestamp}.jpg"
        cv2.imwrite(filename, crop)
        
        if kind == "spectacle":
            A_mm = round(self.stable_measurements["A"] / self.px_per_mm, 1)
            B_mm = round(self.stable_measurements["B"] / self.px_per_mm, 1)
            DBL_mm = round(self.stable_measurements["DBL"] / self.px_per_mm, 1)
            
            with open(f"{self.save_dir}/{kind}_{timestamp}.txt", "w") as f:
                f.write("="*50 + "\n")
                f.write("     SPECTACLE MEASUREMENTS\n")
                f.write("="*50 + "\n\n")
                f.write(f"A (Lens Width):     {A_mm} mm\n")
                f.write(f"B (Lens Height):    {B_mm} mm\n")
                f.write(f"DBL (Bridge Width): {DBL_mm} mm\n")
                f.write("\n" + "="*50 + "\n")
            
            print(f"\n{'='*50}")
            print(f"[SUCCESS] SPECTACLE CAPTURE SAVED")
            print(f"{'='*50}")
            print(f"A (Lens Width):     {A_mm} mm")
            print(f"B (Lens Height):    {B_mm} mm")
            print(f"DBL (Bridge Width): {DBL_mm} mm")
            print(f"{'='*50}\n")
        
        elif kind == "pd":
            if self.ipd_history:
                pd_mm = int(round(sum(self.ipd_history) / len(self.ipd_history)))
            else:
                pd_mm = 0
            
            with open(f"{self.save_dir}/{kind}_{timestamp}.txt", "w") as f:
                f.write("="*50 + "\n")
                f.write("     PD MEASUREMENT\n")
                f.write("="*50 + "\n\n")
                f.write(f"Pupillary Distance: {pd_mm} mm\n")
                f.write("\n" + "="*50 + "\n")
            
            print(f"\n{'='*50}")
            print(f"[SUCCESS] PD CAPTURE SAVED")
            print(f"{'='*50}")
            print(f"Pupillary Distance: {pd_mm} mm")
            print(f"{'='*50}\n")
    
    # ==================== MAIN LOOP ====================
    
    def run(self):
        """Main application loop with combined display"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("ERROR: Could not open webcam.")
            return
        
        print("\n" + "="*70)
        print(" INTEGRATED EYE & SPECTACLE MEASUREMENT SYSTEM")
        print("="*70)
        print("\n COMBINED MODE - Both measurements in single window")
        print("\n CONTROLS:")
        print("  P = Capture PD measurement")
        print("  S = Capture Spectacle measurement")
        print("  R = Reset adjustments")
        print("  Drag = Adjust pupils or spectacle boxes")
        print("  ESC = Exit")
        print("="*70 + "\n")
        
        cv2.namedWindow("Integrated Measurement System")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            raw_frame = frame.copy()
            h, w, _ = frame.shape
            
            # Process face detection
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb)
            
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0]
                
                # Process both measurements
                self.process_pd_measurement(landmarks, w, h)
                self.calculate_ipd_calibration(landmarks, w)
                detected = self.detect_frame_contours(frame, landmarks)
                self.frame_detected = self.is_frame_stable_detected()
                
                if self.frame_detected:
                    self.calculate_measurements()
                
                # Draw combined interface
                self.draw_combined_interface(frame)
                
            else:
                cv2.putText(frame, "NO FACE DETECTED", (w//2 - 150, h//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            
            cv2.imshow("Integrated Measurement System", frame)
            
            # Set mouse callback
            cv2.setMouseCallback("Integrated Measurement System", combined_mouse_callback, self)
            
            # Key handling
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('p') or key == ord('P'):
                if results.multi_face_landmarks:
                    self.save_capture(raw_frame, "pd", landmarks)
            
            elif key == ord('s') or key == ord('S'):
                if results.multi_face_landmarks and self.frame_detected:
                    self.save_capture(raw_frame, "spectacle", landmarks)
            
            elif key == ord('r') or key == ord('R'):
                # Reset PD
                self.manual_left_pupil = None
                self.manual_right_pupil = None
                
                # Reset Spectacle
                self.left_box_offset = {"x": 0, "y": 0}
                self.right_box_offset = {"x": 0, "y": 0}
                self.left_size_adjustment = {"w": 0, "h": 0}
                self.right_size_adjustment = {"w": 0, "h": 0}
                self.manual_adjustments_active = False
                print("[RESET] Reset complete")
            
            elif key == 27:
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("\n[EXIT] Application closed\n")


# ==================== MOUSE CALLBACK ====================

def combined_mouse_callback(event, x, y, flags, param):
    """Combined mouse callback for both PD and spectacle measurements"""
    system = param
    edge_tolerance = 15
    corner_tolerance = 20
    
    if event == cv2.EVENT_LBUTTONDOWN:
        system.drag_start_pos = (x, y)
        
        # Check for pupil dragging (PD measurement)
        if system.auto_left_pupil and system.dist((x, y), system.auto_left_pupil) < 30:
            system.manual_left_pupil = (x, y)
            system.selected_eye = "left"
            return
        elif system.auto_right_pupil and system.dist((x, y), system.auto_right_pupil) < 30:
            system.manual_right_pupil = (x, y)
            system.selected_eye = "right"
            return
        
        # Check for spectacle box dragging
        def check_rect_and_set(rect, side):
            if not rect:
                return False
            x1, y1, x2, y2 = rect
            
            # Check corners
            if abs(x - x1) < corner_tolerance and abs(y - y1) < corner_tolerance:
                system.dragging_edge = f"{side}_top_left_corner"
                return True
            if abs(x - x2) < corner_tolerance and abs(y - y1) < corner_tolerance:
                system.dragging_edge = f"{side}_top_right_corner"
                return True
            if abs(x - x1) < corner_tolerance and abs(y - y2) < corner_tolerance:
                system.dragging_edge = f"{side}_bottom_left_corner"
                return True
            if abs(x - x2) < corner_tolerance and abs(y - y2) < corner_tolerance:
                system.dragging_edge = f"{side}_bottom_right_corner"
                return True
            
            # Check edges
            if abs(y - y1) < edge_tolerance and x1 < x < x2:
                system.dragging_edge = f"{side}_top"
                return True
            if abs(y - y2) < edge_tolerance and x1 < x < x2:
                system.dragging_edge = f"{side}_bottom"
                return True
            if abs(x - x1) < edge_tolerance and y1 < y < y2:
                system.dragging_edge = f"{side}_left"
                return True
            if abs(x - x2) < edge_tolerance and y1 < y < y2:
                system.dragging_edge = f"{side}_right"
                return True
            
            # Check center
            if x1 + 10 <= x <= x2 - 10 and y1 + 10 <= y <= y2 - 10:
                system.dragging_box = side
                return True
            
            return False
        
        if check_rect_and_set(system.right_lens_rect, "right"):
            system.manual_adjustments_active = True
            return
        if check_rect_and_set(system.left_lens_rect, "left"):
            system.manual_adjustments_active = True
            return
    
    elif event == cv2.EVENT_MOUSEMOVE:
        # Handle pupil dragging
        if (flags & cv2.EVENT_FLAG_LBUTTON) and system.selected_eye:
            if system.selected_eye == "left":
                system.manual_left_pupil = (x, y)
            elif system.selected_eye == "right":
                system.manual_right_pupil = (x, y)
            return
        
        # Handle box dragging
        if system.dragging_box:
            if system.drag_start_pos:
                dx = x - system.drag_start_pos[0]
                dy = y - system.drag_start_pos[1]
                if system.dragging_box == "left":
                    system.left_box_offset["x"] += dx
                    system.left_box_offset["y"] += dy
                else:
                    system.right_box_offset["x"] += dx
                    system.right_box_offset["y"] += dy
                system.drag_start_pos = (x, y)
        
        # Handle edge dragging
        elif system.dragging_edge:
            left_box = system.get_stable_box(system.left_box_history)
            right_box = system.get_stable_box(system.right_box_history)
            
            if not left_box and system.left_lens_rect:
                lx1, ly1, lx2, ly2 = system.left_lens_rect
                left_box = (lx1, ly1, lx2 - lx1, ly2 - ly1)
            if not right_box and system.right_lens_rect:
                rx1, ry1, rx2, ry2 = system.right_lens_rect
                right_box = (rx1, ry1, rx2 - rx1, ry2 - ry1)
            
            if not left_box and not right_box:
                return
            
            if system.dragging_edge.startswith("left_"):
                side = "left"
                box = left_box
                offset = system.left_box_offset
                size_adj = system.left_size_adjustment
            else:
                side = "right"
                box = right_box
                offset = system.right_box_offset
                size_adj = system.right_size_adjustment
            
            bx, by, bw, bh = box
            local_x = x - offset.get("x", 0)
            local_y = y - offset.get("y", 0)
            edge = system.dragging_edge.replace(f"{side}_", "")
            
            if edge == "top":
                dy = local_y - by
                size_adj["h"] = -dy
            elif edge == "bottom":
                dy = local_y - (by + bh)
                size_adj["h"] = dy
            elif edge == "left":
                dx = local_x - bx
                size_adj["w"] = -dx
            elif edge == "right":
                dx = local_x - (bx + bw)
                size_adj["w"] = dx
            elif edge == "top_left_corner":
                dx = local_x - bx
                dy = local_y - by
                size_adj["w"] = -dx
                size_adj["h"] = -dy
            elif edge == "top_right_corner":
                dx = local_x - (bx + bw)
                dy = local_y - by
                size_adj["w"] = dx
                size_adj["h"] = -dy
            elif edge == "bottom_left_corner":
                dx = local_x - bx
                dy = local_y - (by + bh)
                size_adj["w"] = -dx
                size_adj["h"] = dy
            elif edge == "bottom_right_corner":
                dx = local_x - (bx + bw)
                dy = local_y - (by + bh)
                size_adj["w"] = dx
                size_adj["h"] = dy
            
            size_adj["w"] = int(max(-bw + 10, min(size_adj["w"], 500)))
            size_adj["h"] = int(max(-bh + 10, min(size_adj["h"], 500)))
            system.update_measurements_from_boxes()
    
    elif event == cv2.EVENT_LBUTTONUP:
        system.dragging_box = None
        system.dragging_edge = None
        system.drag_start_pos = None
        system.selected_eye = None


# ==================== MAIN ====================

def main():
    system = IntegratedEyeSpectacleSystem()
    system.run()


if __name__ == "__main__":
    main()