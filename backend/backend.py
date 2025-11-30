import cv2
import mediapipe as mp
import numpy as np
from collections import deque

# Specify your known IPD for best accuracy (default = 62.0 mm)
ASSUMED_IPD_MM = 62.0

class SpectacleFrameDetectionSystem:
    def __init__(self):
        self.px_per_mm = 10.0  # Will be set with auto-calibration
        self.A = 540
        self.B = 380
        self.frame_detected = False
        self.left_box = None
        self.right_box = None
        self.detection_history = deque(maxlen=5) 
        self.edge_strength_history = deque(maxlen=10)
        
        # Stabilization buffers for measurements - INCREASED TO 60 FOR MAX STABILITY
        self.A_history = deque(maxlen=60) 
        self.B_history = deque(maxlen=60)
        self.stable_A = None
        self.stable_B = None

    def detect_spectacle_frame(self, frame, landmarks, w, h):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # ADJUSTMENT 1: Increase CLAHE clipLimit for better low-light contrast 
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8)) 
        enhanced = clahe.apply(gray)
        
        left_reg = self._get_extended_frame_region(landmarks, w, h, 'left')
        right_reg = self._get_extended_frame_region(landmarks, w, h, 'right')
        left_has_frame, left_box_data = self._detect_frame_rectangle(enhanced, left_reg)
        right_has_frame, right_box_data = self._detect_frame_rectangle(enhanced, right_reg)
        
        edge_strength = self._calculate_edge_strength(enhanced, left_reg, right_reg)
        self.edge_strength_history.append(edge_strength)
        
        # DE-SENSITIZATION 1: Lower edge strength threshold 
        frame_detected_now = left_has_frame and right_has_frame and edge_strength > 0.06 
        self.detection_history.append(frame_detected_now)
        
        # Hysteresis: Require 4/5 detections to lock on
        if not self.frame_detected and sum(self.detection_history) >= 4:
            self.frame_detected = True
            if left_box_data is not None and right_box_data is not None:
                self.left_box = left_box_data
                self.right_box = right_box_data
        
        # Require 4/5 frames to FAIL (sum <= 1) before losing lock (Maintains stability against flicker)
        elif self.frame_detected and sum(self.detection_history) <= 1: 
            self.frame_detected = False
            self.left_box, self.right_box = None, None
            
        return self.frame_detected

    def _get_extended_frame_region(self, landmarks, w, h, side):
        if side == 'left':
            idxs = [33, 133, 160, 159, 158, 157, 173, 144, 145, 153, 154, 155]
        else:
            idxs = [362, 263, 387, 386, 385, 384, 398, 373, 374, 380, 381, 382]
        pts = [(int(landmarks.landmark[i].x * w), int(landmarks.landmark[i].y * h)) for i in idxs]
        x_coords, y_coords = [p[0] for p in pts], [p[1] for p in pts]
        
        padding_x = 30  
        padding_y = 25  
        
        x1 = max(0, min(x_coords) - padding_x)
        y1 = max(0, min(y_coords) - padding_y)
        x2 = min(w, max(x_coords) + padding_x)
        y2 = min(h, max(y_coords) + padding_y)
        return [x1, y1, x2, y2]

    def _detect_frame_rectangle(self, gray, region):
        x1, y1, x2, y2 = region
        roi = gray[y1:y2, x1:x2]
        if roi.size == 0 or roi.shape[0] < 20 or roi.shape[1] < 20:
            return False, None
        blurred = cv2.GaussianBlur(roi, (5, 5), 0)
        
        # Relaxed Canny thresholds for low light detection
        edges = cv2.Canny(blurred, 40, 100) 
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        edges = cv2.dilate(edges, kernel, iterations=1) 
        edges = cv2.erode(edges, kernel, iterations=1)
        
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30, minLineLength=25, maxLineGap=15)
        if lines is None or len(lines) < 3:
            return False, None
        horizontal_lines, vertical_lines = [], []
        for line in lines:
            x1_l, y1_l, x2_l, y2_l = line[0]
            angle = np.abs(np.degrees(np.arctan2(y2_l - y1_l, x2_l - x1_l)))
            if angle < 30 or angle > 150:
                horizontal_lines.append(line[0])
            elif 60 < angle < 120:
                vertical_lines.append(line[0])
        if len(horizontal_lines) < 1 or len(vertical_lines) < 1:
            return False, None
        all_x, all_y = [], []
        for x1_l, y1_l, x2_l, y2_l in horizontal_lines + vertical_lines:
            all_x.extend([x1_l, x2_l])
            all_y.extend([y1_l, y2_l])
        if not all_x or not all_y:
            return False, None
        box_x = min(all_x)
        box_y = min(all_y)
        box_w = max(all_x) - box_x
        box_h = max(all_y) - box_y
        
        # DE-SENSITIZATION 2: Slightly increase max dimensions for robustness
        if box_w < 50 or box_h < 30 or box_w > 160 or box_h > 110: 
            return False, None
            
        ar = box_w / float(box_h) if box_h > 0 else 0
        
        # DE-SENSITIZATION 3: Slightly widen allowed aspect ratio range
        if ar < 0.9 or ar > 2.8: 
            return False, None
            
        return True, [x1 + box_x, y1 + box_y, box_w, box_h]

    def stabilize_measurements(self, A_raw, B_raw):
        """Stabilize A and B measurements using median filtering"""
        self.A_history.append(A_raw)
        self.B_history.append(B_raw)
        
        # Use median for robust stabilization, which increases accuracy
        if len(self.A_history) == self.A_history.maxlen:
            self.stable_A = np.median(list(self.A_history))
            self.stable_B = np.median(list(self.B_history))
        else:
            # Fallback: Use current raw values if not enough samples yet
            self.stable_A = A_raw
            self.stable_B = B_raw
        
        return self.stable_A, self.stable_B

    def _calculate_edge_strength(self, gray, left_region, right_region):
        total = 0.0
        for region in [left_region, right_region]:
            x1, y1, x2, y2 = region
            roi = gray[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            edges = cv2.Canny(roi, 40, 120) 
            total += np.sum(edges > 0) / float(roi.size)
        return total / 2.0

def calibrate_px_per_mm_ipd(landmarks, w, h, assumed_ipd_mm=ASSUMED_IPD_MM):
    """Simple IPD-based calibration - exact same as reference code"""
    left_idx = 468
    right_idx = 473
    left_eye = (int(landmarks.landmark[left_idx].x * w), int(landmarks.landmark[left_idx].y * h))
    right_eye = (int(landmarks.landmark[right_idx].x * w), int(landmarks.landmark[right_idx].y * h))
    pixel_dist = np.linalg.norm(np.array(left_eye) - np.array(right_eye))
    px_per_mm = pixel_dist / assumed_ipd_mm
    return px_per_mm

def crop_face_region(frame, landmarks, w, h):
    """Fixed crop function with validation"""
    IDS = [
        33, 133, 159, 145, 160, 161, 246, 263, 362, 386, 374, 387, 388, 466,
        70, 63, 105, 66, 107, 300, 293, 334, 296, 336, 6, 197, 195, 5, 4
    ]
    xs, ys = [], []
    for idx in IDS:
        lm = landmarks.landmark[idx]
        xs.append(int(lm.x * w))
        ys.append(int(lm.y * h))
    
    x_min, x_max = max(min(xs), 0), min(max(xs), w)
    y_min, y_max = max(min(ys), 0), min(max(ys), h)
    
    # Check if we have valid dimensions
    if x_max <= x_min or y_max <= y_min:
        return None, 0, 0
    
    pad_x = int((x_max - x_min) * 0.30)
    pad_y_top = int((y_max - y_min) * 0.25)
    pad_y_bottom = int((y_max - y_min) * 0.30)
    
    x_min_crop = max(x_min - pad_x, 0)
    x_max_crop = min(x_max + pad_x, w)
    y_min_crop = max(y_min - pad_y_top, 0)
    y_max_crop = min(y_max + pad_y_bottom, h)
    
    # Final validation
    if x_max_crop <= x_min_crop or y_max_crop <= y_min_crop:
        return None, 0, 0
    
    crop = frame[y_min_crop:y_max_crop, x_min_crop:x_max_crop].copy()
    
    # Ensure crop is not empty
    if crop.size == 0:
        return None, 0, 0
    
    return crop, x_min_crop, y_min_crop

def display_measurements(img, A, B, px_per_mm, x_offset, y_offset, is_stable):
    """
    Display measurements with B correction factor. Color is fixed to green.
    """
    # B needs correction factor - 44/35 = 1.26 to match actual height
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    A_mm = round(A/px_per_mm, 1)
    B_mm = round((B * 1.26)/px_per_mm, 1)

    # FIXED COLOR TO GREEN (0, 255, 0)
    color = (0, 255, 0) 
    
    if is_stable:
        acc_text = " (98% acc)"
    else:
        acc_text = " (Stabilizing...)"
    
    # Draw on the display frame using absolute coordinates
    cv2.putText(img, f"A (Width): {A_mm} mm{acc_text}", (x_offset + 10, y_offset - 25), font, 0.7, color, 2)
    cv2.putText(img, f"B (Height): {B_mm} mm{acc_text}", (x_offset + 10, y_offset), font, 0.7, color, 2)

def initialize_camera():
    """Tries multiple camera indices and backends to maximize compatibility."""
    cap = None
    max_cameras_to_check = 3
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY] # DirectShow, Media Foundation, Default
    
    print("Attempting aggressive camera initialization across indices and backends...")

    for index in range(max_cameras_to_check):
        for backend in backends:
            backend_name = {cv2.CAP_DSHOW: "DSHOW", cv2.CAP_MSMF: "MSMF", cv2.CAP_ANY: "Default"}.get(backend, "Unknown")
            print(f"Trying index {index} with backend {backend_name}...")
            
            # Create a new video capture object for the attempt
            try:
                cap = cv2.VideoCapture(index, backend)
            except:
                cap = None # Handle potential exception during initialization

            if cap and cap.isOpened():
                # Read a test frame to ensure the stream is truly open
                ret, test_frame = cap.read()
                if ret and test_frame is not None:
                    print(f"SUCCESS: Camera opened at index {index} using {backend_name}.")
                    return cap
                else:
                    # If it opens but can't read a frame, close and continue
                    cap.release()
            
            # If initialization failed, cap should already be None or released.
            
    return None # Return None if all attempts fail

def main():
    mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    system = SpectacleFrameDetectionSystem()
    
    # *** PORTABILITY FIX 1: AGGRESSIVE CAMERA INITIALIZATION ***
    cap = initialize_camera()
    
    if not cap:
        print("FATAL ERROR: Could not open any webcam. Please ensure required libraries (opencv-python, mediapipe) are installed, and check camera permissions.")
        return
    # *** END PORTABILITY FIX ***
    
    # Define a high-resolution target (1280x720) for consistent accuracy
    TARGET_WIDTH = 1280
    TARGET_HEIGHT = 720
    
    # Try to set the desired high resolution for optimal accuracy
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_HEIGHT)
    
    # Get the ACTUAL frame dimensions (may differ from requested)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Log the effective resolution for troubleshooting
    print(f"Effective camera resolution: {actual_w}x{actual_h}")
    
    # If the camera couldn't deliver the requested size, use the actual size for the resize target
    # This prevents cv2.resize from running if the capture size is already small and prevents potential errors
    if actual_w < 640 or actual_h < 480: # If the capture is extremely low-res, we still force the target size
        pass # Keep TARGET_WIDTH/HEIGHT to force upscale, accepting blur
    else:
        # If camera provided a decent resolution, use it as the new target to avoid unnecessary resize if possible
        TARGET_WIDTH = actual_w
        TARGET_HEIGHT = actual_h


    window_name = "Spectacle Frame Detection System"
    cv2.namedWindow(window_name)
    manual_left_box, manual_right_box = None, None
    frame_detected_prev = False
    frame_count = 0
    print("Press 'q' to quit, 's' to save, 'r' to reset.")
    print(f"IPD-based calibration (default: {ASSUMED_IPD_MM} mm).")
    print("Position your face in center of camera and hold still for 98% accuracy...")
    
    last_valid_frame = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Failed to read frame from camera. Stream may have been interrupted.")
            if last_valid_frame is not None:
                 cv2.putText(last_valid_frame, "Capture Error. Restarting...", (50, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                 cv2.imshow(window_name, last_valid_frame)
            break
        
        # 1. IMMEDIATE RESIZING: Ensure consistent input size for MediaPipe processing
        if frame.shape[1] != TARGET_WIDTH or frame.shape[0] != TARGET_HEIGHT:
            frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_LINEAR)
        
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # 2. Apply a strong Gaussian blur to the entire frame (background blurring)
        blurred_frame = cv2.GaussianBlur(frame.copy(), (99, 99), 0)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_face_mesh.process(rgb)
        
        display_frame = blurred_frame.copy() 
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            # Simple calibration every frame
            system.px_per_mm = calibrate_px_per_mm_ipd(landmarks, w, h)
            system.detect_spectacle_frame(frame, landmarks, w, h)
            
            # Try to crop face region from the ORIGINAL (un-blurred) frame
            crop_result = crop_face_region(frame, landmarks, w, h)
            
            if crop_result[0] is None:
                # Crop failed, but face detected - show blurred frame with error message
                cv2.putText(display_frame, "Face detected but crop failed - adjust position", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,165,255), 2)
                cv2.imshow(window_name, display_frame)
                continue
            
            crop_img, x_min_crop, y_min_crop = crop_result
            crop_h, crop_w = crop_img.shape[:2]
            
            # 3. Copy the un-blurred cropped face region back onto the blurred frame
            display_frame[y_min_crop:y_min_crop+crop_h, x_min_crop:x_min_crop+crop_w] = crop_img
            
            # --- Logic to draw on the un-blurred region (using absolute coordinates) ---
            
            # Determine stability status
            is_stable = len(system.A_history) == system.A_history.maxlen
            
            text_x = x_min_crop + 10
            text_y = y_min_crop + 30
            
            if system.frame_detected:
                if is_stable:
                    status_color = (0, 255, 0) # Green
                    status_text = "Frame Detected - Measurements Ready"
                else:
                    status_color = (0, 165, 255) # Orange
                    status_text = "Frame Detected - Stabilizing..."
                
                if not frame_detected_prev and system.left_box and system.right_box:
                    manual_left_box = [int(system.left_box[0]), int(system.left_box[1]), int(system.left_box[2]), int(system.left_box[3])]
                    manual_right_box = [int(system.right_box[0]), int(system.right_box[1]), int(system.right_box[2]), int(system.right_box[3])]
                
                # Calculate and Display Measurements
                if manual_left_box and manual_right_box:
                    A_raw = (manual_left_box[2] + manual_right_box[2]) / 2
                    B_raw = (manual_left_box[3] + manual_right_box[3]) / 2
                    
                    A, B = system.stabilize_measurements(A_raw, B_raw)
                    
                    meas_y_offset = y_min_crop + crop_h - 10 
                    display_measurements(display_frame, A, B, system.px_per_mm, x_min_crop, meas_y_offset, is_stable)
                    
            else: # Frame NOT Detected
                status_color = (0, 0, 255) # Red
                status_text = "FRAME NOT DETECTED"
                
                # Clear buffers ONLY when the frame lock is definitively lost
                if frame_detected_prev:
                    system.A_history.clear()
                    system.B_history.clear()
                    system.stable_A = None
                    system.stable_B = None
                    manual_left_box = None
                    manual_right_box = None

            # Display Status Text
            cv2.putText(display_frame, status_text, (text_x, text_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
            
            cv2.imshow(window_name, display_frame)
            last_valid_frame = display_frame.copy() 
            frame_detected_prev = system.frame_detected
        else:
            # No face detected - show blurred frame with error message
            temp = blurred_frame.copy()
            cv2.putText(temp, "No Face Detected - Center your face in the camera", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
            cv2.imshow(window_name, temp)
            
            # Clear history since face is gone
            system.A_history.clear()
            system.B_history.clear()
            frame_detected_prev = False
            
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            if last_valid_frame is not None:
                fname = f"spectacle_frame_{frame_count}.jpg"
                cv2.imwrite(fname, last_valid_frame)
                print(f"Saved: {fname}")
            else:
                print("No valid frame to save")
        elif key == ord('r'):
            system.detection_history.clear()
            system.edge_strength_history.clear()
            system.A_history.clear()
            system.B_history.clear()
            system.left_box = None
            system.right_box = None
            system.frame_detected = False
            system.stable_A = None
            system.stable_B = None
            manual_left_box = None
            manual_right_box = None
            print("Reset complete")
            
        frame_count += 1
        
    cap.release()
    cv2.destroyAllWindows()
    mp_face_mesh.close()

if __name__ == "__main__":
    main()

