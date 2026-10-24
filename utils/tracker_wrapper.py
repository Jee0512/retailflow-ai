import os
import cv2
import time
import threading
import numpy as np
import pandas as pd
from collections import defaultdict

class AsyncYOLOTracker:
    def __init__(self, model, conf=0.35, imgsz=320):
        self.model = model
        self.conf = conf
        self.imgsz = imgsz
        self.lock = threading.Lock()
        self.latest_frame = None
        self.latest_frame_idx = -1
        self.latest_detections = []
        self.running = True
        self.worker = threading.Thread(target=self._run_inference, daemon=True)
        self.worker.start()

    def _run_inference(self):
        while self.running:
            frame_to_process = None
            f_idx = -1
            with self.lock:
                if self.latest_frame is not None and self.latest_frame_idx > -1:
                    frame_to_process = self.latest_frame
                    f_idx = self.latest_frame_idx
                    self.latest_frame = None  # Consume frame
            
            if frame_to_process is not None:
                try:
                    h, w = frame_to_process.shape[:2]
                    infer_w, infer_h = 480, 270
                    small_frame = cv2.resize(frame_to_process, (infer_w, infer_h))
                    results = self.model.track(
                        source=small_frame,
                        persist=True,
                        classes=[0],
                        conf=self.conf,
                        verbose=False,
                        tracker="bytetrack.yaml",
                        imgsz=self.imgsz
                    )
                    scale_x = w / float(infer_w)
                    scale_y = h / float(infer_h)
                    
                    boxes = []
                    if results and len(results) > 0 and results[0].boxes is not None:
                        for box in results[0].boxes:
                            if box.id is not None:
                                sx1, sy1, sx2, sy2 = box.xyxy[0].cpu().numpy()
                                tid = int(box.id.cpu().numpy()[0])
                                x1 = int(sx1 * scale_x)
                                y1 = int(sy1 * scale_y)
                                x2 = int(sx2 * scale_x)
                                y2 = int(sy2 * scale_y)
                                boxes.append((x1, y1, x2, y2, tid, None))
                    
                    with self.lock:
                        self.latest_detections = boxes
                except Exception:
                    pass
            else:
                time.sleep(0.005)

    def submit_frame(self, frame, frame_idx):
        with self.lock:
            self.latest_frame = frame
            self.latest_frame_idx = frame_idx

    def get_detections(self):
        with self.lock:
            return list(self.latest_detections)

    def stop(self):
        self.running = False

class CCTVTracker:
    def __init__(self, use_yolo=False, model_name="yolov8n.pt"):
        """
        Initializes the CCTV Tracking Module.
        use_yolo: If True, uses Ultralytics YOLOv8. If False, runs in emulation mode.
        """
        self.use_yolo = use_yolo
        self.model_name = model_name
        self.model = None
        self.async_yolo = None
        
        # Historical paths for drawing trails: {track_id: [(x, y), (x, y), ...]}
        self.history = defaultdict(list)
        
        # Zone definitions matching the synthetic generator
        self.zones = {
            "Entrance": (50, 500, 150, 680),
            "Aisle A (Produce)": (150, 100, 450, 250),
            "Aisle B (Apparel)": (550, 100, 850, 250),
            "Aisle C (Electronics)": (350, 320, 750, 470),
            "Checkout 1": (900, 500, 1050, 650),
            "Checkout 2": (1100, 500, 1250, 650),
            "Exit": (50, 300, 150, 450)
        }
        
        # Tracking states (CUSTOMERS ONLY)
        # customer_dwell: {track_id: {zone_name: frame_count}}
        self.customer_dwell = defaultdict(lambda: defaultdict(int))
        # customer_first_seen: {track_id: frame_index}
        self.customer_first_seen = {}
        # customer_last_seen: {track_id: frame_index}
        self.customer_last_seen = {}
        
        # Bounding box colors for unique track IDs
        self.colors = {}
        
        # Emulation data holder
        self.emulation_df = None
        
        # Staff Filtering Configuration
        self.filter_staff = True
        self.staff_uniform_preset = "Red (Target/Coles)"
        self.staff_custom_color_rgb = (220, 40, 40)
        self.staff_color_threshold = 0.20 # fraction of torso pixels matching uniform
        self.show_staff_boxes = True
        
        # Staff tracking state
        self.staff_votes = defaultdict(list) # track_id -> list of 0/1 votes
        self.confirmed_roles = {} # track_id -> "STAFF" or "CUSTOMER"
        self.staff_first_seen = {} # track_id -> frame_index
        self.staff_last_seen = {} # track_id -> frame_index
        self.active_staff_count = 0
        self.active_customer_count = 0
        self.last_active_detections = []
        
    @property
    def staff_tracks(self):
        """Returns set of all track IDs classified as staff."""
        staff_set = set(self.staff_first_seen.keys())
        for tid, role in self.confirmed_roles.items():
            if role == "STAFF":
                staff_set.add(tid)
        return staff_set

    def load_yolo_model(self):
        """Loads the Ultralytics YOLOv8 Model with background asynchronous inference worker."""
        try:
            from ultralytics import YOLO
            print(f"Loading YOLOv8 model: {self.model_name}...")
            self.model = YOLO(self.model_name)
            if self.async_yolo is not None:
                try:
                    self.async_yolo.stop()
                except Exception:
                    pass
            self.async_yolo = AsyncYOLOTracker(self.model, conf=0.35, imgsz=320)
            print("YOLOv8 and Async Worker loaded successfully.")
        except Exception as e:
            print(f"Error loading YOLOv8: {e}")
            print("Defaulting back to emulated tracking.")
            self.use_yolo = False

    def load_emulation_data(self, csv_path):
        """Loads pre-generated synthetic detections to emulate YOLOv8 tracking."""
        if os.path.exists(csv_path):
            self.emulation_df = pd.read_csv(csv_path)
            self.use_yolo = False
            print(f"Loaded emulation data from {csv_path}")
            return True
        else:
            print(f"Emulation CSV not found at {csv_path}. Real-time YOLOv8 will be forced.")
            self.use_yolo = True
            self.load_yolo_model()
            return False

    def get_color(self, track_id):
        """Returns a stable, distinct RGB color for a given track ID."""
        if track_id not in self.colors:
            self.colors[track_id] = (
                int(hash(str(track_id)) % 200) + 55,
                int(hash(str(track_id * 2)) % 200) + 55,
                int(hash(str(track_id * 3)) % 200) + 55
            )
        return self.colors[track_id]

    def check_zone(self, x, y):
        """Checks which store zone contains the coordinate (x, y). Returns zone name or None."""
        for zone_name, (x1, y1, x2, y2) in self.zones.items():
            if x1 <= x <= x2 and y1 <= y <= y2:
                return zone_name
        return None

    def _get_uniform_mask(self, hsv_torso):
        """Generates a binary mask of pixels in hsv_torso matching the selected staff uniform preset."""
        preset = self.staff_uniform_preset
        
        if "Red" in preset:
            # Red wraps around 0 and 180 in HSV
            lower1 = np.array([0, 70, 50], dtype=np.uint8)
            upper1 = np.array([10, 255, 255], dtype=np.uint8)
            mask1 = cv2.inRange(hsv_torso, lower1, upper1)
            
            lower2 = np.array([170, 70, 50], dtype=np.uint8)
            upper2 = np.array([179, 255, 255], dtype=np.uint8)
            mask2 = cv2.inRange(hsv_torso, lower2, upper2)
            return cv2.bitwise_or(mask1, mask2)
            
        elif "Blue" in preset:
            lower = np.array([95, 60, 40], dtype=np.uint8)
            upper = np.array([130, 255, 255], dtype=np.uint8)
            return cv2.inRange(hsv_torso, lower, upper)
            
        elif "Green" in preset or "Neon" in preset:
            lower = np.array([35, 60, 50], dtype=np.uint8)
            upper = np.array([85, 255, 255], dtype=np.uint8)
            return cv2.inRange(hsv_torso, lower, upper)
            
        elif "Black" in preset:
            lower = np.array([0, 0, 0], dtype=np.uint8)
            upper = np.array([180, 255, 50], dtype=np.uint8)
            return cv2.inRange(hsv_torso, lower, upper)
            
        elif "Orange" in preset:
            lower = np.array([10, 100, 70], dtype=np.uint8)
            upper = np.array([25, 255, 255], dtype=np.uint8)
            return cv2.inRange(hsv_torso, lower, upper)
            
        elif "Custom" in preset:
            r, g, b = self.staff_custom_color_rgb
            custom_bgr = np.uint8([[[b, g, r]]])
            custom_hsv = cv2.cvtColor(custom_bgr, cv2.COLOR_BGR2HSV)[0][0]
            ch, cs, cv = int(custom_hsv[0]), int(custom_hsv[1]), int(custom_hsv[2])
            
            h_tol = 18
            s_tol = 80
            v_tol = 80
            
            h_low = (ch - h_tol) % 180
            h_high = (ch + h_tol) % 180
            s_low = max(40, cs - s_tol)
            s_high = min(255, cs + s_tol)
            v_low = max(40, cv - v_tol)
            v_high = min(255, cv + v_tol)
            
            if h_low > h_high:
                m1 = cv2.inRange(hsv_torso, np.array([0, s_low, v_low], dtype=np.uint8), np.array([h_high, s_high, v_high], dtype=np.uint8))
                m2 = cv2.inRange(hsv_torso, np.array([h_low, s_low, v_low], dtype=np.uint8), np.array([179, s_high, v_high], dtype=np.uint8))
                return cv2.bitwise_or(m1, m2)
            else:
                return cv2.inRange(hsv_torso, np.array([h_low, s_low, v_low], dtype=np.uint8), np.array([h_high, s_high, v_high], dtype=np.uint8))
                
        # Default fallback to Red
        lower = np.array([0, 70, 50], dtype=np.uint8)
        upper = np.array([10, 255, 255], dtype=np.uint8)
        return cv2.inRange(hsv_torso, lower, upper)

    def check_torso_uniform(self, frame, x1, y1, x2, y2):
        """
        Extracts the upper torso region of a detected person and computes
        the fraction of pixels matching the staff uniform color.
        """
        h = y2 - y1
        w = x2 - x1
        if h < 20 or w < 10:
            return 0.0
            
        ty1 = max(0, y1 + int(h * 0.15))
        ty2 = min(frame.shape[0], y1 + int(h * 0.60))
        tx1 = max(0, x1 + int(w * 0.20))
        tx2 = min(frame.shape[1], x2 - int(w * 0.20))
        
        if ty2 <= ty1 or tx2 <= tx1:
            return 0.0
            
        torso_crop = frame[ty1:ty2, tx1:tx2]
        if torso_crop.size == 0:
            return 0.0
            
        hsv_torso = cv2.cvtColor(torso_crop, cv2.COLOR_BGR2HSV)
        mask = self._get_uniform_mask(hsv_torso)
        total_pixels = torso_crop.shape[0] * torso_crop.shape[1]
        if total_pixels == 0:
            return 0.0
            
        match_pixels = cv2.countNonZero(mask)
        return match_pixels / float(total_pixels)

    def classify_track(self, frame, x1, y1, x2, y2, track_id, metadata_role=None):
        """
        Classifies a track as 'STAFF' or 'CUSTOMER'.
        Uses metadata_role if provided (from synthetic data), or CV color analysis on the video.
        Uses temporal voting across frames for stability.
        """
        if not self.filter_staff:
            return "CUSTOMER"
            
        if metadata_role is not None and not pd.isna(metadata_role):
            role_str = str(metadata_role).strip().lower()
            if role_str in ["staff", "employee"]:
                self.confirmed_roles[track_id] = "STAFF"
                return "STAFF"
            elif role_str in ["customer", "shopper"]:
                self.confirmed_roles[track_id] = "CUSTOMER"
                return "CUSTOMER"
                
        # Fast path: if track role is already firmly established with sufficient votes, avoid re-processing torso HSV
        if track_id in self.confirmed_roles and len(self.staff_votes[track_id]) >= 8:
            return self.confirmed_roles[track_id]

        # Computer Vision Torso Uniform Check
        ratio = self.check_torso_uniform(frame, x1, y1, x2, y2)
        is_match = 1 if ratio >= self.staff_color_threshold else 0
        
        votes = self.staff_votes[track_id]
        votes.append(is_match)
        if len(votes) > 15:
            votes.pop(0)
            
        staff_confidence = sum(votes) / float(len(votes))
        
        # Hysteresis thresholding to prevent classification jitter
        if len(votes) >= 2 and staff_confidence >= 0.35:
            self.confirmed_roles[track_id] = "STAFF"
            return "STAFF"
        elif len(votes) >= 4 and staff_confidence < 0.20:
            self.confirmed_roles[track_id] = "CUSTOMER"
            return "CUSTOMER"
            
        return self.confirmed_roles.get(track_id, "CUSTOMER" if staff_confidence < 0.35 else "STAFF")

    def process_frame(self, frame, frame_idx, conf_threshold=0.3, skip_detection=False):
        """
        Processes a single frame. Detects/tracks people, differentiates staff from customers,
        computes dwell times and paths for customers only, and draws distinct annotations.
        Supports skip_detection to reuse previous detections on intermediate frames for ultra-low latency.
        
        Returns:
            annotated_frame: Frame with annotations.
            customer_centers: List of (x, y) coordinates of active customers in this frame.
            active_customer_count: Number of active customers in this frame.
        """
        active_detections = []
        if self.use_yolo and self.model is not None:
            if self.async_yolo is None:
                self.async_yolo = AsyncYOLOTracker(self.model, conf=conf_threshold, imgsz=320)
            self.async_yolo.conf = conf_threshold
            self.async_yolo.submit_frame(frame, frame_idx)
            raw_dets = self.async_yolo.get_detections()
            if raw_dets:
                active_detections = raw_dets
                self.last_active_detections = list(active_detections)
            elif self.last_active_detections:
                active_detections = list(self.last_active_detections)
        
        elif skip_detection and self.last_active_detections:
            active_detections = list(self.last_active_detections)
        
        elif self.emulation_df is not None:
            # Retrieve detections for current frame index from mock csv
            frame_rows = self.emulation_df[self.emulation_df["frame_idx"] == frame_idx]
            for _, row in frame_rows.iterrows():
                role = row.get("role", None) if "role" in row else None
                active_detections.append((
                    int(row["x_min"]),
                    int(row["y_min"]),
                    int(row["x_max"]),
                    int(row["y_max"]),
                    int(row["track_id"]),
                    role
                ))
            self.last_active_detections = list(active_detections)
                
        annotated_frame = frame.copy()
        customer_centers = []
        active_customers = 0
        active_staff = 0
        
        for det in active_detections:
            x1, y1, x2, y2, track_id = det[0], det[1], det[2], det[3], det[4]
            metadata_role = det[5] if len(det) > 5 else None
            
            # Classify role
            role = self.classify_track(frame, x1, y1, x2, y2, track_id, metadata_role)
            
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            current_zone = self.check_zone(cx, cy)
            
            if role == "STAFF":
                active_staff += 1
                if track_id not in self.staff_first_seen:
                    self.staff_first_seen[track_id] = frame_idx
                self.staff_last_seen[track_id] = frame_idx
                
                # NOTE: Staff positions are NOT appended to customer_centers!
                # This guarantees that the heatmap will NEVER accumulate heat from staff!
                
                if self.show_staff_boxes:
                    # Draw distinct Amber/Gold box for staff
                    staff_box_color = (0, 165, 255) # BGR Amber
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), staff_box_color, 2)
                    
                    # Staff badge header
                    badge_lbl = f"STAFF #{track_id} (Filtered)"
                    (tw, th), _ = cv2.getTextSize(badge_lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
                    cv2.rectangle(annotated_frame, (x1, max(0, y1 - 22)), (x1 + tw + 10, y1), staff_box_color, -1)
                    cv2.putText(annotated_frame, badge_lbl, (x1 + 5, max(12, y1 - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
            else:
                active_customers += 1
                customer_centers.append((cx, cy))
                
                # Store tracking timestamps
                if track_id not in self.customer_first_seen:
                    self.customer_first_seen[track_id] = frame_idx
                self.customer_last_seen[track_id] = frame_idx
                
                # Update zone occupancy/dwell times
                if current_zone:
                    self.customer_dwell[track_id][current_zone] += 1
                    
                # Add to trail history
                self.history[track_id].append((cx, cy))
                if len(self.history[track_id]) > 25:
                    self.history[track_id].pop(0)
                    
                # Draw trail line
                trail_pts = self.history[track_id]
                for i in range(1, len(trail_pts)):
                    thickness = int(np.sqrt(25 / float(i + 1)) * 2)
                    color = self.get_color(track_id)
                    cv2.line(annotated_frame, trail_pts[i - 1], trail_pts[i], color, thickness)
                    
                # Draw bounding box and ID label
                color = self.get_color(track_id)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                
                lbl = f"ID: {track_id}"
                if current_zone:
                    lbl += f" ({current_zone.split(' ')[0]})"
                (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                cv2.rectangle(annotated_frame, (x1, max(0, y1 - 20)), (x1 + tw + 10, y1), color, -1)
                cv2.putText(annotated_frame, lbl, (x1 + 5, max(12, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
                
        self.active_customer_count = active_customers
        self.active_staff_count = active_staff
        
        return annotated_frame, customer_centers, active_customers

    def draw_zones(self, frame):
        """Draws zone boundaries transparently on the frame for visualization purposes."""
        overlay = frame.copy()
        for zone_name, (x1, y1, x2, y2) in self.zones.items():
            color = (255, 255, 255)
            if "Aisle" in zone_name:
                color = (52, 152, 219)
            elif "Checkout" in zone_name:
                color = (231, 76, 60)
            elif "Entrance" in zone_name or "Exit" in zone_name:
                color = (46, 204, 113)
                
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            cv2.putText(overlay, zone_name, (x1 + 5, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            
        return cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)

    def get_analytics(self, fps=30.0):
        """
        Compiles tracking analytics data into a summary report.
        Ensures footfall, dwell time, and zone metrics reflect CUSTOMERS ONLY across all video sources.
        """
        if not self.customer_first_seen:
            return pd.DataFrame(), {
                "total_footfall": 0,
                "avg_dwell_time": 0.0,
                "busiest_zone": "None",
                "zone_counts": {},
                "zone_dwells": {},
                "total_staff_detected": len(self.staff_first_seen),
                "active_staff": self.active_staff_count,
                "filter_staff_enabled": self.filter_staff
            }
            
        records = []
        for track_id in self.customer_first_seen.keys():
            first_frame = self.customer_first_seen[track_id]
            last_frame = self.customer_last_seen[track_id]
            total_duration = (last_frame - first_frame) / fps
            
            record = {
                "Shopper_ID": track_id,
                "First_Seen_Frame": first_frame,
                "Last_Seen_Frame": last_frame,
                "Total_Dwell_Time_Sec": round(total_duration, 1)
            }
            
            zone_dwells = {}
            visited_list = []
            max_dwell_zone = "None"
            max_dwell = 0
            
            for zone_name in self.zones.keys():
                frames_in_zone = self.customer_dwell[track_id][zone_name]
                dwell_sec = round(frames_in_zone / fps, 1)
                
                if dwell_sec > 0:
                    zone_dwells[zone_name] = dwell_sec
                    visited_list.append(zone_name)
                    
                    if dwell_sec > max_dwell:
                        max_dwell = dwell_sec
                        max_dwell_zone = zone_name
                        
            record.update({
                "Primary_Zone": max_dwell_zone,
                "Visited_Zones": ", ".join(visited_list) if visited_list else "None",
                **zone_dwells
            })
            records.append(record)
            
        df = pd.DataFrame(records)
        total_footfall = len(df)
        avg_dwell = round(df["Total_Dwell_Time_Sec"].mean(), 1) if total_footfall > 0 else 0.0
        
        zone_total_counts = defaultdict(int)
        zone_total_dwell = defaultdict(float)
        for track_id in self.customer_first_seen.keys():
            for zone_name in self.zones.keys():
                frames_in_zone = self.customer_dwell[track_id][zone_name]
                dwell_sec = round(frames_in_zone / fps, 1)
                if dwell_sec > 0:
                    zone_total_counts[zone_name] += 1
                    zone_total_dwell[zone_name] += dwell_sec
                    
        busiest_zone = "None"
        if zone_total_dwell:
            busiest_zone = max(zone_total_dwell, key=zone_total_dwell.get)
            
        metrics = {
            "total_footfall": total_footfall,
            "avg_dwell_time": avg_dwell,
            "busiest_zone": busiest_zone,
            "zone_counts": dict(zone_total_counts),
            "zone_dwells": dict(zone_total_dwell),
            "total_staff_detected": len(self.staff_first_seen),
            "active_staff": self.active_staff_count,
            "filter_staff_enabled": self.filter_staff
        }
            
        return df, metrics

    def reset(self):
        """Resets all tracking histories and records."""
        self.history.clear()
        self.customer_dwell.clear()
        self.customer_first_seen.clear()
        self.customer_last_seen.clear()
        self.colors.clear()
        self.staff_votes.clear()
        self.confirmed_roles.clear()
        self.staff_first_seen.clear()
        self.staff_last_seen.clear()
        self.active_staff_count = 0
        self.active_customer_count = 0
        self.last_active_detections = []
        if self.async_yolo is not None:
            try:
                with self.async_yolo.lock:
                    self.async_yolo.latest_detections = []
                    self.async_yolo.latest_frame = None
                    self.async_yolo.latest_frame_idx = -1
            except Exception:
                pass
