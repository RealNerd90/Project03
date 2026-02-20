import torch
import cv2
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image, ImageDraw, ImageFont
import os
import re
import csv
import json
import math
from datetime import datetime, timezone, timedelta
import asyncio
import sys
if sys.platform == "win32":
    try:
        from winrt.windows.devices.geolocation import Geolocator, PositionStatus, GeolocationAccessStatus
        from winrt.windows.foundation import IAsyncOperation
    except ImportError:
        print("Could not import Geolocation API. Location check may not work.")
        Geolocator = None

# ---------------------------------------------------------------------------
# Location-based attendance (admin-set radius)
# ---------------------------------------------------------------------------
# ADMIN: Edit the values below to set the attendance location and radius.
# Later, this will be moved to a proper admin interface.
LOCATION_CONFIG_PATH = "location_config.json"
DEFAULT_LOCATION_CONFIG = {
    "enabled": True,  # ADMIN: Set to True to enable location-based attendance, False to disable
    "center_lat": 26.11800075650252,    # ADMIN: Set your office/classroom latitude here
    "center_lon": 91.8136276152452,  # ADMIN: Set your office/classroom longitude here
    "radius_meters": 100,  # ADMIN: Set allowed radius in meters (e.g. 100 = 100 meters)
}

def _load_location_config():
    """
    Load admin location config. Returns dict with enabled, center_lat, center_lon, radius_meters.
    Priority: Code DEFAULT_LOCATION_CONFIG (if enabled=True) > location_config.json > defaults.
    """
    # Always start with code defaults (so code edits take effect)
    cfg = dict(DEFAULT_LOCATION_CONFIG)
    
    # If JSON file exists, merge it (but code values take precedence if enabled in code)
    if os.path.exists(LOCATION_CONFIG_PATH):
        try:
            with open(LOCATION_CONFIG_PATH, "r", encoding="utf-8") as f:
                json_cfg = json.load(f)
                # Only override from JSON if code has enabled=False (allows JSON to override disabled state)
                if not cfg.get("enabled", False):
                    cfg["enabled"] = bool(json_cfg.get("enabled", False))
                # Always allow JSON to override coordinates/radius (for flexibility)
                if "center_lat" in json_cfg:
                    cfg["center_lat"] = float(json_cfg["center_lat"])
                if "center_lon" in json_cfg:
                    cfg["center_lon"] = float(json_cfg["center_lon"])
                if "radius_meters" in json_cfg:
                    cfg["radius_meters"] = max(10, float(json_cfg["radius_meters"]))
        except Exception as e:
            print(f"⚠️  Warning: Could not load location_config.json: {e}. Using code defaults.")
    
    return cfg


def _save_location_config(cfg):
    """Save admin location config to JSON."""
    with open(LOCATION_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


def _haversine_meters(lat1, lon1, lat2, lon2):
    """Return distance in meters between two (lat, lon) points."""
    R = 6371000  # Earth radius in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


async def _get_current_location():
    """
    Get current location using Windows Geolocation API.
    Returns (lat, lon) or (None, None) on failure.
    """
    if sys.platform != "win32" or Geolocator is None:
        return _get_current_location_ip()  # Fallback to IP for non-windows

    try:
        access_status = await Geolocator.request_access_async()
        if access_status != GeolocationAccessStatus.ALLOWED:
            print("⚠️  Location access denied in Windows settings.")
            return (None, None)

        geolocator = Geolocator()
        print("🌍 Getting current position... (may take a moment)")
        position = await geolocator.get_geoposition_async()
        lat = position.coordinate.latitude
        lon = position.coordinate.longitude
        print(f"  -> GPS Location: Lat={lat:.6f}, Lon={lon:.6f} (Accuracy: {position.coordinate.accuracy:.0f}m)")
        return (float(lat), float(lon))
    except Exception as e:
        print(f"⚠️  Could not get GPS location: {e}")
        return (None, None)

async def check_location_allowed():
    """
    If location-based attendance is enabled, check if current device is within
    admin-set radius. Returns (allowed: bool, message: str).
    """
    cfg = _load_location_config()
    if not cfg["enabled"]:
        return True, "Location check disabled."
    lat, lon = await _get_current_location()
    if lat is None or lon is None:
        return False, "Could not get current location. Check internet or try again."
    dist = _haversine_meters(lat, lon, cfg["center_lat"], cfg["center_lon"])
    if dist <= cfg["radius_meters"]:
        return True, f"Within range ({dist:.0f} m)."
    return False, f"Outside allowed radius ({dist:.0f} m > {cfg['radius_meters']} m)."

class AttendanceSystem:
    def __init__(self, database_path='sample_faces', threshold=0.6):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize MTCNN for face detection
        # Tuned parameters to reduce false positives
        self.mtcnn = MTCNN(
            keep_all=True,  # Detect all faces (we'll filter later)
            min_face_size=50, # Increased from 20 to ignore small background faces
            thresholds=[0.7, 0.8, 0.8], # Increased thresholds for stricter detection
            factor=0.709,
            device=self.device
        )
        
        # Initialize FaceNet for recognition
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        
        self.database_path = database_path
        self.threshold = threshold
        # Map: student_name -> list of embeddings (supports multiple angles per student)
        self.embeddings_db = {}
        self.names_db = []

        # Fast search index (rebuilt in load_database)
        # _db_matrix: [N, 512] tensor on self.device
        # _db_names_flat: list[str] length N (parallel to _db_matrix rows)
        self._db_matrix = None
        self._db_names_flat = []
        
        if not os.path.exists('output'):
            os.makedirs('output')
            
        self.load_database()

    def load_database(self):
        """
        Load reference faces from the database folder.

        New layout (preferred):
          sample_faces/<student_name>/<angle>.jpg
          e.g. sample_faces/Alice/left.jpg, right.jpg, up.jpg, down.jpg, front.jpg

        Backward compatible layout (still supported):
          sample_faces/<student_name>__<tag>.jpg
        """
        if not os.path.exists(self.database_path):
            os.makedirs(self.database_path)
            print(f"Created database folder: {self.database_path}")
            return

        # --- Legacy migration: move flat files into per-student folders ---
        # Supports:
        # - "<name>__<tag>.jpg" (older convention)
        # - "<name>_<tag>.jpg"  (common manual naming)
        # - "<name>-<tag>.jpg" / "<name> <tag>.jpg"
        allowed_tags = {"left", "right", "up", "down", "front"}
        migrated = 0
        try:
            for entry in os.listdir(self.database_path):
                full_path = os.path.join(self.database_path, entry)
                if not os.path.isfile(full_path):
                    continue
                if not entry.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue

                stem, ext = os.path.splitext(entry)
                ext = ext.lower()
                student = None
                tag = None

                if '__' in stem:
                    student_part, tag_part = stem.split('__', 1)
                    student = student_part.strip()
                    tag = tag_part.strip().lower()
                else:
                    m = re.match(r"^(?P<student>.+?)[ _-]+(?P<tag>left|right|up|down|front)$", stem, flags=re.IGNORECASE)
                    if m:
                        student = m.group("student").strip()
                        tag = m.group("tag").strip().lower()

                if not student or not tag or tag not in allowed_tags:
                    continue

                dest_dir = os.path.join(self.database_path, student)
                os.makedirs(dest_dir, exist_ok=True)

                dest_path = os.path.join(dest_dir, f"{tag}{ext}")
                if os.path.exists(dest_path):
                    # Don't overwrite existing refs; keep both with suffix.
                    i = 2
                    while True:
                        candidate = os.path.join(dest_dir, f"{tag}_{i}{ext}")
                        if not os.path.exists(candidate):
                            dest_path = candidate
                            break
                        i += 1

                os.replace(full_path, dest_path)
                migrated += 1
        except Exception:
            # Migration is best-effort; loader below still works for folder layout.
            pass

        print(f"\nLoading face database from {self.database_path}...")
        self.embeddings_db = {}
        self.names_db = []
        loaded_counts = {}  # base_name -> number of reference images loaded
        flat_embeddings = []  # list[Tensor(512,)]
        flat_names = []       # list[str] parallel to flat_embeddings

        # Cache embeddings to avoid re-computing every run (huge speedup).
        # Cache format: { "files": { filepath: { "mtime": float, "name": str, "emb": Tensor } } }
        cache_path = os.path.join(self.database_path, "_embeddings_cache.pt")
        cache = {"files": {}}
        try:
            if os.path.exists(cache_path):
                cache = torch.load(cache_path, map_location="cpu")
                if not isinstance(cache, dict) or "files" not in cache or not isinstance(cache["files"], dict):
                    cache = {"files": {}}
        except Exception:
            cache = {"files": {}}
        cache_files = cache.get("files", {})
        used_cache_files = set()

        def _iter_reference_images():
            # Preferred: directories per student
            for entry in os.listdir(self.database_path):
                full_path = os.path.join(self.database_path, entry)
                if os.path.isdir(full_path):
                    student = entry.strip()
                    for sub in os.listdir(full_path):
                        if sub.lower().endswith(('.png', '.jpg', '.jpeg')):
                            # Only load images that look like angle refs (keeps DB clean/fast),
                            # but accept common naming patterns:
                            #   left.jpg / right.jpg / up.jpg / down.jpg / front.jpg
                            #   Amar_left.jpg, left_2.jpg, front-1.jpg, etc.
                            base = os.path.splitext(sub)[0].lower().strip()
                            tag = None
                            if base in ("left", "right", "up", "down", "front"):
                                tag = base
                            elif '__' in base:
                                # e.g. amar__left
                                tag = base.split('__', 1)[1].strip().lower()
                            else:
                                m = re.match(r"^.+?[ _-]+(?P<tag>left|right|up|down|front)(?:[ _-]?\d+)?$", base, flags=re.IGNORECASE)
                                if m:
                                    tag = m.group("tag").strip().lower()

                            if tag not in ("left", "right", "up", "down", "front"):
                                continue
                            yield student, os.path.join(full_path, sub), f"{entry}/{sub}"
                else:
                    # Backward compatible flat files
                    if entry.lower().endswith(('.png', '.jpg', '.jpeg')):
                        stem = os.path.splitext(entry)[0]
                        student = stem.split('__', 1)[0].strip()
                        yield student, full_path, entry
        
        for name, filepath, display_name in _iter_reference_images():
            if not name:
                continue

            # Try cache first (skip detection/resnet if unchanged)
            try:
                mtime = os.path.getmtime(filepath)
            except Exception:
                mtime = None

            cached = cache_files.get(filepath)
            if (
                isinstance(cached, dict)
                and cached.get("mtime") == mtime
                and cached.get("name") == name
                and isinstance(cached.get("emb"), torch.Tensor)
                and cached["emb"].ndim == 1
            ):
                emb_vec = cached["emb"].float().cpu()
                used_cache_files.add(filepath)

                self.embeddings_db.setdefault(name, []).append(emb_vec)
                flat_embeddings.append(emb_vec)
                flat_names.append(name)

                if name not in loaded_counts:
                    self.names_db.append(name)
                    loaded_counts[name] = 0
                loaded_counts[name] += 1
                continue

            try:
                img = Image.open(filepath)
                    
                # Get detected faces (returns list if keep_all=True)
                # Convert to RGB to ensure 3 channels (fix for RGBA/Grayscale issues)
                img = img.convert('RGB')
                boxes, _ = self.mtcnn.detect(img)
                pass_2_attempt = False # Flag to track if we used lenient detection

                # Fallback mechanism: If strict detection fails, try lenient detection
                if boxes is None or len(boxes) == 0:
                    # Keep logs clean: don't spam per-file warnings during normal startup.
                    # Create a temporary lenient MTCNN with very low thresholds
                    mtcnn_lenient = MTCNN(
                        keep_all=True,
                        min_face_size=20,
                        thresholds=[0.4, 0.5, 0.5], # Ultra-low thresholds
                        factor=0.709,
                        post_process=True,
                        device=self.device
                    )
                    boxes, _ = mtcnn_lenient.detect(img)
                    pass_2_attempt = True
                    del mtcnn_lenient

                if boxes is not None and len(boxes) > 0:
                    # Find largest face
                    largest_idx = 0
                    max_area = 0
                    for i, box in enumerate(boxes):
                        area = (box[2] - box[0]) * (box[3] - box[1])
                        if area > max_area:
                            max_area = area
                            largest_idx = i

                    # Manual crop + normalize for robust embeddings
                    box = boxes[largest_idx]
                    x1, y1, x2, y2 = [int(b) for b in box]

                    # Ensure coordinates are within image bounds
                    w, h = img.size
                    x1 = max(0, x1); y1 = max(0, y1)
                    x2 = min(w, x2); y2 = min(h, y2)

                    face_img = img.crop((x1, y1, x2, y2))

                    # Resize & Normalize for FaceNet
                    face_img = face_img.resize((160, 160))
                    face_arr = np.array(face_img).astype(np.float32)
                    face_arr = (face_arr - 127.5) / 128.0
                    face_tensor = torch.tensor(face_arr).permute(2, 0, 1).unsqueeze(0)

                    # Generate embedding
                    embedding = self.resnet(face_tensor.to(self.device)).detach()
                    emb_vec = embedding.squeeze(0).float().cpu()

                    # Update cache
                    cache_files[filepath] = {"mtime": mtime, "name": name, "emb": emb_vec}
                    used_cache_files.add(filepath)

                    self.embeddings_db.setdefault(name, []).append(emb_vec)
                    flat_embeddings.append(emb_vec)
                    flat_names.append(name)

                    if name not in loaded_counts:
                        self.names_db.append(name)
                        loaded_counts[name] = 0
                    loaded_counts[name] += 1

                else:
                    # Skip quietly to keep startup clean/fast
                    pass

            except Exception as e:
                # Keep startup clean; you can add a debug flag later if needed.
                pass
                    
        total_refs = sum(loaded_counts.values()) if loaded_counts else 0
        # Print a clean per-student summary (not per angle image)
        if migrated:
            print(f"  ↪ Migrated {migrated} legacy images into per-student folders")
        if loaded_counts:
            for student in sorted(loaded_counts.keys(), key=lambda s: s.lower()):
                print(f"  ✓ {student}: {loaded_counts[student]} refs")
        print(f"Database loaded: {len(self.names_db)} students, {total_refs} reference images\n")

        # Remove stale cache entries + persist cache
        try:
            # Drop entries for files that no longer exist
            for fp in list(cache_files.keys()):
                if not os.path.exists(fp):
                    cache_files.pop(fp, None)
            cache["files"] = cache_files
            torch.save(cache, cache_path)
        except Exception:
            pass

        # Build fast search index
        if flat_embeddings:
            self._db_matrix = torch.stack(flat_embeddings).to(self.device)
            self._db_names_flat = flat_names
        else:
            self._db_matrix = None
            self._db_names_flat = []
    
    def register_student(self, name, image_path):
        """Register a new student with a fast check for strictly one face."""
        print(f"\nAttempting to register: {name}")
        
        if not os.path.exists(image_path):
            print(f"❌ Error: Image file not found at {image_path}")
            return False

        try:
            img = Image.open(image_path)
            boxes, _ = self.mtcnn.detect(img)
            
            if boxes is None:
                print("❌ Registration Failed: No face detected.")
                return False
            
            # Select largest face
            selected_box = None
            max_area = 0
            
            if len(boxes) > 1:
                print(f"⚠️  Warning: Multiple faces ({len(boxes)}) detected. Selecting the largest one.")
                for box in boxes:
                    area = (box[2] - box[0]) * (box[3] - box[1])
                    if area > max_area:
                        max_area = area
                        selected_box = box
            else:
                selected_box = boxes[0]

            # Save the image to the student's folder (front reference)
            student_dir = os.path.join(self.database_path, name)
            os.makedirs(student_dir, exist_ok=True)
            save_path = os.path.join(student_dir, "front.jpg")
            img.save(save_path)
            print(f"✅ Success! {name} registered. Image saved to {save_path}")
            
            # Reload DB to include new student
            self.load_database()
            return True
            
        except Exception as e:
            print(f"❌ Error during registration: {e}")
            return False

    def register_from_camera(self, name, camera_index=0):
        """
        Capture a student's photo directly from the webcam to ensure high quality (and matching environment).
        """
        print(f"\n📷 Starting Camera Registration for: {name}")
        print(" Auto capture mode (no manual align box).")
        print(" Please follow on-screen instructions to capture 4 angles: LEFT, RIGHT, UP, DOWN.")
        print(" [Q] to Cancel")
        
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print("❌ Error: Could not open camera.")
            return False

        # Student folder (one folder per student)
        student_dir = os.path.join(self.database_path, name)
        os.makedirs(student_dir, exist_ok=True)

        # --- Camera performance tuning (best-effort; depends on camera/driver) ---
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        try:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        except Exception:
            pass
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 60)

        # 4 requested angles (guided)
        targets = [
            ("left",  "Turn your face LEFT"),
            ("right", "Turn your face RIGHT"),
            ("up",    "Move your face UP"),
            ("down",  "Move your face DOWN"),
        ]
        captured = {}
        current_idx = 0

        # Stability gating for auto-capture
        stable_needed = 10  # frames
        stable_count = 0
        last_pose = None

        def _largest_face_index(boxes_np):
            if boxes_np is None or len(boxes_np) == 0:
                return None
            best_i = 0
            max_area_local = 0.0
            for i, b in enumerate(boxes_np):
                area = float((b[2] - b[0]) * (b[3] - b[1]))
                if area > max_area_local:
                    max_area_local = area
                    best_i = i
            return best_i

        def _estimate_pose(landmarks5):
            """
            Rough head-pose estimation from 5 landmarks: left_eye, right_eye, nose, mouth_left, mouth_right.
            Returns: 'left', 'right', 'up', 'down', or 'center'.
            """
            (lx, ly), (rx, ry), (nx, ny), (mlx, mly), (mrx, mry) = landmarks5
            eye_mid_x = (lx + rx) / 2.0
            eye_mid_y = (ly + ry) / 2.0
            mouth_mid_y = (mly + mry) / 2.0

            eye_dist = max(1.0, abs(rx - lx))
            yaw = (nx - eye_mid_x) / eye_dist  # negative => left, positive => right
            denom = max(1.0, (mouth_mid_y - eye_mid_y))
            pitch_ratio = (ny - eye_mid_y) / denom  # smaller => up, larger => down

            if yaw < -0.18:
                return "left"
            if yaw > 0.18:
                return "right"
            if pitch_ratio < 0.42:
                return "up"
            if pitch_ratio > 0.62:
                return "down"
            return "center"

        def _safe_crop_and_save(bgr_frame, box, save_path):
            h0, w0, _ = bgr_frame.shape
            x1, y1, x2, y2 = [int(v) for v in box]
            bw = x2 - x1
            bh = y2 - y1
            pad_x = int(bw * 0.25)
            pad_y = int(bh * 0.35)
            x1 = max(0, x1 - pad_x); y1 = max(0, y1 - pad_y)
            x2 = min(w0, x2 + pad_x); y2 = min(h0, y2 + pad_y)
            crop = bgr_frame[y1:y2, x1:x2]
            if crop is None or crop.size == 0:
                cv2.imwrite(save_path, bgr_frame)
            else:
                cv2.imwrite(save_path, crop)
            
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to grab frame")
                break

            # Detect face + landmarks (largest face only for registration)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_frame)
            boxes, probs, landmarks = self.mtcnn.detect(pil_img, landmarks=True)

            # UI overlay: instruction text only (no guide box)
            ui = frame.copy()
            h, w, _ = ui.shape
            cv2.rectangle(ui, (10, 10), (w - 10, 95), (0, 0, 0), -1)

            # Safety: if completed, exit
            if current_idx >= len(targets):
                break

            target_key, target_text = targets[current_idx]
            cv2.putText(ui, f"Register: {name}   ({len(captured)}/{len(targets)})", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
            cv2.putText(ui, f"{target_text}", (20, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

            if boxes is None or len(boxes) == 0 or landmarks is None or len(landmarks) == 0:
                stable_count = 0
                last_pose = None
                cv2.putText(ui, "No face detected. Look at the camera.", (20, 135),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
            else:
                i = _largest_face_index(boxes)
                box = boxes[i]
                lm = landmarks[i]
                area = float((box[2] - box[0]) * (box[3] - box[1]))

                if area < 12000:
                    stable_count = 0
                    last_pose = None
                    cv2.putText(ui, "Move closer for better capture.", (20, 135),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
                else:
                    pose = _estimate_pose(lm)
                    pose_ok = (pose == target_key)

                    if pose_ok:
                        if last_pose == pose:
                            stable_count += 1
                        else:
                            stable_count = 1
                            last_pose = pose
                        cv2.putText(ui, "Hold still...", (20, 135),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
                    else:
                        stable_count = 0
                        last_pose = pose
                        cv2.putText(ui, f"Detected: {pose.upper()} (adjust)", (20, 135),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

                    cv2.putText(ui, f"Stability: {min(stable_count, stable_needed)}/{stable_needed}", (20, 170),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                    # Auto-capture when stable pose matches instruction
                    if pose_ok and stable_count >= stable_needed and target_key not in captured:
                        save_path = os.path.join(student_dir, f"{target_key}.jpg")
                        _safe_crop_and_save(frame, box, save_path)
                        captured[target_key] = save_path
                        print(f"✅ Captured {target_key.upper()} -> {save_path}")
                        current_idx += 1
                        stable_count = 0
                        last_pose = None

            # Force Window to Top
            cv2.namedWindow("Register Student", cv2.WND_PROP_TOPMOST)
            cv2.setWindowProperty("Register Student", cv2.WND_PROP_TOPMOST, 1)
            cv2.imshow("Register Student", ui)

            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                print("❌ Registration cancelled.")
                break

            if len(captured) >= len(targets):
                print("✅ Registration complete! Updating face database...")
                cap.release()
                cv2.destroyAllWindows()
                self.load_database()
                return True
        
        cap.release()
        cv2.destroyAllWindows()
        return False

    def _find_best_match(self, embedding):
        """
        Fast nearest-neighbor match against all stored reference embeddings.
        Returns (best_name, min_dist). If DB is empty => ("Unknown", inf).
        """
        if self._db_matrix is None or not self._db_names_flat:
            return "Unknown", float('inf')

        if isinstance(embedding, torch.Tensor):
            q = embedding
        else:
            q = torch.tensor(embedding)

        if q.ndim == 2:
            q = q.squeeze(0)
        q = q.to(self.device).float()

        # Vectorized L2 distances: [N]
        dists = torch.norm(self._db_matrix - q, dim=1)
        min_val, idx = torch.min(dists, dim=0)
        return self._db_names_flat[int(idx.item())], float(min_val.item())

    async def mark_attendance(self, image_path):
        """Mark attendance using largest face detected."""
        # Capture the exact timestamp immediately when attendance is being marked
        # Define IST timezone manually to avoid ZoneInfo issues on Windows
        IST = timezone(timedelta(hours=5, minutes=30))
        attendance_timestamp = datetime.now(IST)
        
        print(f"\nProcessing attendance for: {image_path}")
        
        if not os.path.exists(image_path):
            print(f"❌ Error: Image file not found at {image_path}")
            return

        try:
            img = Image.open(image_path)
            img_draw = img.copy()
            draw = ImageDraw.Draw(img_draw)
            
            # Detect faces
            boxes, probs = self.mtcnn.detect(img)
            
            if boxes is None:
                print("❌ Attendance Failed: No face detected.")
                return

            # Select largest face
            selected_idx = 0
            max_area = 0
            
            if len(boxes) > 1:
                print(f"⚠️  Multiple faces ({len(boxes)}) detected. Selecting the largest one for attendance.")
                for i, box in enumerate(boxes):
                    area = (box[2] - box[0]) * (box[3] - box[1])
                    if area > max_area:
                        max_area = area
                        selected_idx = i
            else:
                selected_idx = 0

            # Process the single face
            print("✅ Face detected. Verifying identity...")
            box = boxes[selected_idx]
            
            # Get embedding
            faces = self.mtcnn(img)
            if isinstance(faces, list):
                if len(faces) == 0:
                    return
                # Must pick the same index!
                # MTCNN(keep_all=True) returns faces in same order as boxes usually, 
                # strictly speaking we should crop using the box to be 100% sure, 
                # but facenet_pytorch typically aligns. 
                face_tensor = faces[selected_idx]
            else:
                face_tensor = faces # Should be list if keep_all=True
                
            # Ensure 4D tensor
            if face_tensor.ndim == 3:
                face_tensor = face_tensor.unsqueeze(0)
                
            embedding = self.resnet(face_tensor.to(self.device)).detach()

            # Compare with database (fast, vectorized)
            best_match_name, min_dist = self._find_best_match(embedding)

            print(f"  -> Match result: {best_match_name} (Distance: {min_dist:.4f})")
            
            if min_dist < self.threshold:
                await self.log_attendance(best_match_name, attendance_timestamp)
                color = "green"
                label = f"{best_match_name}"
            else:
                color = "orange"
                label = "Unknown"
                print("  ⚠️  Identity not recognized.")

            # visual - Using OpenCV for reliable drawing
            # Convert PIL to BGR for OpenCV
            img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            
            # Coordinates must be integers
            x1, y1, x2, y2 = [int(b) for b in box]
            
            # Determine color (BGR)
            # green: (0, 255, 0), orange: (0, 165, 255) -> BGR (0, 165, 255) is orange-ish
            color_bgr = (0, 255, 0) if color == "green" else (0, 165, 255)
            
            # Draw Box
            cv2.rectangle(img_cv, (x1, y1), (x2, y2), color_bgr, 2)
            
            # Draw Text
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            thickness = 2
            text_size, _ = cv2.getTextSize(label, font, font_scale, thickness)
            
            # Text background
            cv2.rectangle(img_cv, (x1, y1 - text_size[1] - 10), (x1 + text_size[0] + 10, y1), color_bgr, -1)
            
            # Text
            cv2.putText(img_cv, label, (x1 + 5, y1 - 5), font, font_scale, (255, 255, 255), thickness)

            # Save result
            output_filename = f"attendance_{os.path.basename(image_path)}"
            save_path = os.path.join('output', output_filename)
            cv2.imwrite(save_path, img_cv)
            print(f"✅ Processed image saved to {save_path}")
            
            # Popup display
            print("📷 Displaying result... Press any key to close the window.")
            cv2.namedWindow("Attendance Result", cv2.WINDOW_NORMAL)
            cv2.imshow("Attendance Result", img_cv)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        except Exception as e:
            print(f"❌ Error processing attendance: {e}")

    async def run_live_mode(self, camera_index=0):
        """
        Run the attendance system in live mode.
        - Scans until a face is matched.
        - Requires strict confirmation (multiple consecutive frames) for accuracy.
        - Marks attendance and closes automatically on success.
        """
        print(f"\n🎥 Starting Live Attendance Mode (Camera Index: {camera_index})...")
        print("Scaning for faces... (Press 'q' to quit manually)")

        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print(f"❌ Error: Could not open camera with index {camera_index}.")
            return

        # Optimization & Accuracy params
        process_every_n_frames = 4  # Process detection every 4 frames
        frame_count = 0
        
        # Stability tracking: We need a face to be recognized for 'integrity_threshold' consecutive checks
        # to ensure it's not a glitch.
        current_match_name = None
        consecutive_match_count = 0
        integrity_threshold = 3  # Require 3 consecutive confirmations
        
        # UI State
        face_locations = []
        face_names = []
        system_message = "Scanning..."
        success_trigger = False
        success_timer_start = None

        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Error: Failed to grab frame.")
                break

            # Process generation
            if frame_count % process_every_n_frames == 0 and not success_trigger:
                
                # Pre-processing
                rgb_small_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb_small_frame)
                
                # Detect
                boxes, _ = self.mtcnn.detect(pil_img)
                
                face_locations = []
                face_names = []
                
                # Reset stability if no faces found
                if boxes is None:
                    consecutive_match_count = 0
                    current_match_name = None
                else:
                    # We only care about the largest face for "Gate Access" style
                    # Find largest box
                    largest_box = None
                    max_area = 0
                    for box in boxes:
                        area = (box[2] - box[0]) * (box[3] - box[1])
                        if area > 1000 and area > max_area: # Ignore tiny faces
                            max_area = area
                            largest_box = box
                    
                    if largest_box is not None:
                        # Process only the largest/main face
                        face_locations = [largest_box]
                        
                        x1, y1, x2, y2 = [int(b) for b in largest_box]
                        h, w, _ = frame.shape
                        x1 = max(0, x1); y1 = max(0, y1); x2 = min(w, x2); y2 = min(h, y2)
                        
                        face_img = pil_img.crop((x1, y1, x2, y2))
                        
                        # Resize & Normalize
                        face_img = face_img.resize((160, 160))
                        face_arr = np.array(face_img).astype(np.float32)
                        face_arr = (face_arr - 127.5) / 128.0
                        face_tensor = torch.tensor(face_arr).permute(2, 0, 1).unsqueeze(0)
                        
                        embedding = self.resnet(face_tensor.to(self.device)).detach()

                        # Compare (fast, vectorized)
                        best_match_name, min_dist = self._find_best_match(embedding)
                        
                        # Strict threshold for "High Accuracy" request
                        # Default was 0.6, let's use self.threshold (which is 0.6)
                        if min_dist > self.threshold:
                            best_match_name = "Unknown"
                        
                        face_names.append(best_match_name)
                        
                        # Stability Logic
                        if best_match_name != "Unknown":
                            if best_match_name == current_match_name:
                                consecutive_match_count += 1
                            else:
                                consecutive_match_count = 1
                                current_match_name = best_match_name
                                
                            system_message = f"Verifying {current_match_name}... ({consecutive_match_count}/{integrity_threshold})"
                            
                            if consecutive_match_count >= integrity_threshold:
                                # SUCCESS!
                                IST = timezone(timedelta(hours=5, minutes=30))
                                attendance_timestamp = datetime.now(IST)
                                await self.log_attendance(current_match_name, attendance_timestamp)
                                
                                system_message = f"Done! Attendance Marked: {current_match_name}"
                                success_trigger = True
                                success_timer_start = datetime.now()
                        else:
                            consecutive_match_count = 0
                            current_match_name = None
                            system_message = "Unknown Face"
                            
            frame_count += 1

            # --- Drawing UI ---
            for (box), name in zip(face_locations, face_names):
                x1, y1, x2, y2 = [int(b) for b in box]
                
                color = (0, 0, 255) # Default Red
                
                if success_trigger:
                    color = (0, 255, 0) # Green on success
                    name = current_match_name # Enforce the matched name
                elif name == "Unknown":
                    color = (0, 0, 255)
                elif consecutive_match_count > 0:
                    # Creating a transition verification color (Yellow/Orange)
                    color = (0, 255, 255)
                    
                # Draw Box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw Name Label
                cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), color, cv2.FILLED)
                cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

            # Overlay Status
            cv2.rectangle(frame, (10, 10), (frame.shape[1] - 10, 50), (0, 0, 0), -1)
            cv2.putText(frame, system_message, (frame.shape[1]//2 - 100, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Force Window to Top
            cv2.namedWindow('Live Marking', cv2.WND_PROP_TOPMOST)
            cv2.setWindowProperty('Live Marking', cv2.WND_PROP_TOPMOST, 1)
            cv2.imshow('Live Marking', frame)

            # Auto-Close Logic
            if success_trigger:
                if (datetime.now() - success_timer_start).total_seconds() > 1.0: # Faster close (1s)
                    print(f"✅ Auto-closing after success for {current_match_name}")
                    break

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    async def log_attendance(self, name, timestamp):
        """Log the attendance to a CSV file. If out of range, log as 'OUT OF RANGE'."""
        allowed, msg = await check_location_allowed()
        filename = "attendance_log.csv"
        date_str = timestamp.strftime("%Y-%m-%d")
        
        if not allowed:
            print(f"❌ Attendance not recorded: {msg}")
            time_str = "OUT OF RANGE"
        else:
            time_str = timestamp.strftime("%H:%M:%S")
            print(f"📝 Attendance recorded for {name} at {time_str} (Local Time)")

        file_exists = os.path.exists(filename)
        with open(filename, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Name", "Date", "Time"])
            writer.writerow([name, date_str, time_str])


# NOTE: admin_set_location() removed from menu - location is now set in code above.
# This function is kept for future admin interface implementation.
# def admin_set_location():
#     """Let admin set attendance location: center (lat, lon) and radius in meters. Enable/disable check."""
#     ...


async def async_main():
    # Load the full database root so new registrations create their own folder:
    #   sample_faces/<StudentName>/{left,right,up,down}.jpg
    system = AttendanceSystem(database_path="sample_faces")

    while True:
        print("\n=== Smart Attendance System ===")
        print("1. Register Student")
        print("2. Mark Attendance")
        print("3. Exit")
        choice = input("Enter choice: ").strip()

        if choice == '1':
            name = input("Enter Name: ")
            print("  a. Enter Image Path")
            print("  b. Capture from Camera")
            reg_choice = input("  Choose method (a/b): ").lower()
            if reg_choice == 'a':
                img_path = input("  Enter Image Path: ").strip('"')
                system.register_student(name, img_path)
            elif reg_choice == 'b':
                cam_idx_str = input("  Enter Camera Index (default 0): ")
                cam_idx = int(cam_idx_str) if cam_idx_str.strip() else 0
                system.register_from_camera(name, cam_idx)
            else:
                print("❌ Invalid choice.")

        elif choice == '2':
            cam_idx_str = input("Enter Camera Index (default 0): ")
            cam_idx = int(cam_idx_str) if cam_idx_str.strip() else 0
            await system.run_live_mode(cam_idx)

        elif choice == '3':
            break

def main():
    asyncio.run(async_main())

if __name__ == "__main__":
    main()
