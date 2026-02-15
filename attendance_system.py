import torch
import cv2
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image, ImageDraw, ImageFont
import os
import csv
from datetime import datetime, timezone, timedelta

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
        self.embeddings_db = {}
        self.names_db = []
        
        if not os.path.exists('output'):
            os.makedirs('output')
            
        self.load_database()

    def load_database(self):
        """Load reference faces from sample_faces folder"""
        if not os.path.exists(self.database_path):
            os.makedirs(self.database_path)
            print(f"Created database folder: {self.database_path}")
            return

        print(f"\nLoading face database from {self.database_path}...")
        self.embeddings_db = {}
        self.names_db = []
        
        for filename in os.listdir(self.database_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                name = os.path.splitext(filename)[0]
                filepath = os.path.join(self.database_path, filename)
                
                try:
                    img = Image.open(filepath)
                    
                    # Get detected faces (returns list if keep_all=True)
                    # Convert to RGB to ensure 3 channels (fix for RGBA/Grayscale issues)
                    img = img.convert('RGB')
                    boxes, _ = self.mtcnn.detect(img)
                    pass_2_attempt = False # Flag to track if we used lenient detection

                    # Fallback mechanism: If strict detection fails, try lenient detection
                    if boxes is None or len(boxes) == 0:
                        print(f"  ⚠️  Strict detection failed for {filename}. Retrying with SUPER lenient settings...")
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
                        
                        # Extract that specific face tensor
                        # We need to manually crop and process because self.mtcnn might have failed
                        # and we have boxes from either self.mtcnn or mtcnn_lenient
                        
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
                        embedding = self.resnet(face_tensor.to(self.device)).detach().cpu()
                        self.embeddings_db[name] = embedding
                        self.names_db.append(name)
                        
                        if pass_2_attempt:
                            print(f"  ✓ Loaded: {name} (via lenient fallback)")
                        else:
                            print(f"  ✓ Loaded: {name}")
                    else:
                         print(f"  ⚠️  Skipping {filename}: No face detected even with lenient settings.")
                        
                except Exception as e:
                    print(f"  ❌ Error loading {filename}: {e}")
                    
        print(f"Database loaded: {len(self.names_db)} faces registered\n")
    
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

            # Save the image to the database folder
            save_path = os.path.join(self.database_path, f"{name}.jpg")
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
        print(" Instructions: Position your face clearly in the box.")
        print(" [SPACE] to Capture")
        print(" [Q] to Cancel")
        
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print("❌ Error: Could not open camera.")
            return False
            
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to grab frame")
                break
                
            # Draw a guide box in the center
            h, w, c = frame.shape
            # Box size increased to 320x320 (offset 160)
            cv2.rectangle(frame, (w//2 - 160, h//2 - 160), (w//2 + 160, h//2 + 160), (255, 255, 0), 2)
            cv2.putText(frame, "Align Face Here", (w//2 - 150, h//2 - 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # Force Window to Top
            cv2.namedWindow("Register Student", cv2.WND_PROP_TOPMOST)
            cv2.setWindowProperty("Register Student", cv2.WND_PROP_TOPMOST, 1)
            cv2.imshow("Register Student", frame)
            
            key = cv2.waitKey(1)
            if key & 0xFF == ord(' '): # Spacebar
                # Check for face detection BEFORE saving to ensure quality
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb_frame)
                boxes, _ = self.mtcnn.detect(pil_img)
                
                if boxes is not None and len(boxes) > 0:
                    # Save the image
                    save_path = os.path.join(self.database_path, f"{name}.jpg")
                    cv2.imwrite(save_path, frame)
                    print(f"✅ Captured! Image saved to {save_path}")
                    cap.release()
                    cv2.destroyAllWindows()
                    self.load_database() # Reload with new face
                    return True
                else:
                    print("⚠️  No face detected in capture. Please try again.")
            
            elif key & 0xFF == ord('q'):
                print("❌ Registration cancelled.")
                break
        
        cap.release()
        cv2.destroyAllWindows()
        return False

    def mark_attendance(self, image_path):
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
                
            embedding = self.resnet(face_tensor.to(self.device)).detach().cpu()
            
            # Compare with database
            best_match_name = "Unknown"
            min_dist = float('inf')
            
            # Simple Euclidean distance search
            for name, db_embedding in self.embeddings_db.items():
                dist = (embedding - db_embedding).norm().item()
                if dist < min_dist:
                    min_dist = dist
                    best_match_name = name

            print(f"  -> Match result: {best_match_name} (Distance: {min_dist:.4f})")
            
            if min_dist < self.threshold:
                self.log_attendance(best_match_name, attendance_timestamp)
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

    def run_live_mode(self, camera_index=0):
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
                        
                        embedding = self.resnet(face_tensor.to(self.device)).detach().cpu()

                        # Compare
                        best_match_name = "Unknown"
                        min_dist = float('inf')

                        for name, db_embedding in self.embeddings_db.items():
                            dist = (embedding - db_embedding).norm().item()
                            if dist < min_dist:
                                min_dist = dist
                                best_match_name = name
                        
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
                                self.log_attendance(current_match_name, attendance_timestamp)
                                
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

    def log_attendance(self, name, timestamp):
        """Log the attendance to a CSV file using the exact timestamp provided."""
        filename = "attendance_log.csv"
        date_str = timestamp.strftime("%Y-%m-%d")
        time_str = timestamp.strftime("%H:%M:%S")
        
        file_exists = os.path.exists(filename)
        

        
        with open(filename, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Name", "Date", "Time"])
            
            writer.writerow([name, date_str, time_str])
            print(f"📝 Attendance recorded for {name} at {time_str} (Local Time)")

def main():
    system = AttendanceSystem()
    
    # Simple CLI loop
    while True:
        print("\n=== Smart Attendance System ===")
        print("1. Register Student")
        print("2. Mark Attendance")
        print("3. Exit")
        choice = input("Enter choice: ")
        
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
            system.run_live_mode(cam_idx)
            
        elif choice == '3':
            break

if __name__ == "__main__":
    main()
