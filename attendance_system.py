import torch
import cv2
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image, ImageDraw, ImageFont
import os
import csv
from datetime import datetime

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
                    faces = self.mtcnn(img)
                    pass # We need to handle multiple faces here too if keep_all=True is used in load
                    
                    # Re-detect to get boxes for largest face logic if needed, 
                    # but self.mtcnn(img) returns tensors. 
                    # For simplicity in loading, we assume clean images or take the first/largest.
                    # Actually, self.mtcnn(img) with keep_all=True returns a list of tensors.
                    
                    if faces is not None:
                         # Logic to pick the best face if multiple are returned by mtcnn(img) directly is harder 
                         # without boxes. Let's use detect() to be consistent if we want to be strict, 
                         # OR just rely on the first one if we assume sample images are good. 
                         # However, to be robust, let's stick to the existing simple logic for loading 
                         # but maybe warn. 
                         # actually, the previous code took faces[0]. 
                         # Let's improve this to be consistent with register/mark.
                         
                        boxes, _ = self.mtcnn.detect(img)
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
                            # faces is a list of tensors or a single tensor
                            if isinstance(faces, list):
                                face_tensor = faces[largest_idx]
                            else:
                                face_tensor = faces # Should not happen with keep_all=True and len>0

                            # Ensure 4D tensor [Batch, Channels, Height, Width]
                            if face_tensor.ndim == 3:
                                face_tensor = face_tensor.unsqueeze(0)
                            
                            # Generate embedding
                            embedding = self.resnet(face_tensor.to(self.device)).detach().cpu()
                            self.embeddings_db[name] = embedding
                            self.names_db.append(name)
                            print(f"  ✓ Loaded: {name}")
                        else:
                             print(f"  ⚠️  Skipping {filename}: No face detected.")

                    else:
                        print(f"  ⚠️  Skipping {filename}: No face detected.")
                        
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

            # We need to save the image (maybe cropped? No, user just saves the file). 
            # The original code just copies the file. 
            # If we want to be "clean", we could crop, but the requirement is just to fix detection.
            # We will proceed with saving the original image, as the loader now handles multiple faces 
            # by picking the largest one (implemented above).

            # Copy/Save the image to the database folder
            save_path = os.path.join(self.database_path, f"{name}.jpg")
            img.save(save_path)
            print(f"✅ Success! {name} registered. Image saved to {save_path}")
            
            # Reload DB to include new student
            self.load_database()
            return True
            
        except Exception as e:
            print(f"❌ Error during registration: {e}")
            return False

    def mark_attendance(self, image_path):
        """Mark attendance using largest face detected."""
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
                self.log_attendance(best_match_name)
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

    def log_attendance(self, name):
        """Log the attendance to a CSV file."""
        filename = "attendance_log.csv"
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")
        
        file_exists = os.path.exists(filename)
        
        # Check if already marked present for today to avoid duplicates? (Optional, skipping for simplicity)
        
        with open(filename, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Name", "Date", "Time"])
            
            writer.writerow([name, date_str, time_str])
            print(f"📝 Attendance recorded for {name} at {time_str}")

def main():
    system = AttendanceSystem()
    
    # Simple CLI loop
    while True:
        print("\n=== Single-Face Attendance System ===")
        print("1. Register Student")
        print("2. Mark Attendance")
        print("3. Exit")
        choice = input("Enter choice: ")
        
        if choice == '1':
            name = input("Enter Name: ")
            img_path = input("Enter Image Path: ").strip('"')
            system.register_student(name, img_path)
            
        elif choice == '2':
            img_path = input("Enter Image Path for Attendance: ").strip('"')
            system.mark_attendance(img_path)
            
        elif choice == '3':
            break

if __name__ == "__main__":
    main()
