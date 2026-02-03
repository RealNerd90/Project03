"""
Face Detection and Recognition Demo using MTCNN and FaceNet
This demo detects faces using MTCNN and recognizes them using FaceNet embeddings
"""

import os
import cv2
import numpy as np
import time
from PIL import Image
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path

class FaceRecognitionDemo:
    def __init__(self, database_path='sample_faces', threshold=0.6):
        """
        Initialize the face recognition system
        
        Args:
            database_path: Path to folder containing reference face images
            threshold: Distance threshold for face matching (lower = stricter)
        """
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize MTCNN for face detection
        # HIGH ACCURACY MODE: Tuned to reject non-face objects (leaves, textures)
        self.mtcnn = MTCNN(
            image_size=160, 
            margin=20,
            min_face_size=40,
            thresholds=[0.8, 0.85, 0.9], # Very strict thresholds
            factor=0.709,
            post_process=True,
            device=self.device,
            keep_all=True
        )
        
        # Initialize FaceNet for face recognition
        self.facenet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        
        self.database_path = database_path
        self.threshold = threshold
        self.face_database = {}
        
        # Load face database
        self.load_face_database()
    
    def load_face_database(self):
        """Load and encode all faces from the database folder"""
        if not os.path.exists(self.database_path):
            os.makedirs(self.database_path)
            print(f"Created database folder: {self.database_path}")
            print("Please add reference images to this folder (filename = person's name)")
            return
        
        print(f"\nLoading face database from {self.database_path}...")
        supported_formats = ['.jpg', '.jpeg', '.png', '.bmp']
        
        for filename in os.listdir(self.database_path):
            file_path = os.path.join(self.database_path, filename)
            file_ext = os.path.splitext(filename)[1].lower()
            
            if file_ext in supported_formats:
                try:
                    # Load image
                    img = Image.open(file_path).convert('RGB')
                    
                    # Detect face
                    face_tensor = self.mtcnn(img)
                    
                    if face_tensor is not None:
                        # Get the first face if multiple detected
                        if len(face_tensor.shape) == 4:
                            face_tensor = face_tensor[0]
                        
                        # Generate embedding
                        with torch.no_grad():
                            embedding = self.facenet(face_tensor.unsqueeze(0).to(self.device))
                        
                        # Store in database (use filename without extension as name)
                        name = os.path.splitext(filename)[0]
                        self.face_database[name] = embedding.cpu().numpy()
                        print(f"  ✓ Loaded: {name}")
                    else:
                        print(f"  ✗ No face detected in: {filename}")
                        
                except Exception as e:
                    print(f"  ✗ Error loading {filename}: {str(e)}")
        
        print(f"\nDatabase loaded: {len(self.face_database)} faces registered\n")
    
    def recognize_face(self, face_embedding):
        """
        Compare face embedding against database
        
        Args:
            face_embedding: Embedding vector of detected face
            
        Returns:
            (name, distance) or (None, None) if no match
        """
        if len(self.face_database) == 0:
            return None, None
        
        min_distance = float('inf')
        recognized_name = None
        
        # Compare with all database faces
        for name, db_embedding in self.face_database.items():
            # Calculate Euclidean distance
            distance = np.linalg.norm(face_embedding - db_embedding)
            
            if distance < min_distance:
                min_distance = distance
                recognized_name = name
        
        # Check if distance is below threshold
        if min_distance < self.threshold:
            return recognized_name, min_distance
        else:
            return "Unknown", min_distance
    
    def process_image(self, image_path, output_path=None, show_result=True):
        """
        Detect and recognize faces in an image
        
        Args:
            image_path: Path to input image
            output_path: Path to save output image (optional)
            show_result: Whether to display the result
            
        Returns:
            Dictionary with detection results and metrics
        """
        print(f"\n{'='*60}")
        print(f"Processing: {os.path.basename(image_path)}")
        print(f"{'='*60}")
        
        # Start timing
        start_time = time.time()
        
        # Load image
        img = Image.open(image_path).convert('RGB')
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
        # Detect faces
        detection_start = time.time()
        boxes, probs, landmarks = self.mtcnn.detect(img, landmarks=True)
        detection_time = time.time() - detection_start
        
        results = {
            'image_path': image_path,
            'num_faces': 0,
            'faces': [],
            'detection_time': detection_time,
            'recognition_time': 0,
            'total_time': 0
        }
        
        # Filter weak detections (Strict high-accuracy mode)
        final_boxes = []
        final_probs = []
        final_landmarks = []
        
        if boxes is not None:
            for i, prob in enumerate(probs):
                # FILTER: Only keep detections with > 98% confidence
                # Extremely strict filter to avoid false positives in complex backgrounds
                if prob > 0.98:
                    final_boxes.append(boxes[i])
                    final_probs.append(prob)
                    if landmarks is not None:
                        final_landmarks.append(landmarks[i])
        
        if final_boxes:
            results['num_faces'] = len(final_boxes)
            recognition_start = time.time()
            
            # Extract faces manually for the filtered boxes
            faces = []
            for box in final_boxes:
                # Ensure coordinates are within image bounds
                box = [int(max(0, b)) for b in box]
                # Crop face
                face = img.crop((box[0], box[1], box[2], box[3]))
                # Resize to 160x160 (FaceNet requirement)
                face = face.resize((160, 160), Image.BILINEAR)
                # Convert to tensor and normalize
                face = np.array(face).astype(np.float32)
                face = torch.tensor(face).permute(2, 0, 1)
                face = (face - 127.5) / 128.0
                faces.append(face)
            
            if faces:
                # Stack into batch
                face_tensors = torch.stack(faces)
                
                # Generate embeddings
                with torch.no_grad():
                    embeddings = self.facenet(face_tensors.to(self.device)).cpu().numpy()
                
                # Recognize each face
                for i, (box, prob, embedding) in enumerate(zip(final_boxes, final_probs, embeddings)):
                    name, distance = self.recognize_face(embedding)
                    
                    if name is None:
                        name = "Unknown"
                        distance = 0.0
                    
                    # Draw bounding box
                    x1, y1, x2, y2 = [int(coord) for coord in box]
                    color = (0, 255, 0) if name != "Unknown" else (0, 165, 255)
                    cv2.rectangle(img_cv, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw label
                    label = f"{name}" if name != "Unknown" else "Unknown"
                    debug_label = f"{label} ({prob*100:.1f}%)"
                    
                    label_y = y1 - 10 if y1 - 10 > 10 else y1 + 20
                    
                    # Background for text
                    (text_width, text_height), _ = cv2.getTextSize(debug_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(img_cv, (x1, label_y - text_height - 5), (x1 + text_width, label_y + 5), color, -1)
                    cv2.putText(img_cv, debug_label, (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    # Draw landmarks if available
                    if final_landmarks and i < len(final_landmarks):
                        for landmark in final_landmarks[i]:
                            cv2.circle(img_cv, tuple(landmark.astype(int)), 2, (255, 0, 0), -1)
                    
                    # Store result
                    results['faces'].append({
                        'name': name,
                        'confidence': float(prob),
                        'distance': float(distance),
                        'box': box.tolist()
                    })
                
                results['recognition_time'] = time.time() - recognition_start
        
        results['total_time'] = time.time() - start_time
        
        # Print results
        print(f"\n📊 Detection Results:")
        print(f"  • Faces detected: {results['num_faces']}")
        print(f"  • Detection time: {results['detection_time']*1000:.2f} ms")
        print(f"  • Recognition time: {results['recognition_time']*1000:.2f} ms")
        print(f"  • Total time: {results['total_time']*1000:.2f} ms")
        print(f"  • Speed: {1/results['total_time']:.2f} FPS")
        
        if results['faces']:
            print(f"\n👤 Recognized Faces:")
            for i, face in enumerate(results['faces'], 1):
                status = "✓" if face['name'] != "Unknown" else "✗"
                print(f"  {status} Face {i}: {face['name']} (confidence: {face['confidence']:.3f}, distance: {face['distance']:.3f})")
        
        # Save output image
        save_success = False
        if output_path:
            os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else 'output', exist_ok=True)
            save_success = cv2.imwrite(output_path, img_cv)
            
            if save_success:
                print(f"\n💾 Saved output to: {output_path}")
            else:
                print(f"\n❌ Error: Failed to save image to {output_path}")
                print("   The file might be open in another program. Please close it and try again.")
        
        # Display result
        if show_result and save_success:
            # Open the image with default viewer instead of matplotlib
            if output_path and os.path.exists(output_path):
                try:
                    # Windows
                    os.startfile(output_path)
                    print(f"🖼️  Opening image in default viewer...")
                except AttributeError:
                    # macOS/Linux
                    import subprocess
                    try:
                        subprocess.run(['open', output_path] if sys.platform == 'darwin' else ['xdg-open', output_path])
                        print(f"🖼️  Opening image in default viewer...")
                    except:
                        print(f"ℹ️  Please open the image manually: {output_path}")
        
        return results


import sys

def main():
    """Main demo function"""
    print("\n" + "="*60)
    print("Face Detection & Recognition Demo")
    print("Using MTCNN + FaceNet")
    print("="*60)
    
    # Initialize the face recognition system
    recognizer = FaceRecognitionDemo(
        database_path='sample_faces',
        threshold=0.6  # Adjust for stricter/looser matching
    )
    
    # Check if a specific image was provided via command line
    specific_image = None
    if len(sys.argv) > 1:
        specific_image = sys.argv[1]
        if not os.path.exists(specific_image):
            print(f"\n⚠️  Error: File '{specific_image}' not found!")
            return
        
        # Set up to process just this image
        test_images_path = os.path.dirname(specific_image)
        if not test_images_path: test_images_path = '.'
        test_images = [os.path.basename(specific_image)]
        print(f"\n🎯 Mode: Processing specific image: {specific_image}")
        
    else:
        # Standard Mode: Process all images in test_images folder
        
        # Create necessary directories
        os.makedirs('test_images', exist_ok=True)
        os.makedirs('output', exist_ok=True)
        
        # Process test images
        test_images_path = 'test_images'
        
        if not os.listdir(test_images_path):
            print(f"\n⚠️  No test images found in '{test_images_path}' folder")
            print("Please add some test images to process.")
            return
        
        # Get all images in folder
        supported_formats = ['.jpg', '.jpeg', '.png', '.bmp']
        test_images = [f for f in os.listdir(test_images_path) 
                       if os.path.splitext(f)[1].lower() in supported_formats]
    
        if not test_images:
            print(f"\n⚠️  No valid image files found in '{test_images_path}'")
            return
    
    all_results = []
    for image_file in test_images:
        image_path = os.path.join(test_images_path, image_file)
        output_path = os.path.join('output', f'detected_{image_file}')
        
        result = recognizer.process_image(
            image_path=image_path,
            output_path=output_path,
            show_result=True  # Automatically open images after processing
        )
        all_results.append(result)
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total images processed: {len(all_results)}")
    print(f"Total faces detected: {sum(r['num_faces'] for r in all_results)}")
    print(f"Average detection time: {np.mean([r['detection_time'] for r in all_results])*1000:.2f} ms")
    print(f"Average recognition time: {np.mean([r['recognition_time'] for r in all_results])*1000:.2f} ms")
    print(f"Average total time: {np.mean([r['total_time'] for r in all_results])*1000:.2f} ms")
    print(f"Average speed: {np.mean([1/r['total_time'] for r in all_results]):.2f} FPS")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
