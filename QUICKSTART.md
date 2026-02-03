# Quick Start Guide

## Step 1: Install Dependencies

Open a terminal in the Project03 folder and run:

```bash
pip install -r requirements.txt
```

This will install:
- PyTorch (deep learning framework)
- facenet-pytorch (FaceNet model)
- MTCNN (face detection)
- OpenCV (image processing)
- Other required libraries

**Note**: First run will download pre-trained models (~100MB)

## Step 2: Add Reference Faces

1. Go to the `sample_faces/` folder
2. Add clear photos of people you want to recognize
3. Name files with person's name: `PersonName.jpg`

Example:
```
sample_faces/
├── Alice.jpg
├── Bob.png
└── Charlie.jpg
```

## Step 3: Add Test Images

1. Go to the `test_images/` folder
2. Add images containing faces you want to detect
3. Can be group photos or individual portraits

## Step 4: Run the Demo

```bash
python face_detection_demo.py
```

The script will:
- ✅ Load reference faces and build database
- ✅ Process all test images
- ✅ Detect faces with MTCNN
- ✅ Recognize faces with FaceNet
- ✅ Save annotated images to `output/`
- ✅ Display performance metrics

## Expected Output

```
============================================================
Face Detection & Recognition Demo
Using MTCNN + FaceNet
============================================================
Using device: cuda:0 (or cpu)

Loading face database from sample_faces...
  ✓ Loaded: Alice
  ✓ Loaded: Bob
  ✓ Loaded: Charlie

Database loaded: 3 faces registered

============================================================
Processing: group_photo.jpg
============================================================

📊 Detection Results:
  • Faces detected: 3
  • Detection time: 125.34 ms
  • Recognition time: 45.67 ms
  • Total time: 171.01 ms
  • Speed: 5.85 FPS

👤 Recognized Faces:
  ✓ Face 1: Alice (confidence: 0.998, distance: 0.423)
  ✓ Face 2: Bob (confidence: 0.995, distance: 0.387)
  ✗ Face 3: Unknown (confidence: 0.992, distance: 0.892)

💾 Saved output to: output/detected_group_photo.jpg
```

## Customization

Edit `face_detection_demo.py` to adjust:

```python
recognizer = FaceRecognitionDemo(
    database_path='sample_faces',
    threshold=0.6  # Lower = stricter matching
)
```

Parameters:
- `threshold`: 0.4-0.5 (strict) to 0.7-0.8 (loose)
- Lower threshold = fewer false positives
- Higher threshold = more matches but less accurate

## Troubleshooting

### No faces detected
- Ensure images have clear, visible faces
- Check image quality and lighting
- Try adjusting `min_face_size` parameter

### Wrong recognition
- Add more reference images per person
- Adjust `threshold` value
- Ensure reference images are clear

### Slow performance
- Use GPU if available (CUDA)
- Reduce image resolution
- Process fewer images at once

## File Structure

```
Project03/
├── face_detection_demo.py    # Main script
├── requirements.txt           # Dependencies
├── README.md                  # Full documentation
├── QUICKSTART.md             # This guide
├── sample_faces/             # Reference images
│   └── README.md
├── test_images/              # Images to test
│   └── README.md
└── output/                   # Results (auto-generated)
    └── detected_*.jpg
```

Ready to test! 🚀
