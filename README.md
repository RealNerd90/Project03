# Face Detection & Recognition Demo

A demonstration of face detection and recognition using MTCNN and FaceNet models.

## Features

- **Face Detection**: Uses MTCNN (Multi-task Cascaded Convolutional Networks) for accurate face detection
- **Face Recognition**: Uses FaceNet for generating face embeddings and recognition
- **Performance Metrics**: Displays detection speed, recognition accuracy, and FPS
- **Visual Annotations**: Draws bounding boxes, names, and confidence scores on detected faces

## Project Structure

```
Project03/
├── face_detection_demo.py    # Main demo script
├── requirements.txt           # Python dependencies
├── sample_faces/              # Reference face images (add your own)
├── test_images/               # Images to test face detection
├── output/                    # Processed images with annotations
└── README.md                  # This file
```

## Installation

1. Install Python 3.8 or higher

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Add Reference Faces

Add reference face images to the `sample_faces/` folder. The filename (without extension) will be used as the person's name.

Example:
```
sample_faces/
├── John_Doe.jpg
├── Jane_Smith.jpg
└── Bob_Johnson.png
```

### 2. Add Test Images

Add images you want to test to the `test_images/` folder.

### 3. Run the Demo

```bash
python face_detection_demo.py
```

The script will:
- Load all reference faces from `sample_faces/`
- Process all images in `test_images/`
- Detect faces using MTCNN
- Recognize faces using FaceNet
- Save annotated images to `output/`
- Display results with performance metrics

## Configuration

You can adjust parameters in the `FaceRecognitionDemo` class:

- `threshold`: Distance threshold for face matching (default: 0.6)
  - Lower value = stricter matching
  - Higher value = looser matching
  
- `min_face_size`: Minimum face size to detect (default: 20 pixels)

## Performance Metrics

The demo displays:
- **Detection time**: Time taken to detect faces in the image
- **Recognition time**: Time taken to recognize all detected faces
- **Total time**: Total processing time
- **FPS**: Frames per second (processing speed)
- **Confidence**: MTCNN detection confidence (0-1)
- **Distance**: Face embedding distance (lower = better match)

## Output

Processed images are saved to the `output/` folder with:
- Green bounding boxes for recognized faces
- Orange bounding boxes for unknown faces
- Name labels with distance scores
- Facial landmarks (eye, nose, mouth positions)

## Requirements

- Python 3.8+
- PyTorch
- facenet-pytorch
- MTCNN
- OpenCV
- NumPy
- Pillow
- Matplotlib

## Notes

- The first run will download pre-trained models (VGGFace2)
- GPU acceleration is automatically used if available
- Supports JPG, PNG, and BMP image formats
