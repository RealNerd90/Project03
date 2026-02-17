# Smart Attendance System (Face Recognition)

A Python/OpenCV attendance app that registers students from images or webcam and marks attendance using face recognition (MTCNN + FaceNet).

## Project structure

```
Project03/
├── attendance_system.py      # Main application (CLI + webcam UI)
├── requirements.txt          # Python dependencies
├── attendance_log.csv        # Output log (auto-created/appended)
├── output/                   # Saved annotated results (auto-created)
└── sample_faces/             # Face database (one folder per student)
    ├── _embeddings_cache.pt  # Embedding cache for fast startup (auto-created)
    ├── <StudentName>/
    │   ├── left.jpg          # Auto-captured angle reference
    │   ├── right.jpg
    │   ├── up.jpg
    │   ├── down.jpg
    │   └── front.jpg         # Optional (image-path registration)
    └── README.md             # Notes about face database folder
```

## How it works

- **Face detection**: `facenet_pytorch.MTCNN`
- **Face embedding**: `facenet_pytorch.InceptionResnetV1` (pretrained `vggface2`)
- **Matching**: Euclidean distance between the live embedding and all stored reference embeddings.
- **Multi-angle references**: Each student can have multiple reference images (angles). The best match across all references is used.
- **Fast startup**: The app caches computed embeddings to `sample_faces/_embeddings_cache.pt`. On later runs, unchanged images load instantly.

## Face database layout (important)

Create one folder per student inside `sample_faces/`:

```
sample_faces/
  Amar/
    Amar_left.jpg   (also accepts left.jpg)
    Amar_right.jpg
    Amar_up.jpg
    Amar_down.jpg
```

The loader accepts common naming patterns inside each student folder as long as the filename contains one of:
`left`, `right`, `up`, `down`, `front`.

## Setup

### 1) Install Python packages

From the project folder:

```powershell
cd C:\Users\HP\Documents\Project03
python -m pip install -r requirements.txt
```

If you change dependencies later:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 2) Run the app

```powershell
python attendance_system.py
```

## Usage

When you run `attendance_system.py`, you’ll see a simple menu:

- **1. Register Student**
  - **a. Enter Image Path**
    - Saves the image into `sample_faces/<Name>/front.jpg`
    - Reloads the database
  - **b. Capture from Camera**
    - Auto-detects face and auto-captures **4 angles**:
      - left / right / up / down
    - Creates `sample_faces/<Name>/` automatically
    - No manual “align box” is required; follow on-screen instructions

- **2. Mark Attendance**
  - Runs live webcam matching
  - Uses multiple reference images per student
  - Writes to `attendance_log.csv`

## Outputs

- **Attendance log**: `attendance_log.csv`
  - Columns: `Name, Date, Time`
- **Saved annotated images**: `output/` (created automatically)

## Performance notes

- Webcam capture attempts higher FPS/resolution (best-effort; depends on camera/driver).
- Database matching is vectorized and fast once the embedding cache is built.
- If you add/remove face images, the cache updates automatically on next run.

## Troubleshooting

- **Camera not opening**: try a different camera index when prompted (0, 1, 2...).
- **No face detected**: improve lighting, move closer, remove occlusions (mask/hand), avoid strong backlight.
- **Recognition poor**: ensure each student has clear angle images; re-register to capture cleaner references.

