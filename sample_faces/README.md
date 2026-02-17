# Sample Faces Directory

This directory contains the reference face images used as the recognition database.

## Instructions

### Recommended structure (current)

Create **one folder per student** and store multiple angle images inside it:

```
sample_faces/
  Amar/
    left.jpg
    right.jpg
    up.jpg
    down.jpg
    front.jpg   (optional)
```

The app will also accept filenames like `Amar_left.jpg` inside the student folder, as long as the name contains:
`left`, `right`, `up`, `down`, or `front`.

## Tips for Best Results

- Use well-lit photos
- Face should be clearly visible and preferably looking at the camera
- Avoid heavy shadows or occlusions
- Higher resolution images work better
- Multiple angles of the same person improve accuracy (left/right/up/down)

## Example Structure

```
sample_faces/
├── Amar/
│   ├── left.jpg
│   ├── right.jpg
│   ├── up.jpg
│   └── down.jpg
└── Binod/
    ├── left.jpg
    ├── right.jpg
    ├── up.jpg
    └── down.jpg
```

Once you've added your reference images, run `attendance_system.py` to load/build the database.
