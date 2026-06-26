# YOLOv8 Real-Time Object Detection

**Course:** Computer Vision — Lab Final SP 2026  
**Tools:** YOLOv8 · OpenCV · Ultralytics · Python

---

## Project Structure

```
yolo_detection/
├── config.py        # All settings (model, source, colors, thresholds)
├── detector.py      # Loads YOLOv8, runs inference, returns detections
├── video_source.py  # Abstracts webcam / video file input
├── display.py       # Draws bounding boxes, labels, FPS overlay
├── main.py          # Entry point — orchestrates the detection loop
└── requirements.txt
```

---

## Setup

```bash
pip install -r requirements.txt
```

The model (`yolov8n.pt`) downloads automatically on first run (~6 MB).

---

## Run

```bash
# Webcam (default)
python main.py

# Video file — edit SOURCE in config.py first:
#   SOURCE = "your_video.mp4"
python main.py
```

**Controls while running:**

| Key | Action |
|-----|--------|
| `Q` / `ESC` | Quit |
| `S` | Toggle saving output to `output.avi` |

---

## Configuration (`config.py`)

| Setting | Default | Description |
|---------|---------|-------------|
| `MODEL_NAME` | `yolov8n.pt` | Model size: n < s < m < l < x |
| `SOURCE` | `0` | `0` = webcam, `"file.mp4"` = video |
| `CONFIDENCE_THRESHOLD` | `0.40` | Min confidence to show a detection |
| `IOU_THRESHOLD` | `0.45` | NMS overlap threshold |
| `SHOW_FPS` | `True` | Overlay FPS counter |
| `SHOW_COUNT` | `True` | Show object count per frame |
| `SAVE_OUTPUT` | `False` | Write annotated video to disk |

---

## How It Works

```
Frame → [detector.py] → detections → [display.py] → annotated frame → window
  ↑
[video_source.py]
  ↑
[config.py] drives all parameters
```

1. `VideoSource` reads frames from webcam or file via `cv2.VideoCapture`.  
2. `ObjectDetector` passes each frame to YOLOv8 — the model predicts bounding boxes, class IDs, and confidence scores using a single forward pass through the CNN.  
3. Non-Maximum Suppression (NMS) removes overlapping duplicate detections.  
4. `FrameRenderer` draws colored boxes and labels using OpenCV primitives.  
5. `main.py` loops until `Q`/`ESC` or end of source.

---

## Technical Requirements Met

| Requirement | How |
|-------------|-----|
| ≥ 15 object classes | YOLOv8 detects 80 COCO classes out-of-the-box |
| ≥ 15 FPS on CPU | YOLOv8n achieves ~20–30 FPS on CPU |
| Bounding boxes + labels + confidence | `display.py` |
| Live webcam + video file | `video_source.py` with `config.SOURCE` |
| 3 frameworks | YOLOv8 (Ultralytics), OpenCV, PyTorch (YOLOv8 backend) |
| Real dataset | COCO (pre-trained weights, 80 categories, 330K images) |

---

## Dataset

Pre-trained on **COCO** (Common Objects in Context):
- 330,000 images, 80 object categories
- Categories include: person, car, bicycle, dog, cat, laptop, phone, chair, bottle, and 71 more.
