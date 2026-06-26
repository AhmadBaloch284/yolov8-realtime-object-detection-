# detector.py
from ultralytics import YOLO
import config

_ANIMAL_CLASSES = frozenset({"dog", "cat", "horse", "bear", "cow", "sheep"})
_PERSON_OVERLAP_IOU = 0.5
_REMOTE_CONFIDENCE_CUTOFF = 0.50
_PERSON_MIN_CONFIDENCE = 0.40
_PERSON_MIN_AREA_RATIO = 0.15
_LAPTOP_MAX_WIDTH = 400
_LAPTOP_MAX_HEIGHT = 500
_CHAIR_AS_PHONE_MAX_WIDTH = 350
_CHAIR_AS_PHONE_MAX_HEIGHT = 500
_PERSON_HAND_MAX_WIDTH = 200
_PERSON_HAND_MAX_HEIGHT = 250
_MOUSE_AS_PHONE_MAX_CONFIDENCE = 0.45
_CONFIRMED_PERSON_CONFIDENCE = 0.50
_SPLIT_PERSON_MAX_CONFIDENCE = 0.45


def _box_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter_area
    return inter_area / union if union > 0 else 0.0


def _set_cell_phone(det, class_names):
    name_to_id = {name: cid for cid, name in class_names.items()}
    phone_id = name_to_id.get("cell phone")
    det["class_name"] = "cell phone"
    if phone_id is not None:
        det["class_id"] = phone_id


def _correct_phone_remote_labels(detections, class_names):
    """Fix phone/remote confusion using confidence and person context."""
    has_person = any(d["class_name"] == "person" for d in detections)
    for det in detections:
        if det["class_name"] == "cell phone":
            continue
        if det["class_name"] != "remote":
            continue
        if det["confidence"] < _REMOTE_CONFIDENCE_CUTOFF or has_person:
            _set_cell_phone(det, class_names)
    return detections


def _correct_chair_labels(detections, class_names):
    """Relabel small chair boxes that are likely handheld phones."""
    persons = [d for d in detections if d["class_name"] == "person"]
    for det in detections:
        if det["class_name"] != "chair":
            continue
        x1, y1, x2, y2 = det["box"]
        width = x2 - x1
        height = y2 - y1
        if width >= _CHAIR_AS_PHONE_MAX_WIDTH or height >= _CHAIR_AS_PHONE_MAX_HEIGHT:
            continue
        near_person = any(
            _box_iou(det["box"], p["box"]) >= _PERSON_OVERLAP_IOU for p in persons
        )
        if near_person or not persons:
            _set_cell_phone(det, class_names)
    return detections


def _correct_laptop_labels(detections, class_names):
    """Fix laptop/phone confusion and drop redundant laptops."""
    for det in detections:
        if det["class_name"] != "laptop":
            continue
        x1, y1, x2, y2 = det["box"]
        width = x2 - x1
        height = y2 - y1
        if width < _LAPTOP_MAX_WIDTH and height < _LAPTOP_MAX_HEIGHT:
            _set_cell_phone(det, class_names)

    has_cell_phone = any(d["class_name"] == "cell phone" for d in detections)
    if not has_cell_phone:
        return detections

    return [d for d in detections if not (d["class_name"] == "laptop")]


def _filter_animals_overlapping_persons(detections):
    persons = [d for d in detections if d["class_name"] == "person"]
    if not persons:
        return detections

    filtered = []
    for det in detections:
        if det["class_name"] in _ANIMAL_CLASSES and any(
            _box_iou(det["box"], p["box"]) >= _PERSON_OVERLAP_IOU for p in persons
        ):
            continue
        filtered.append(det)
    return filtered


def _box_area(box):
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def _filter_false_persons(detections, frame):
    """Drop low-confidence small person boxes (hands/arms), keep full bodies."""
    h, w = frame.shape[:2]
    min_area = _PERSON_MIN_AREA_RATIO * w * h
    filtered = []
    for det in detections:
        if det["class_name"] == "person":
            if (
                det["confidence"] < _PERSON_MIN_CONFIDENCE
                and _box_area(det["box"]) < min_area
            ):
                continue
        filtered.append(det)
    return filtered


def _filter_hand_sized_false_persons(detections):
    """Remove small person boxes likely to be a hand-held mouse."""
    filtered = []
    for det in detections:
        if det["class_name"] == "person":
            x1, y1, x2, y2 = det["box"]
            if (
                x2 - x1 < _PERSON_HAND_MAX_WIDTH
                and y2 - y1 < _PERSON_HAND_MAX_HEIGHT
            ):
                continue
        filtered.append(det)
    return filtered


def _filter_mouse_as_cell_phone(detections):
    """Remove low-confidence phones when a confident person is present."""
    has_confirmed_person = any(
        d["class_name"] == "person" and d["confidence"] > _CONFIRMED_PERSON_CONFIDENCE
        for d in detections
    )
    if not has_confirmed_person:
        return detections
    return [
        d for d in detections
        if not (
            d["class_name"] == "cell phone"
            and d["confidence"] < _MOUSE_AS_PHONE_MAX_CONFIDENCE
        )
    ]


def _filter_split_persons(detections):
    """Keep only the best person when a weak duplicate split box exists."""
    persons = [d for d in detections if d["class_name"] == "person"]
    if len(persons) <= 1:
        return detections
    if not any(p["confidence"] < _SPLIT_PERSON_MAX_CONFIDENCE for p in persons):
        return detections
    best = max(persons, key=lambda p: p["confidence"])
    return [d for d in detections if d["class_name"] != "person" or d is best]


class ObjectDetector:

    def __init__(self):
        print(f"[Detector] Loading model: {config.MODEL_NAME}")
        self.model = YOLO(config.MODEL_NAME)
        print(f"[Detector] Model ready. {len(self.model.names)} classes available.")

    def detect(self, frame):
        
        # Full inference every frame — no caching or frame skipping
        results = self.model(
            frame,
            conf=config.CONFIDENCE_THRESHOLD,
            iou=config.IOU_THRESHOLD,
            imgsz=640,
            device="cpu",
            stream=False,
            verbose=False,
        )

        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                label = self.model.names[cls]

                detections.append({
                    "box":        [x1, y1, x2, y2],
                    "class_id":   cls,
                    "class_name": label,
                    "confidence": conf,
                })

        detections = _correct_phone_remote_labels(detections, self.model.names)
        detections = _filter_animals_overlapping_persons(detections)
        detections = _correct_chair_labels(detections, self.model.names)
        detections = _correct_laptop_labels(detections, self.model.names)
        detections = _filter_false_persons(detections, frame)
        detections = _filter_hand_sized_false_persons(detections)
        detections = _filter_mouse_as_cell_phone(detections)
        detections = _filter_split_persons(detections)
        return detections
