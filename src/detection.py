"""YOLO detection helpers."""

import numpy as np
from PIL import Image
from ultralytics import YOLO

from src.config import CLASS_NAMES


def load_yolo(model_path: str) -> YOLO:
    """Load the YOLO model from disk (cached by the caller)."""
    return YOLO(model_path)


def run_detection(model: YOLO, image: Image.Image, conf: float, iou: float) -> list[dict]:
    """Run YOLO on a PIL image and return a list of detection dicts."""
    results = model.predict(source=image, conf=conf, iou=iou, verbose=False)
    if not results or results[0].boxes is None or len(results[0].boxes) == 0:
        return []
    boxes = results[0].boxes
    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    clss = boxes.cls.cpu().numpy().astype(int)
    names = getattr(results[0], "names", None) or getattr(model, "names", CLASS_NAMES)
    detections = []
    for i, (box, c, k) in enumerate(zip(xyxy, confs, clss)):
        x1, y1, x2, y2 = (int(v) for v in box)
        label = names.get(int(k), str(k)) if isinstance(names, dict) else names[int(k)]
        detections.append(
            {
                "id": i + 1,
                "label": f"Part {i + 1}",
                "class_id": int(k),
                "class_name": str(label),
                "confidence": float(c),
                "box": (x1, y1, x2, y2),
            }
        )
    return detections


def draw_boxes(image: Image.Image, detections: list[dict]) -> Image.Image:
    """Draw bounding boxes + labels on a copy of the image (PIL only)."""
    from PIL import ImageDraw

    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    colors = {"Bearing": "red", "Bolt": "green", "Gear": "blue", "Nut": "orange"}
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        color = colors.get(det["class_name"], "yellow")
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        text = f"{det['label']}: {det['class_name']} {det['confidence']:.2f}"
        tx0, ty0 = x1, max(0, y1 - 16)
        tx1, ty1 = x1 + max(120, len(text) * 7), y1
        draw.rectangle([tx0, ty0, tx1, ty1], fill=color)
        draw.text((tx0 + 2, ty0 + 1), text, fill="white")
    return annotated


def crop_detection(
    image: Image.Image, box: tuple[int, int, int, int], padding_ratio: float = 0.05
) -> Image.Image:
    """Crop a detection with proportional padding, clipped to image bounds."""
    w, h = image.size
    x1, y1, x2, y2 = box
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    pad_x = int((x2 - x1) * padding_ratio)
    pad_y = int((y2 - y1) * padding_ratio)
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)
    if x2 <= x1 or y2 <= y1:
        return image.copy()
    return image.crop((x1, y1, x2, y2))
