import structlog
import numpy as np
from ultralytics import YOLO
from config import settings

logger = structlog.get_logger(__name__)

_model = None


def get_model() -> YOLO:
    global _model
    if _model is None:
        _model = YOLO(settings.YOLO_MODEL)
        logger.info("yolo_model_loaded", model=settings.YOLO_MODEL)
    return _model


def detect_persons(frame: np.ndarray) -> list:
    model = get_model()
    results = model.predict(
        frame,
        conf=settings.PERSON_CONF_THRESHOLD,
        classes=[0],
        device=settings.YOLO_DEVICE,
        verbose=False
    )

    detections = []
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, xyxy)
            detections.append({
                "box": (x1, y1, x2, y2),
                "confidence": conf
            })

    return detections
