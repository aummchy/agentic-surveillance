import logging
import structlog
import numpy as np
from config import settings

logger = structlog.get_logger(__name__)

# Import YOLO — ultralytics' set_logging() runs here and adds its own handler
from ultralytics import YOLO

# Silence ultralytics AFTER import (overrides set_logging's handler)
_ul = logging.getLogger("ultralytics")
_ul.setLevel(logging.CRITICAL)
_ul.propagate = False
_ul.handlers.clear()

_model = None


def get_model() -> YOLO:
    global _model
    if _model is None:
        _model = YOLO(settings.YOLO_MODEL)
        logger.info("yolo_model_loaded", model=settings.YOLO_MODEL)
    return _model


def track_persons(frame: np.ndarray, persist: bool = True) -> list:
    model = get_model()
    results = model.track(
        frame,
        persist=persist,
        tracker="bytetrack.yaml",
        classes=[0],
        conf=settings.PERSON_CONF_THRESHOLD,
        iou=0.5,
        device=settings.YOLO_DEVICE,
        verbose=False
    )

    tracks = []
    for r in results:
        if r.boxes is None or r.boxes.id is None:
            continue

        ids = r.boxes.id.cpu().numpy()
        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()

        for tid, box, conf in zip(ids, boxes, confs):
            x1, y1, x2, y2 = map(int, box)
            tracks.append({
                "track_id": int(tid),
                "box": (x1, y1, x2, y2),
                "confidence": float(conf)
            })

    return tracks
