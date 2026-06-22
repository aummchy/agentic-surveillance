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
        logger.info("yolo_model_loaded_tracking", model=settings.YOLO_MODEL)
    return _model


def track_persons(frame: np.ndarray, persist: bool = True) -> list:
    model = get_model()
    results = model.track(
        frame,
        persist=persist,
        tracker="bytetrack.yaml",
        classes=[0],
        conf=settings.PERSON_CONF_THRESHOLD,
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
