import logging
import threading
import structlog
import numpy as np
from pathlib import Path
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
_model_lock = threading.Lock()

# ByteTrack config tuned for indoor fixed-camera surveillance
# Raised thresholds to reduce ID switches from occlusions and noise.
_TRACKER_CONFIG = str(Path(__file__).resolve().parent.parent / "config" / "bytetrack_surveillance.yaml")


def get_model() -> YOLO:
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                _model = YOLO(settings.YOLO_MODEL)
                logger.info("yolo_model_loaded", model=settings.YOLO_MODEL)
                if settings.DEBUG_RECOGNITION:
                    import yaml
                    with open(_TRACKER_CONFIG, "r") as f:
                        bt_cfg = yaml.safe_load(f)
                    logger.debug("bytetrack_config",
                                 track_high_thresh=bt_cfg.get("track_high_thresh"),
                                 track_low_thresh=bt_cfg.get("track_low_thresh"),
                                 new_track_thresh=bt_cfg.get("new_track_thresh"),
                                 track_buffer=bt_cfg.get("track_buffer"),
                                 match_thresh=bt_cfg.get("match_thresh"),
                                 fuse_score=bt_cfg.get("fuse_score"))
    return _model


def track_persons(frame: np.ndarray, persist: bool = True) -> list:
    model = get_model()
    results = model.track(
        frame,
        persist=persist,
        tracker=_TRACKER_CONFIG,
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
