import sys
import os
import time
import logging
import threading
import queue
import cv2
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import settings
from agents.camera_agent import CameraAgent
from agents.matching_agent import run_matching
from agents.decision_agent import decide
from agents.alert_agent import dispatch
from utils.db_utils import store_face, log_event
from utils.image_utils import save_image
from utils.embedding_utils import get_insightface
from pipeline.models import Track

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

worker_pool = None
track_queue = None


def handle_track_finalized(track: Track):
    if track_queue:
        track_queue.put(track)


def worker_process_tracks():
    while True:
        try:
            track = track_queue.get(timeout=1.0)
            process_finalized_track(track)
            track_queue.task_done()
        except queue.Empty:
            continue
        except Exception as e:
            logger.error(f"Worker error: {e}")


def process_finalized_track(track: Track):
    try:
        if track.embedding is None:
            logger.info(f"Track {track.track_id}: no embedding, skipping recognition")
            _log_event(track, "unknown", "none", False, None)
            return

        from agents.matching_agent import run_matching_from_embedding
        match_result = run_matching_from_embedding(track.embedding)

        decision = decide(track, match_result)

        image_url = None
        if track.best_full_frame is not None:
            image_dir = Path("captures/unknown_faces" if decision.status in ("unknown", "masked_unknown") else "captures/known_faces")
            image_dir.mkdir(parents=True, exist_ok=True)
            image_path = str(image_dir / f"{track.track_id}.jpg")
            save_image(track.best_full_frame, image_path)

        if decision.should_register and track.embedding:
            person_id = track.track_id
            name = "Unknown" if decision.status in ("unknown", "masked_unknown") else (match_result.name or "Unknown")
            role = "unknown" if decision.status in ("unknown", "masked_unknown") else (match_result.role or "visitor")
            tags = ["auto_registered"] if decision.status in ("unknown", "masked_unknown") else []

            try:
                store_face(
                    person_id=person_id,
                    name=name,
                    role=role,
                    embedding=track.embedding,
                    image_url=image_url,
                    tags=tags,
                    camera_id=settings.CAMERA_ID
                )
            except Exception as e:
                logger.error(f"Failed to store face: {e}")

        if decision.should_alert:
            dispatch(track, decision, image_url)
            track.alerted = True

        _log_event(track, decision.status, decision.alert_level,
                    track.alerted, match_result.similarity_score if match_result.matched else 0.0)

        logger.info(f"Track {track.track_id}: {decision.status} ({decision.alert_level})")

    except Exception as e:
        logger.error(f"Failed to process track {track.track_id}: {e}")


def _log_event(track: Track, status: str, alert_level: str,
               alerted: bool, similarity_score: float):
    try:
        log_event(
            track_id=track.track_id,
            camera_id=settings.CAMERA_ID,
            status=status,
            alert_level=alert_level,
            person_id=None,
            name=None,
            is_masked=track.is_masked,
            similarity_score=similarity_score,
            image_url=None,
            reason=f"Track finalized: {status}",
            alerted=alerted
        )
    except Exception as e:
        logger.error(f"Failed to log event: {e}")


def main():
    global worker_pool, track_queue

    try:
        settings.validate_config()
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)

    logger.info("Starting surveillance system...")
    logger.info(f"Camera: {settings.CAMERA_ID} (index {settings.CAMERA_INDEX})")
    logger.info(f"Alert channels: {settings.ALERT_CHANNELS}")

    track_queue = queue.Queue()
    worker_threads = []
    for i in range(2):
        t = threading.Thread(target=worker_process_tracks, daemon=True)
        t.start()
        worker_threads.append(t)

    camera = CameraAgent(on_track_finalized=handle_track_finalized)

    logger.info("Press 'Q' to stop")
    camera.start()


if __name__ == "__main__":
    main()
