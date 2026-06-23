import sys
import os
import threading
import queue
import cv2
import asyncio
import concurrent.futures
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import settings
from agents.camera_agent import CameraAgent
from agents.decision_agent import decide
from agents.alert_agent import dispatch
from agents.memory import MemoryAgent
from utils.db_utils import store_face, log_event, check_atlas_search_index, backfill_missing_embeddings
from utils.image_utils import save_image, upload_to_cloudinary
from pipeline.models import Track
from dashboard.backend.routes.live import broadcast_frame, broadcast_alert

import structlog

logger = structlog.get_logger(__name__)

worker_pool = None
track_queue = None
loop = None
memory_agent = MemoryAgent()
_encode_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="jpeg")
_shutdown_event = threading.Event()


def handle_track_finalized(track: Track):
    if track_queue:
        track_queue.put(track)


def handle_frame_annotated(frame):
    if loop and loop.is_running():
        def _encode_and_broadcast():
            _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            asyncio.run_coroutine_threadsafe(broadcast_frame(buffer.tobytes()), loop)
        _encode_executor.submit(_encode_and_broadcast)


def worker_process_tracks():
    while not _shutdown_event.is_set():
        track = None
        try:
            track = track_queue.get(timeout=1.0)
            process_finalized_track(track)
        except queue.Empty:
            continue
        except Exception as e:
            logger.error("worker_failed", track_id=track.track_id if track else None, error=str(e))
        finally:
            if track is not None:
                track_queue.task_done()


def process_finalized_track(track: Track):
    try:
        image_url = track.image_url  # Reuse if already uploaded during progressive recognition
        if not image_url and track.best_full_frame is not None:
            save_image(track.best_full_frame, f"captures/{track.track_id}.jpg")
            image_url = upload_to_cloudinary(track.best_full_frame)
            if not image_url:
                image_url = f"captures/{track.track_id}.jpg"
            logger.info("track_image_saved", track_id=track.track_id, url=image_url)
        # Release raw frame to free memory (JPEG bytes retained)
        track.best_full_frame = None

        if track.embedding is None:
            logger.info("track_no_embedding", track_id=track.track_id)
            _log_event(track, "unknown", "none", False, 0.0, image_url)
            return

        from agents.matching_agent import run_matching_from_embedding
        match_result = run_matching_from_embedding(track.embedding)

        if match_result.matched and match_result.name:
            track.person_name = match_result.name

        recognition_result = getattr(track, 'pending_recognition', None)
        memory_context = {}
        if match_result.matched:
            memory_context = memory_agent.run({
                "person_id": match_result.person_id,
                "camera_id": settings.CAMERA_ID,
                "similarity": match_result.similarity_score,
                "status": "known" if match_result.matched else "unknown",
            })

        decision = decide(track, match_result, recognition_result, memory_context)

        if decision.should_register:
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
                logger.error("store_face_failed", track_id=track.track_id, error=str(e))

        if match_result.matched:
            memory_agent.record_visit(
                person_id=match_result.person_id,
                camera_id=settings.CAMERA_ID,
                status=decision.status,
                similarity=match_result.similarity_score,
                is_masked=track.is_masked
            )

        if decision.should_alert and not track.alerted:
            dispatch(track, decision, image_url)
            track.alerted = True

        if decision.status in ("unknown", "masked_unknown") and not match_result.matched:
            alert_payload = {
                "person_id": track.track_id,
                "status": decision.status,
                "name": "Unknown",
                "image_url": image_url,
                "timestamp": datetime.utcnow().isoformat(),
                "camera_id": settings.CAMERA_ID,
                "reason": decision.reason,
                "alert_level": decision.alert_level
            }
            if loop and loop.is_running():
                asyncio.run_coroutine_threadsafe(broadcast_alert(alert_payload), loop)
            logger.info("alert_broadcast_sent", track_id=track.track_id)

        _log_event(track, decision.status, decision.alert_level,
                    track.alerted, match_result.similarity_score if match_result.matched else 0.0,
                    image_url, match_result.person_id if match_result.matched else None,
                    match_result.name if match_result.matched else None)

        logger.info("track_finalized", track_id=track.track_id, status=decision.status, alert_level=decision.alert_level)

    except Exception as e:
        logger.error("track_processing_failed", track_id=track.track_id, error=str(e))


def _log_event(track: Track, status: str, alert_level: str,
               alerted: bool, similarity_score: float, image_url: str = None,
               person_id: str = None, name: str = None):
    try:
        log_event(
            track_id=track.track_id,
            camera_id=settings.CAMERA_ID,
            status=status,
            alert_level=alert_level,
            person_id=person_id,
            name=name,
            is_masked=track.is_masked,
            similarity_score=similarity_score,
            image_url=image_url,
            reason=f"Track finalized: {status}",
            alerted=alerted
        )
    except Exception as e:
        logger.error("event_log_failed", error=str(e))


def main():
    global worker_pool, track_queue, loop

    try:
        settings.validate_config()
    except ValueError as e:
        logger.error("config_validation_failed", error=str(e))
        sys.exit(1)

    logger.info("starting_surveillance",
                camera_id=settings.CAMERA_ID,
                camera_index=settings.CAMERA_INDEX,
                alert_channels=settings.ALERT_CHANNELS)

    try:
        check_atlas_search_index()
        backfill_missing_embeddings()
    except Exception as e:
        logger.warning("startup_check_failed", error=str(e))

    track_queue = queue.Queue()
    worker_threads = []
    for i in range(2):
        t = threading.Thread(target=worker_process_tracks, daemon=True)
        t.start()
        worker_threads.append(t)

    import uvicorn
    from dashboard.backend.main import app

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="warning", access_log=False)
    server = uvicorn.Server(config)

    def run_server():
        loop.run_until_complete(server.serve())

    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    logger.info("dashboard_api_started", url="http://localhost:8000")

    camera = CameraAgent(
        on_track_finalized=handle_track_finalized,
        on_frame_annotated=handle_frame_annotated
    )

    logger.info("press_q_to_stop")
    camera.start()

    # Graceful shutdown
    _shutdown_event.set()
    try:
        track_queue.join()
    except Exception:
        pass
    _encode_executor.shutdown(wait=False)
    logger.info("shutdown_complete")


if __name__ == "__main__":
    main()
