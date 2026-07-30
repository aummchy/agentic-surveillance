import sys
import os
import threading
import queue
import asyncio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import settings
from agents.camera_agent import CameraAgent
from agents.track_processor import TrackProcessor
from agents.scoring import log_formula_header
from utils.db_utils import check_atlas_search_index, backfill_missing_embeddings, close_client
from agents.alert_agent import shutdown as alert_shutdown
from utils.llm_client import shutdown as llm_shutdown, is_available as llm_available
from pipeline.models import Track

import structlog

logger = structlog.get_logger(__name__)

track_queue = None
loop = None
track_processor = None
_shutdown_event = threading.Event()


def _check_llm_background():
    """Check LLM availability in background (non-blocking)."""
    if llm_available():
        logger.info("llm_connected", model=settings.OLLAMA_MODEL, url=settings.OLLAMA_URL)
    else:
        logger.info("llm_unavailable", model=settings.OLLAMA_MODEL, url=settings.OLLAMA_URL,
                       note="Alerts and reports will use template strings. Start Ollama to enable LLM features.")


def _run_startup_checks():
    """Run MongoDB checks in background (non-blocking for camera startup)."""
    try:
        check_atlas_search_index()
    except Exception as e:
        logger.warning("atlas_check_failed", error=str(e))
    try:
        backfill_missing_embeddings()
    except Exception as e:
        logger.warning("backfill_failed", error=str(e))


def _prewarm_yolo():
    """Pre-load YOLO model in background to avoid 1-3s cold load on first frame."""
    try:
        from pipeline.tracker import get_model
        get_model()
        logger.info("yolo_prewarm_complete")
    except Exception as e:
        logger.warning("yolo_prewarm_failed", error=str(e))


def _prewarm_insightface():
    """Pre-load InsightFace model in background to avoid 2-5s cold load on first recognition."""
    try:
        from utils.embedding_utils import get_insightface
        get_insightface()
        logger.info("insightface_prewarm_complete")
    except Exception as e:
        logger.warning("insightface_prewarm_failed", error=str(e))


def handle_track_finalized(track: Track):
    if track_processor and track_queue:
        track_processor.enqueue(track_queue, track)


def handle_frame_annotated(frame):
    if track_processor:
        track_processor.handle_frame(frame)


def worker_process_tracks():
    while not _shutdown_event.is_set():
        track = None
        try:
            track = track_queue.get(timeout=1.0)
            track_processor.process(track)
        except queue.Empty:
            continue
        except Exception as e:
            logger.error("worker_failed", track_id=track.track_id if track else None, error=str(e), exc_info=True)
        finally:
            if track is not None:
                track_queue.task_done()


def main():
    global track_queue, loop, track_processor

    # ── 1. Load + validate settings (~100ms) ─────────────────────
    try:
        settings.validate_config()
    except ValueError as e:
        logger.error("config_validation_failed", error=str(e))
        sys.exit(1)

    log_formula_header()

    logger.info("starting_surveillance",
                camera_id=settings.CAMERA_ID,
                camera_source=getattr(settings, "CAMERA_SOURCE", "") or settings.CAMERA_INDEX,
                alert_channels=settings.ALERT_CHANNELS)

    # ── 2. LLM check in background (non-blocking) ────────────────
    threading.Thread(target=_check_llm_background, daemon=True).start()

    # ── 3. Start worker threads (~1ms) ───────────────────────────
    track_queue = queue.Queue()
    worker_threads = []
    for i in range(2):
        t = threading.Thread(target=worker_process_tracks, daemon=True)
        t.start()
        worker_threads.append(t)

    # ── 4. Start FastAPI server early (non-blocking) ─────────────
    import uvicorn
    from dashboard.backend.main import app

    loop = asyncio.new_event_loop()

    config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="warning", access_log=False)
    server = uvicorn.Server(config)

    def run_server():
        loop.run_until_complete(server.serve())

    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    logger.info("dashboard_api_started", url="http://localhost:8000")

    # ── 5. MongoDB checks in background (non-blocking) ───────────
    threading.Thread(target=_run_startup_checks, daemon=True).start()

    # ── 6. Pre-warm models in background (before camera.start()) ─
    threading.Thread(target=_prewarm_yolo, daemon=True).start()
    threading.Thread(target=_prewarm_insightface, daemon=True).start()

    # ── 7. Construct TrackProcessor + CameraAgent ────────────────
    track_processor = TrackProcessor(loop=loop)

    camera = CameraAgent(
        on_track_finalized=handle_track_finalized,
        on_frame_annotated=handle_frame_annotated
    )

    # ── 8. Start camera (BLOCKS here until Ctrl+C) ──────────────
    logger.info("press_q_to_stop")
    camera.start()

    # ── 9. Graceful shutdown (only reached after camera stops) ───
    _shutdown_event.set()
    server.should_exit = True
    try:
        track_queue.join()
    except Exception:
        pass
    track_processor.shutdown()
    alert_shutdown()
    llm_shutdown()
    close_client()
    # Wait for uvicorn server thread to finish, then close event loop
    server_thread.join(timeout=5)
    try:
        loop.close()
    except Exception:
        pass
    logger.info("shutdown_complete")


if __name__ == "__main__":
    main()
