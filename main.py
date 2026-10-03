import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import settings
from agents.camera_agent import CameraAgent
from agents.track_processor import TrackProcessor
from agents.scoring import log_formula_header
from utils.db_utils import close_client
from agents.alert_agent import shutdown as alert_shutdown
from utils.llm_client import shutdown as llm_shutdown
from pipeline.models import Track
from runtime.background import start_llm_check, start_mongo_checks, start_prewarms
from runtime.api_server import ApiServer
from runtime.track_workers import TrackWorkers

import structlog

logger = structlog.get_logger(__name__)


def main():
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
    start_llm_check()

    # ── 3. Start worker threads (~1ms) ───────────────────────────
    track_processor = None
    workers = TrackWorkers()

    def process_track(track: Track):
        track_processor.process(track)

    workers.start(process_track)

    # ── 4. Start FastAPI server early (non-blocking) ─────────────
    api = ApiServer()
    api.start()

    # ── 5. MongoDB checks in background (non-blocking) ───────────
    start_mongo_checks()

    # ── 6. Pre-warm models in background (before camera.start()) ─
    start_prewarms()

    # ── 7. Construct TrackProcessor + CameraAgent ────────────────
    track_processor = TrackProcessor(loop=api.loop)

    def handle_track_finalized(track: Track):
        if track_processor and workers:
            track_processor.enqueue(workers.queue, track)

    def handle_frame_annotated(frame):
        if track_processor:
            track_processor.handle_frame(frame)

    camera = CameraAgent(
        on_track_finalized=handle_track_finalized,
        on_frame_annotated=handle_frame_annotated
    )

    # ── 8. Start camera (BLOCKS here until Ctrl+C) ──────────────
    logger.info("press_q_to_stop")
    camera.start()

    # ── 9. Graceful shutdown (only reached after camera stops) ───
    workers.stop()
    api.request_exit()
    workers.drain()
    track_processor.shutdown()
    alert_shutdown()
    llm_shutdown()
    close_client()
    api.finalize(timeout=5)
    logger.info("shutdown_complete")


if __name__ == "__main__":
    main()
