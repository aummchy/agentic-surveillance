"""One-shot background startup tasks (each runs on its own daemon thread).

These must never block camera startup; each task swallows and logs its own
failures so a missing service (Ollama, Atlas) only degrades features.
"""

import threading

import structlog

from config import settings
from utils.db_utils import check_atlas_search_index, backfill_missing_embeddings
from utils.llm_client import is_available as llm_available

logger = structlog.get_logger(__name__)


def check_llm_background():
    """Check LLM availability in background (non-blocking)."""
    if llm_available():
        logger.info("llm_connected", model=settings.OLLAMA_MODEL, url=settings.OLLAMA_URL)
    else:
        logger.info("llm_unavailable", model=settings.OLLAMA_MODEL, url=settings.OLLAMA_URL,
                       note="Alerts and reports will use template strings. Start Ollama to enable LLM features.")


def run_startup_checks():
    """Run MongoDB checks in background (non-blocking for camera startup)."""
    try:
        check_atlas_search_index()
    except Exception as e:
        logger.warning("atlas_check_failed", error=str(e))
    try:
        backfill_missing_embeddings()
    except Exception as e:
        logger.warning("backfill_failed", error=str(e))


def prewarm_yolo():
    """Pre-load YOLO model in background to avoid 1-3s cold load on first frame."""
    try:
        from pipeline.tracker import get_model
        get_model()
        logger.info("yolo_prewarm_complete")
    except Exception as e:
        logger.warning("yolo_prewarm_failed", error=str(e))


def prewarm_insightface():
    """Pre-load InsightFace model in background to avoid 2-5s cold load on first recognition."""
    try:
        from utils.embedding_utils import get_insightface
        get_insightface()
        logger.info("insightface_prewarm_complete")
    except Exception as e:
        logger.warning("insightface_prewarm_failed", error=str(e))


def start_llm_check():
    threading.Thread(target=check_llm_background, daemon=True).start()


def start_mongo_checks():
    threading.Thread(target=run_startup_checks, daemon=True).start()


def start_prewarms():
    threading.Thread(target=prewarm_yolo, daemon=True).start()
    threading.Thread(target=prewarm_insightface, daemon=True).start()
