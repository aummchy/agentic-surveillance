import sys
import os
import time
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
from agents.scoring import log_formula_header
from utils.db_utils import store_face, log_event, check_atlas_search_index, backfill_missing_embeddings, close_client
from utils.image_utils import save_image, upload_to_cloudinary, upload_jpeg_to_cloudinary
from pipeline.models import Track
from dashboard.backend.routes.live import broadcast_frame, broadcast_alert, broadcast_event
from agents.alert_agent import shutdown as alert_shutdown
from utils.llm_client import shutdown as llm_shutdown, is_available as llm_available

import structlog

logger = structlog.get_logger(__name__)

track_queue = None
loop = None
memory_agent = MemoryAgent()
_encode_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="jpeg")
_shutdown_event = threading.Event()


def handle_track_finalized(track: Track):
    if track_queue:
        track_queue.put(track)


def handle_frame_annotated(frame):
    if loop and loop.is_running():
        def _encode_and_broadcast():
            try:
                _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, settings.JPEG_QUALITY_BROADCAST])
                if buffer is not None:
                    asyncio.run_coroutine_threadsafe(broadcast_frame(buffer.tobytes()), loop)
            except Exception as e:
                logger.error("frame_broadcast_failed", error=str(e))
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
            logger.error("worker_failed", track_id=track.track_id if track else None, error=str(e), exc_info=True)
        finally:
            if track is not None:
                track_queue.task_done()


def process_finalized_track(track: Track):
    try:
        image_url = track.image_url  # Reuse if already uploaded during progressive recognition
        if not image_url and track.best_full_frame is not None:
            # Use pre-encoded JPEG bytes if available (avoids re-encoding from numpy)
            if track.best_frame_jpeg:
                image_url = upload_jpeg_to_cloudinary(track.best_frame_jpeg)
                if not image_url:
                    # Fallback: save the JPEG bytes directly
                    save_image(track.best_full_frame, f"captures/{track.track_id}.jpg")
                    image_url = f"captures/{track.track_id}.jpg"
            else:
                save_image(track.best_full_frame, f"captures/{track.track_id}.jpg")
                image_url = upload_to_cloudinary(track.best_full_frame)
                if not image_url:
                    image_url = f"captures/{track.track_id}.jpg"
            logger.info("track_image_saved", track_id=track.track_id, url=image_url)
        # Release raw frame to free memory (JPEG bytes retained)
        track.best_full_frame = None

        if track.embedding is None:
            logger.info("track_no_embedding", track_id=track.track_id,
                        face_detected=track.face_detected_once,
                        frames_seen=track.total_frames_seen,
                        visibility=track.visibility)
            _log_event(track, "unknown", "none", False, 0.0, image_url)
            return

        logger.info("track_embedding_present",
                    track_id=track.track_id,
                    embedding_len=len(track.embedding),
                    is_masked=track.is_masked,
                    face_detected=track.face_detected_once,
                    frames_seen=track.total_frames_seen)

        # Reuse match result from progressive recognition if available
        match_result = getattr(track, 'pending_match_result', None)
        fresh_match = match_result is None
        if fresh_match:
            from agents.matching_agent import run_matching_from_embedding
            match_result = run_matching_from_embedding(track.embedding)

        if match_result.matched and match_result.name:
            track.person_name = match_result.name

        # Reuse recognition only if match was also reused (same embedding).
        # If match is fresh (embedding regenerated at finalization), re-compute recognition.
        recognition_result = getattr(track, 'pending_recognition', None) if not fresh_match else None
        # Reuse cached memory context only if match was reused
        memory_context = getattr(track, 'pending_memory_context', None) if not fresh_match else None
        if memory_context is None and match_result.matched:
            memory_context = memory_agent.run({
                "person_id": match_result.person_id,
                "camera_id": settings.CAMERA_ID,
                "similarity": match_result.similarity_score,
                "status": "known" if match_result.matched else "unknown",
            })

        # Re-compute recognition if not reused (fresh match or stale recognition)
        if recognition_result is None:
            from agents.recognition import RecognitionAgent
            rec_agent = RecognitionAgent()
            track_duration = time.time() - track.first_seen
            recognition_result = rec_agent.run({
                "similarity": match_result.similarity_score if match_result.matched else 0.0,
                "is_masked": track.is_masked,
                "face_quality": track.best_face_score if track.best_face_score > 0 else None,
                "track_duration": track_duration,
                "memory_context": memory_context or {},
            })

        decision = decide(track, match_result, recognition_result, memory_context)

        if decision.should_register:
            person_id = track.track_id
            name = "Unknown" if decision.status in ("unknown", "masked_unknown") else (match_result.name or "Unknown")
            role = "unknown" if decision.status in ("unknown", "masked_unknown") else (match_result.role or "visitor")
            tags = ["auto_registered"] if decision.status in ("unknown", "masked_unknown") else []

            merged = False
            if decision.status in ("unknown", "masked_unknown"):
                try:
                    from utils.db_utils import find_similar_unknowns, update_face
                    similar = find_similar_unknowns(track.embedding)
                    if similar:
                        existing_id = similar[0]["person_id"]
                        update_face(
                            existing_id,
                            image_url=image_url,
                            embedding=track.embedding,
                            quality_score=track.best_face_score,
                        )
                        logger.info("auto_register_merged",
                                    existing_person_id=existing_id,
                                    similarity=round(similar[0]["similarity_score"], 4),
                                    new_track_id=track.track_id)
                        merged = True
                except Exception as e:
                    logger.error("auto_register_dedup_failed", track_id=track.track_id, error=str(e))

            if not merged:
                try:
                    stored_id = store_face(
                        person_id=person_id,
                        name=name,
                        role=role,
                        embedding=track.embedding,
                        image_url=image_url,
                        tags=tags,
                        camera_id=settings.CAMERA_ID,
                        skip_search=False,
                        quality_score=track.best_face_score if track.best_face_score > 0 else None
                    )
                    logger.info("store_face_success",
                                track_id=track.track_id,
                                stored_person_id=stored_id,
                                role=role,
                                name=name,
                                embedding_len=len(track.embedding))
                except Exception as e:
                    logger.error("store_face_failed", track_id=track.track_id, error=str(e), exc_info=True)

        if match_result.matched:
            memory_agent.record_visit(
                person_id=match_result.person_id,
                camera_id=settings.CAMERA_ID,
                status=decision.status,
                similarity=match_result.similarity_score,
                is_masked=track.is_masked,
                visit_action=memory_context.get("action", "recorded") if memory_context else "recorded"
            )

        # Move face crop to person-name folder if matched
        best_crop_path = getattr(track, 'best_face_crop_path', None)
        person_name = match_result.name if match_result.matched else None
        if best_crop_path and person_name and os.path.exists(best_crop_path):
            import shutil
            src = best_crop_path
            filename = os.path.basename(src)
            dst_dir = f"captures/face_crops/{person_name}"
            dst = os.path.join(dst_dir, filename)
            try:
                os.makedirs(dst_dir, exist_ok=True)
                shutil.move(src, dst)
                setattr(track, 'best_face_crop_path', dst)
                best_crop_path = dst
                logger.debug("face_crop_moved", src=src, dst=dst, person=person_name)
            except Exception as e:
                logger.error("face_crop_move_failed", src=src, dst=dst, error=str(e))

        # Dispatch external alerts (email/sms/console) — respects global dedup
        alert_dispatched = False
        if decision.should_alert and not track.alerted:
            alert_dispatched = dispatch(track, decision, image_url)
            track.alerted = True

        # Broadcast alert to dashboard only when actually dispatched (no dedup bypass)
        if decision.should_alert and alert_dispatched:
            alert_payload = {
                "person_id": track.track_id,
                "status": decision.status,
                "name": decision.name or "Unknown",
                "image_url": image_url,
                "timestamp": datetime.utcnow().isoformat(),
                "camera_id": settings.CAMERA_ID,
                "reason": decision.reason,
                "alert_level": decision.alert_level,
                "nl_summary": decision.nl_summary or "",
            }
            if loop and loop.is_running():
                asyncio.run_coroutine_threadsafe(broadcast_alert(alert_payload), loop)
            logger.debug("alert_broadcast_sent", track_id=track.track_id)

        _log_event(track, decision.status, decision.alert_level,
                    track.alerted, match_result.similarity_score if match_result.matched else 0.0,
                    image_url, match_result.person_id if match_result.matched else None,
                    match_result.name if match_result.matched else None)

        event_payload = {
            "track_id": track.track_id,
            "camera_id": settings.CAMERA_ID,
            "status": decision.status,
            "alert_level": decision.alert_level,
            "person_id": match_result.person_id if match_result.matched else None,
            "name": match_result.name if match_result.matched else None,
            "person_name": track.person_name or (match_result.name if match_result.matched else None),
            "similarity_score": match_result.similarity_score if match_result.matched else 0.0,
            "image_url": image_url,
            "best_face_crop_url": f"http://localhost:8000/{best_crop_path.replace(chr(92), '/')}" if best_crop_path else None,
            "reason": f"Track finalized: {decision.status}",
            "alerted": track.alerted,
            "timestamp": datetime.utcnow().isoformat(),
        }
        if loop and loop.is_running():
            asyncio.run_coroutine_threadsafe(broadcast_event(event_payload), loop)

        display_name = match_result.name if match_result.matched else None
        if not display_name and match_result.matched:
            display_name = (match_result.person_id or "unknown")[:8]
        logger.info("track_finalized",
                    track_id=track.track_id,
                    status=decision.status,
                    alert_level=decision.alert_level,
                    confidence=recognition_result.get("confidence", 0) if recognition_result else 0,
                    person_id=match_result.person_id if match_result.matched else None,
                    name=display_name,
                    similarity=round(match_result.similarity_score, 4) if match_result.matched else None,
                    top2=round(match_result.second_best_similarity, 4) if match_result.matched and match_result.second_best_similarity is not None else None,
                    margin=round(match_result.margin, 4) if match_result.matched and match_result.margin is not None else None,
                    candidate_count=match_result.candidate_count if match_result.matched else 0,
                    visit_action=memory_context.get("action") if memory_context else None,
                    total_frames_seen=track.total_frames_seen,
                    frames_with_detectable_face=track.frames_with_detectable_face,
                    face_detected_once=track.face_detected_once,
                    is_masked=track.is_masked,
                    visibility=track.visibility,
                    best_face_score=track.best_face_score,
                    visit_count=memory_context.get("visit_count", 0) if memory_context else 0,
                    alerted=track.alerted)

    except Exception as e:
        logger.error("track_processing_failed", track_id=track.track_id, error=str(e), exc_info=True)
        # Always log an event so the track is not silently lost
        try:
            _log_event(track, "error", "none", False, 0.0, getattr(track, 'image_url', None))
        except Exception:
            logger.error("event_log_failed_after_error", track_id=track.track_id, exc_info=True)


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
    global track_queue, loop

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

    # Check LLM availability
    if llm_available():
        logger.info("llm_connected", model=settings.OLLAMA_MODEL, url=settings.OLLAMA_URL)
    else:
        logger.info("llm_unavailable", model=settings.OLLAMA_MODEL, url=settings.OLLAMA_URL,
                       note="Alerts and reports will use template strings. Start Ollama to enable LLM features.")

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
    server.should_exit = True
    try:
        track_queue.join()
    except Exception:
        pass
    _encode_executor.shutdown(wait=False)
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
