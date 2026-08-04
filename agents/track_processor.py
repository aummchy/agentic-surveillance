import asyncio
import concurrent.futures
import queue
import threading
import time
import cv2
import structlog
from datetime import datetime
from config import settings
from pipeline.models import Track, DedupStatus
from agents.decision_agent import decide
from agents.alert_agent import dispatch
from agents.memory import MemoryAgent
from utils.db_utils import store_face, log_event, deduplicate_identity
from utils.image_utils import resolve_track_image_url, resolve_track_person_crop_url
from dashboard.backend.routes.live import broadcast_frame, broadcast_alert, broadcast_event

logger = structlog.get_logger(__name__)


class TrackProcessor:
    """Orchestrates track finalization and dashboard broadcasting.

    Owns the executor for JPEG encoding and delegates to agents
    for matching, recognition, memory, and decision.
    """

    def shutdown(self):
        self._shutdown_event.set()
        self._encode_executor.shutdown(wait=False)

    def enqueue(self, track_queue: queue.Queue, track: Track):
        track_queue.put(track)

    def __init__(self, loop, memory_agent=None):
        self._loop = loop
        self._memory_agent = memory_agent or MemoryAgent()
        self._encode_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="jpeg"
        )
        self._shutdown_event = threading.Event()
        self._frame_counter = 0

    def handle_frame(self, frame):
        self._frame_counter += 1
        if self._frame_counter % settings.FRAME_SKIP != 0:
            return
        if self._loop and self._loop.is_running():
            def _encode_and_broadcast():
                try:
                    preview = cv2.resize(frame, (1280, 720))
                    _, buffer = cv2.imencode(
                        ".jpg", preview,
                        [cv2.IMWRITE_JPEG_QUALITY, settings.JPEG_QUALITY_BROADCAST]
                    )
                    if buffer is not None:
                        asyncio.run_coroutine_threadsafe(
                            broadcast_frame(buffer.tobytes()), self._loop
                        )
                except Exception as e:
                    logger.error("frame_broadcast_failed", error=str(e))
            self._encode_executor.submit(_encode_and_broadcast)

    def process(self, track: Track):
        try:
            snap = track.snapshot()
            image_url = resolve_track_image_url(track)
            person_crop_url = resolve_track_person_crop_url(track)
            with track._lock:
                track.best_full_frame = None

            if snap.embedding is None:
                logger.info("track_no_embedding", track_id=snap.track_id,
                            face_detected=snap.face_detected_once,
                            frames_seen=snap.total_frames_seen,
                            visibility=snap.visibility)
                self._log_event(track, "unknown", "none", False, 0.0, image_url)
                return

            logger.info("track_embedding_present",
                        track_id=snap.track_id,
                        embedding_len=len(snap.embedding),
                        is_masked=snap.is_masked,
                        face_detected=snap.face_detected_once,
                        frames_seen=snap.total_frames_seen)

            match_result = snap.pending_match_result
            fresh_match = match_result is None
            if fresh_match:
                from agents.matching_agent import run_matching_from_embedding
                match_result = run_matching_from_embedding(snap.embedding)

            if match_result.matched and match_result.name:
                with track._lock:
                    if track.person_name_similarity == 0.0 or match_result.similarity_score >= track.person_name_similarity:
                        track.person_name = match_result.name
                        track.person_name_similarity = match_result.similarity_score

            recognition_result = snap.pending_recognition if not fresh_match else None
            memory_context = snap.pending_memory_context if not fresh_match else None
            if memory_context is None and match_result.matched:
                memory_context = self._memory_agent.run({
                    "person_id": match_result.person_id,
                    "camera_id": settings.CAMERA_ID,
                    "similarity": match_result.similarity_score,
                    "status": "known" if match_result.matched else "unknown",
                })

            if recognition_result is None:
                from agents.recognition import RecognitionAgent
                rec_agent = RecognitionAgent()
                track_duration = time.time() - snap.first_seen
                recognition_result = rec_agent.run({
                    "track_id": snap.track_id,
                    "similarity": match_result.similarity_score,
                    "is_masked": snap.is_masked,
                    "face_quality": snap.best_face_score if snap.best_face_score > 0 else None,
                    "track_duration": track_duration,
                    "memory_context": memory_context or {},
                })

            decision = decide(track, match_result, recognition_result, memory_context)

            if decision.should_register:
                name = "Unknown" if decision.status in ("unknown", "masked_unknown") else (match_result.name or "Unknown")
                role = "unknown" if decision.status in ("unknown", "masked_unknown") else (match_result.role or "visitor")
                tags = ["auto_registered"] if decision.status in ("unknown", "masked_unknown") else []

                if decision.status in ("unknown", "masked_unknown") and snap.best_face_score <= settings.REGISTRATION_QUALITY_MIN:
                    logger.info("registration_skipped_low_quality",
                                track_id=snap.track_id,
                                quality=snap.best_face_score,
                                quality_min=settings.REGISTRATION_QUALITY_MIN)
                else:
                    try:
                        dedup = deduplicate_identity(snap.embedding)
                    except Exception as e:
                        logger.error("dedup_failed", track_id=snap.track_id, error=str(e))
                        dedup = None

                    if dedup is not None and dedup.status == DedupStatus.MERGED:
                        logger.info("auto_register_merged",
                                    existing_person_id=dedup.person_id,
                                    similarity=round(dedup.similarity, 4),
                                    new_track_id=snap.track_id)
                    elif dedup is None or dedup.status == DedupStatus.FAILED:
                        logger.warning("dedup_unavailable",
                                       track_id=snap.track_id,
                                       reason=dedup.reason if dedup else "exception",
                                       note="Registration aborted to avoid duplicate")
                    else:
                        # DedupStatus.NEW — safe to insert
                        try:
                            stored_id = store_face(
                                person_id=snap.track_id,
                                name=name,
                                role=role,
                                embedding=snap.embedding,
                                image_url=image_url,
                                tags=tags,
                                camera_id=settings.CAMERA_ID,
                                quality_score=snap.best_face_score if snap.best_face_score > 0 else None,
                                person_crop_url=person_crop_url,
                            )
                            logger.info("store_face_success",
                                        track_id=snap.track_id,
                                        stored_person_id=stored_id,
                                        role=role,
                                        name=name,
                                        embedding_len=len(snap.embedding))
                        except Exception as e:
                            logger.error("store_face_failed", track_id=snap.track_id, error=str(e), exc_info=True)

            if match_result.matched:
                self._memory_agent.record_visit(
                    person_id=match_result.person_id,
                    camera_id=settings.CAMERA_ID,
                    status=decision.status,
                    similarity=match_result.similarity_score,
                    is_masked=snap.is_masked,
                    visit_action=memory_context.get("action", "recorded") if memory_context else "recorded",
                )
                # Increment visit_count in memory_context so the log below
                # reflects the count AFTER this visit (not before).
                if memory_context:
                    memory_context["visit_count"] = memory_context.get("visit_count", 0) + 1

            alert_dispatched = False
            if decision.should_alert and track.mark_alerted_once():
                alert_dispatched = dispatch(track, decision, image_url)
                if alert_dispatched:
                    track.last_alert_time = time.time()

            if decision.should_alert and alert_dispatched:
                alert_payload = {
                    "person_id": track.track_id,
                    "status": decision.status,
                    "name": decision.name or "Unknown",
                    "image_url": image_url,
                    "person_crop_url": person_crop_url,
                    "timestamp": datetime.utcnow().isoformat(),
                    "camera_id": settings.CAMERA_ID,
                    "reason": decision.reason,
                    "alert_level": decision.alert_level,
                    "nl_summary": decision.nl_summary or "",
                }
                if self._loop and self._loop.is_running():
                    asyncio.run_coroutine_threadsafe(broadcast_alert(alert_payload), self._loop)
                logger.debug("alert_broadcast_sent", track_id=track.track_id)

            self._log_event(track, decision.status, decision.alert_level,
                            track.alerted, match_result.similarity_score,
                            image_url, match_result.person_id if match_result.matched else None,
                            match_result.name if match_result.matched else None,
                            person_crop_url=person_crop_url)

            with track._lock:
                alerted_final = track.alerted
                person_name_final = track.person_name

            event_payload = {
                "track_id": snap.track_id,
                "camera_id": settings.CAMERA_ID,
                "status": decision.status,
                "alert_level": decision.alert_level,
                "person_id": match_result.person_id if match_result.matched else None,
                "name": match_result.name if match_result.matched else None,
                "person_name": person_name_final or (match_result.name if match_result.matched else None),
                "similarity_score": match_result.similarity_score,
                "image_url": image_url,
                "person_crop_url": person_crop_url,
                "best_face_crop_url": (
                    f"http://localhost:8000/{snap.best_face_crop_path.replace(chr(92), '/')}"
                    if snap.best_face_crop_path else None
                ),
                "reason": f"Track finalized: {decision.status}",
                "alerted": alerted_final,
                "timestamp": datetime.utcnow().isoformat(),
            }
            if self._loop and self._loop.is_running():
                asyncio.run_coroutine_threadsafe(broadcast_event(event_payload), self._loop)

            display_name = match_result.name if match_result.matched else None
            if not display_name and match_result.matched:
                display_name = (match_result.person_id or "unknown")[:8]
            logger.info("track_finalized",
                        track_id=track.track_id,
                        status=decision.status,
                        alert_level=decision.alert_level,
                        confidence=track.confidence,
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
            try:
                self._log_event(track, "error", "none", False, 0.0, getattr(track, 'image_url', None))
            except Exception:
                logger.error("event_log_failed_after_error", track_id=track.track_id, exc_info=True)

    def _log_event(self, track: Track, status: str, alert_level: str,
                   alerted: bool, similarity_score: float, image_url: str = None,
                   person_id: str = None, name: str = None,
                   person_crop_url: str = None):
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
                person_crop_url=person_crop_url,
                reason=f"Track finalized: {status}",
                alerted=alerted,
            )
        except Exception as e:
            logger.error("event_log_failed", error=str(e))