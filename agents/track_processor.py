import asyncio
import concurrent.futures
import queue
import threading
import time
import cv2
import structlog
from datetime import datetime, timezone
from config import settings
from config.status import Status, STATUS_LABELS
from pipeline.models import Track, DedupStatus, MatchResult
from agents.policy import decide
from agents.alert_agent import dispatch
from agents.memory import MemoryAgent
from agents.recognition import RecognitionAgent
from agents.matching_agent import run_matching_from_embedding
from utils.db_utils import store_face, log_event, deduplicate_identity
from utils.image_utils import resolve_track_image_url, resolve_track_person_crop_url
from dashboard.backend.routes.live import broadcast_frame, broadcast_alert, broadcast_event

logger = structlog.get_logger(__name__)

# Statuses that trigger auto-registration of new unknowns
_UNREGISTERED_STATUSES = (Status.UNKNOWN, Status.MASKED_UNKNOWN)


def _is_unregistered(status: int) -> bool:
    """Check if a status represents an unregistered person."""
    return status in _UNREGISTERED_STATUSES


def _get_display_name(match_result: MatchResult) -> str | None:
    """Extract a display name from match result, with person_id fallback."""
    if not match_result.matched:
        return None
    return match_result.name or (match_result.person_id or "unknown")[:8]


class TrackProcessor:
    """Orchestrates track finalization and dashboard broadcasting.

    Owns the executor for JPEG encoding and delegates to agents
    for matching, recognition, memory, and decision.
    """

    def __init__(self, loop, memory_agent=None):
        self._loop = loop
        self._memory_agent = memory_agent or MemoryAgent()
        self._encode_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="jpeg"
        )
        self._shutdown_event = threading.Event()
        self._frame_counter = 0

    def shutdown(self):
        self._shutdown_event.set()
        self._encode_executor.shutdown(wait=False)

    def enqueue(self, track_queue: queue.Queue, track: Track):
        track_queue.put(track)

    def handle_frame(self, frame):
        self._frame_counter += 1
        if self._frame_counter % settings.FRAME_SKIP != 0:
            return
        if self._loop and self._loop.is_running():
            def _encode_and_broadcast():
                try:
                    preview = cv2.resize(frame, (settings.BROADCAST_WIDTH, settings.BROADCAST_HEIGHT))
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
                self._log_event(track, Status.UNKNOWN, "none", False, 0.0, image_url)
                return

            logger.info("track_embedding_present",
                        track_id=snap.track_id,
                        embedding_len=len(snap.embedding),
                        is_masked=snap.is_masked,
                        face_detected=snap.face_detected_once,
                        frames_seen=snap.total_frames_seen)

            match_result = self._run_matching(snap)
            self._update_person_name(track, match_result)

            recognition_result, memory_context = self._run_recognition_and_memory(
                snap, match_result
            )

            decision = decide(track, match_result, recognition_result, memory_context)

            if decision.should_register:
                self._handle_registration(
                    snap, decision, match_result, image_url, person_crop_url
                )

            if match_result.matched:
                self._record_visit(match_result, snap, decision, memory_context)

            alert_dispatched = self._dispatch_alert(track, decision, image_url)

            if decision.should_alert and alert_dispatched:
                self._broadcast_alert(track, decision, image_url, person_crop_url)

            self._log_event(track, decision.status, decision.alert_level,
                            track.alerted, match_result.similarity_score,
                            image_url,
                            match_result.person_id if match_result.matched else None,
                            match_result.name if match_result.matched else None,
                            person_crop_url=person_crop_url)

            self._broadcast_event(
                snap, decision, match_result, image_url, person_crop_url,
                memory_context
            )

            logger.info("track_finalized",
                        track_id=track.track_id,
                        status=decision.status,
                        alert_level=decision.alert_level,
                        confidence=track.confidence,
                        person_id=match_result.person_id if match_result.matched else None,
                        name=_get_display_name(match_result),
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
                self._log_event(track, Status.UNKNOWN, "none", False, 0.0, getattr(track, 'image_url', None))
            except Exception as e2:
                logger.error("event_log_failed_after_error", track_id=track.track_id, error=str(e2), exc_info=True)

    def _run_matching(self, snap) -> MatchResult:
        """Run vector search matching if no pending result exists."""
        match_result = snap.pending_match_result
        if match_result is None:
            match_result = run_matching_from_embedding(snap.embedding)
        return match_result

    def _update_person_name(self, track: Track, match_result: MatchResult):
        """Update track's person name if this match has higher similarity."""
        if match_result.matched and match_result.name:
            with track._lock:
                if track.person_name_similarity == 0.0 or match_result.similarity_score >= track.person_name_similarity:
                    track.person_name = match_result.name
                    track.person_name_similarity = match_result.similarity_score

    def _run_recognition_and_memory(self, snap, match_result: MatchResult):
        """Run recognition and memory agents, using pending results when available."""
        fresh_match = snap.pending_match_result is None
        recognition_result = snap.pending_recognition if not fresh_match else None
        memory_context = snap.pending_memory_context if not fresh_match else None

        if memory_context is None and match_result.matched:
            memory_context = self._memory_agent.run({
                "person_id": match_result.person_id,
                "camera_id": settings.CAMERA_ID,
                "similarity": match_result.similarity_score,
                "status": Status.KNOWN,
            })

        if recognition_result is None:
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

        if recognition_result is None:
            logger.warning("recognition_result_none",
                           track_id=snap.track_id,
                           note="Recognition agent returned None, using empty dict")
            recognition_result = {}

        return recognition_result, memory_context

    def _handle_registration(self, snap, decision, match_result, image_url, person_crop_url):
        """Handle auto-registration of unknown persons."""
        if _is_unregistered(decision.status):
            name = "Unknown"
            role = "unknown"
            tags = ["auto_registered"]
        else:
            name = match_result.name or "Unknown"
            role = match_result.role or "visitor"
            tags = []

        if _is_unregistered(decision.status) and snap.best_face_score <= settings.REGISTRATION_QUALITY_MIN:
            logger.info("registration_skipped_low_quality",
                        track_id=snap.track_id,
                        quality=snap.best_face_score,
                        quality_min=settings.REGISTRATION_QUALITY_MIN)
            return

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
        elif dedup.status == DedupStatus.NEW:
            self._store_new_face(snap, name, role, tags, image_url, person_crop_url)
        else:
            logger.warning("dedup_unexpected_status",
                           track_id=snap.track_id,
                           status=dedup.status)

    def _store_new_face(self, snap, name, role, tags, image_url, person_crop_url):
        """Store a new face record in MongoDB."""
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

    def _record_visit(self, match_result, snap, decision, memory_context):
        """Record a visit in the memory agent."""
        self._memory_agent.record_visit(
            person_id=match_result.person_id,
            camera_id=settings.CAMERA_ID,
            status=decision.status,
            similarity=match_result.similarity_score,
            is_masked=snap.is_masked,
            visit_action=memory_context.get("action", "recorded") if memory_context else "recorded",
        )
        if memory_context:
            memory_context["visit_count"] = memory_context.get("visit_count", 0) + 1

    def _dispatch_alert(self, track, decision, image_url) -> bool:
        """Dispatch alert if conditions are met. Returns True if alert was sent."""
        if decision.should_alert and track.mark_alerted_once():
            alert_dispatched = dispatch(track, decision, image_url)
            if alert_dispatched:
                track.last_alert_time = time.time()
                return True
        return False

    def _broadcast_alert(self, track, decision, image_url, person_crop_url):
        """Broadcast alert payload via WebSocket."""
        alert_payload = {
            "person_id": track.track_id,
            "status": decision.status,
            "name": decision.name or "Unknown",
            "image_url": image_url,
            "person_crop_url": person_crop_url,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "camera_id": settings.CAMERA_ID,
            "reason": decision.reason,
            "alert_level": decision.alert_level,
            "nl_summary": decision.nl_summary or "",
        }
        if self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(broadcast_alert(alert_payload), self._loop)
        logger.debug("alert_broadcast_sent", track_id=track.track_id)

    def _broadcast_event(self, snap, decision, match_result, image_url,
                         person_crop_url, memory_context):
        """Broadcast event payload via WebSocket."""
        # Read from track snapshot (already captured)
        alerted_final = snap.alerted if hasattr(snap, 'alerted') else False
        person_name_final = getattr(snap, 'person_name', None)

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
                f"{settings.API_BASE_URL}/{snap.best_face_crop_path.replace(chr(92), '/')}"
                if snap.best_face_crop_path else None
            ),
            "reason": f"Track finalized: {STATUS_LABELS.get(decision.status, str(decision.status))}",
            "alerted": alerted_final,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(broadcast_event(event_payload), self._loop)

    def _log_event(self, track: Track, status: int, alert_level: str,
                   alerted: bool, similarity_score: float, image_url: str = None,
                   person_id: str = None, name: str = None,
                   person_crop_url: str = None):
        """Log event to MongoDB. Non-critical — failures are logged and swallowed."""
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
                reason=f"Track finalized: {STATUS_LABELS.get(status, str(status))}",
                alerted=alerted,
            )
        except Exception as e:
            logger.error("event_log_failed", error=str(e))
