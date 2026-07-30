import cv2
import time
import structlog
import threading
import concurrent.futures
import numpy as np
from config import settings
from pipeline.tracker import track_persons
from pipeline.track_state import TrackState
from pipeline.models import Track
from pipeline.recognition_pipeline import RecognitionPipeline
from utils.image_utils import draw_annotations, save_image, resolve_track_image_url, compute_iou
from agents.policy import RESOLVED_STATUSES
from agents.timing import TimingCollector

logger = structlog.get_logger(__name__)


class CameraAgent:
    def __init__(self, on_track_finalized=None, on_frame_annotated=None,
                 recognition_pipeline=None):
        self.track_state = TrackState()
        self.on_track_finalized = on_track_finalized
        self.on_frame_annotated = on_frame_annotated
        self._running = False
        self._cap = None
        self._frame_count = 0
        self._stop_event = threading.Event()
        self._pipeline = recognition_pipeline or RecognitionPipeline()
        self._recognizing_tracks = set()
        self._finalized_track_ids = set()
        self._track_sets_lock = threading.Lock()
        self._recognition_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="recognition"
        )
        self._timing = TimingCollector()

    @staticmethod
    def _camera_source():
        """Return RTSP URL string if CAMERA_SOURCE is set, else CAMERA_INDEX int."""
        src = getattr(settings, "CAMERA_SOURCE", "")
        if src:
            return src
        return settings.CAMERA_INDEX

    @staticmethod
    def _open_capture():
        """Open VideoCapture with configured backend (dshow/msmf/auto)."""
        source = CameraAgent._camera_source()
        backend = getattr(settings, "CAMERA_BACKEND", "")
        import cv2
        if backend and isinstance(source, int):
            be = getattr(cv2, f"CAP_{backend.upper()}", None)
            if be is not None:
                cap = cv2.VideoCapture(source, be)
            else:
                cap = cv2.VideoCapture(source)
        else:
            cap = cv2.VideoCapture(source)
        # Set read timeout for network streams so read() doesn't block indefinitely
        if isinstance(source, str):
            for prop in ("CAP_PROP_READ_TIMEOUT_MSEC", "CAP_PROP_OPEN_TIMEOUT_MSEC"):
                attr = getattr(cv2, prop, None)
                if attr is not None:
                    try:
                        cap.set(attr, 1000)
                    except Exception:
                        pass
        return cap

    def _apply_frame_props(self):
        """Set frame dimensions on the current capture and return actual resolution."""
        if self._cap and self._cap.isOpened():
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.FRAME_WIDTH)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.FRAME_HEIGHT)
            actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return f"{actual_w}x{actual_h}"
        return "unknown"

    def start(self):
        self._running = True

        source = CameraAgent._camera_source()

        # Open camera once (no test-then-reopen)
        self._cap = CameraAgent._open_capture()
        if not self._cap.isOpened():
            logger.error("camera_open_failed", source=source)
            self._cap.release()
            return

        resolution = self._apply_frame_props()

        logger.info("camera_started", source=source, backend=getattr(settings, "CAMERA_BACKEND", "auto"), resolution=resolution,
                    yolo_model=settings.YOLO_MODEL, face_model=settings.INSIGHTFACE_MODEL)

        try:
            self._loop()
        except KeyboardInterrupt:
            logger.info("Camera stopped by user")
        finally:
            self.stop()

    def stop(self):
        self._running = False
        self._stop_event.set()
        self._recognition_executor.shutdown(wait=True)
        if self._cap:
            self._cap.release()
        logger.info("camera_stopped")

    def _loop(self):
        consecutive_failures = 0
        max_failures = 10  # ~1 second at 100ms sleep
        source = CameraAgent._camera_source()
        is_file_source = isinstance(source, str) and "://" not in source
        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                if is_file_source:
                    logger.info("video_complete", source=source, frames=self._frame_count)
                    break
                consecutive_failures += 1
                if consecutive_failures >= max_failures:
                    logger.error("camera_disconnected", failures=consecutive_failures)
                    # Try to reconnect
                    self._cap.release()
                    time.sleep(1.0)
                    self._cap = CameraAgent._open_capture()
                    if self._cap.isOpened():
                        resolution = self._apply_frame_props()
                        logger.info("camera_reconnected", resolution=resolution)
                        consecutive_failures = 0
                    else:
                        logger.error("camera_reconnect_failed")
                        time.sleep(2.0)
                else:
                    time.sleep(0.1)
                continue

            consecutive_failures = 0

            self._frame_count += 1

            frame = cv2.resize(frame, (settings.FRAME_WIDTH, settings.FRAME_HEIGHT))
            tracks = track_persons(frame)

            if settings.DEBUG_RECOGNITION:
                logger.debug("recognition_cycle",
                             frame=self._frame_count,
                             track_count=len(tracks),
                             recognizing=len(self._recognizing_tracks))

            active_ids = set()
            for t in tracks:
                composite_id = self.track_state.make_composite_id(settings.CAMERA_ID, t["track_id"])
                active_ids.add(composite_id)
                track = self.track_state.update(settings.CAMERA_ID, t["track_id"], t["box"])

                if track and self._frame_count % settings.RECOGNITION_INTERVAL_FRAMES == 0:
                    if settings.DEBUG_RECOGNITION:
                        logger.debug("recognition_tick",
                                     track_id=track.track_id,
                                     frame=self._frame_count,
                                     decision=track.decision,
                                     best_score=round(track.best_face_score, 3),
                                     rescan_attempts=track.rescan_attempts)
                    # Skip recognition if already verified/known or high-confidence match
                    already_resolved = track.decision in RESOLVED_STATUSES
                    high_confidence = track.pending_match_result and track.pending_match_result.similarity_score > settings.HIGH_CONFIDENCE_SIMILARITY

                    # For unresolved tracks: re-run with time-based rescan for unknowns with similarity
                    should_skip = False
                    if already_resolved or high_confidence:
                        should_skip = True
                    elif track.last_recognition_status in ("unknown", "uncertain"):
                        has_similarity = (
                            track.pending_match_result
                            and track.pending_match_result.similarity_score > 0
                        )
                        if has_similarity and track.rescan_attempts < settings.MAX_RESCAN_ATTEMPTS:
                            time_since_last = time.time() - track.last_recognition_time
                            if time_since_last >= settings.RESCAN_INTERVAL_SECS:
                                track.rescan_attempts += 1
                            else:
                                should_skip = True
                        else:
                            quality_improved = track.best_face_score > (
                                track.last_recognition_quality + settings.MIN_QUALITY_IMPROVEMENT
                            )
                            if not quality_improved:
                                should_skip = True

                    if should_skip:
                        if settings.DEBUG_RECOGNITION:
                            skip_reason = "resolved" if already_resolved else "high_confidence" if high_confidence else "quality_throttle"
                            logger.debug("recognition_skipped",
                                         track_id=track.track_id,
                                         reason=skip_reason,
                                         decision=track.decision,
                                         current_score=round(track.best_face_score, 3),
                                         previous_best=round(track.last_recognition_quality, 3),
                                         required_delta=round(track.last_recognition_quality + settings.MIN_QUALITY_IMPROVEMENT, 3),
                                         delta=round(track.best_face_score - track.last_recognition_quality, 3))
                        logger.debug("skip_recognition_throttled",
                                   track_id=track.track_id,
                                   decision=track.decision,
                                   quality=round(track.best_face_score, 3),
                                   last_quality=round(track.last_recognition_quality, 3),
                                   last_status=track.last_recognition_status)
                    else:
                        with self._track_sets_lock:
                            already_recognizing = track.track_id in self._recognizing_tracks
                        if not already_recognizing:
                            logger.debug("progressive_recognition_scheduled",
                                       track_id=track.track_id,
                                       frame=self._frame_count)
                            if settings.DEBUG_RECOGNITION:
                                submit_time = self._timing.record_submit(track.track_id)
                                pending = self._timing.submit_pending_count()
                                logger.debug("recognition_scheduled",
                                             track_id=track.track_id,
                                             pending=pending)
                            self._recognition_executor.submit(
                                self._progressive_recognition, frame.copy(), track
                            )

            expired = self.track_state.get_expired_tracks()
            for track in expired:
                with self._track_sets_lock:
                    if track.track_id not in self._recognizing_tracks and track.track_id not in self._finalized_track_ids:
                        self._finalized_track_ids.add(track.track_id)
                        self._recognition_executor.submit(self._finalize_track, track)

            all_tracks = self.track_state.get_all()

            # Prune _finalized_track_ids to only keep active tracks
            active_track_ids = {t.track_id for t in all_tracks}
            with self._track_sets_lock:
                self._finalized_track_ids &= active_track_ids

            if getattr(settings, "DEBUG_DUPLICATE_BOXES", False):
                snapshot = self.track_state.debug_snapshot()
                logger.debug("duplicate_boxes_diag",
                             raw_track_count=len(tracks),
                             state_track_count=len(snapshot),
                             all_tracks_count=len(all_tracks),
                             active_ids=list(active_track_ids),
                             snapshot=snapshot)
                high_iou_pairs = []
                for i, a in enumerate(all_tracks):
                    for b in all_tracks[i + 1:]:
                        iou = compute_iou(a.person_box, b.person_box)
                        if iou > 0.3:
                            high_iou_pairs.append({
                                "pair": (a.track_id, b.track_id),
                                "iou": round(iou, 3),
                                "boxes": [a.person_box, b.person_box]
                            })
                if high_iou_pairs:
                    logger.debug("high_iou_track_pairs",
                                 pairs=high_iou_pairs,
                                 frame=self._frame_count)

            annotated = draw_annotations(frame, all_tracks)
            if self.on_frame_annotated:
                self.on_frame_annotated(annotated)

    def _progressive_recognition(self, frame: np.ndarray, track: Track):
        worker_start = time.perf_counter()
        if settings.DEBUG_RECOGNITION:
            submit_time = self._timing.pop_submit(track.track_id)
            queue_delay = (worker_start - submit_time) if submit_time is not None else None
            self._timing.record_start(track.track_id)
            logger.debug("recognition_worker_start",
                         track_id=track.track_id,
                         queue_delay_ms=round(queue_delay * 1000, 1) if queue_delay is not None else None,
                         thread=threading.current_thread().name)
        with self._track_sets_lock:
            self._recognizing_tracks.add(track.track_id)
        self.track_state.begin_recognition(track.track_id)

        # Guarantee every track gets at least one photo, independent of face quality
        self.track_state.ensure_fallback_frame(track.track_id, frame)

        try:
            result = self._pipeline.run(frame, track)

            if result.skip_reason == "high_confidence":
                return
            if result.skip_reason in ("no_face", "low_quality", "embedding_failed"):
                if result.quality and result.quality.is_valid and result.face_crop is not None:
                    self.track_state.set_best_face(
                        track.track_id, result.face_crop,
                        result.quality.overall_score, frame, result.face_ratio)
                return

            self.track_state.update_face_visibility(track.track_id, True, result.face_ratio)

            if result.face_crop is not None:
                self.track_state.set_best_face(
                    track.track_id, result.face_crop,
                    result.quality.overall_score, frame, result.face_ratio)
                if getattr(settings, 'DEBUG_FACE_CROPS', False):
                    crop_path = f"captures/debug/face_crops/{track.track_id}/{self._frame_count}.jpg"
                    save_image(result.face_crop, crop_path)
                    self.track_state.set_face_crop_path(track.track_id, crop_path)
                    logger.info("face_crop_saved", track_id=track.track_id, path=crop_path,
                                url=f"http://localhost:8000/{crop_path.replace(chr(92), '/')}")

            if result.embedding is not None:
                self.track_state.set_embedding(
                    track.track_id, result.embedding,
                    result.is_masked, det_score=result.det_score)
                self.track_state.set_cached_embedding(track.track_id, result.embedding)

            if result.match is not None:
                if result.match.matched and result.match.name:
                    self.track_state.set_person_name(track.track_id, result.match.name)
                self.track_state.set_pending_match_result(track.track_id, result.match)

            if result.recognition is not None:
                self.track_state.set_pending_recognition_data(track.track_id, result.recognition)

            if result.memory:
                self.track_state.set_pending_memory_context(track.track_id, result.memory)

            if result.decision is not None and result.recognition is not None:
                new_confidence = int(result.recognition.get("confidence", 0))

                if settings.DEBUG_RECOGNITION:
                    logger.debug("decision_check",
                                 track_id=track.track_id,
                                 new_confidence=new_confidence,
                                 existing_confidence=track.confidence,
                                 decision_status=result.decision.status,
                                 should_alert=result.decision.should_alert,
                                 alert_level=result.decision.alert_level)

                if result.decision.should_alert and result.decision.alert_level == "critical" and not track.alerted:
                    with self._track_sets_lock:
                        already_finalized = track.track_id in self._finalized_track_ids
                    if not already_finalized:
                        from agents.alert_agent import dispatch
                        image_url = resolve_track_image_url(track)
                        dispatch(track, result.decision, image_url)
                        self.track_state.set_decision(track.track_id, result.decision.status, True)
                        track.confidence = new_confidence
                        logger.info("progressive_critical_alert",
                                    track_id=track.track_id,
                                    alert_level=result.decision.alert_level,
                                    status=result.decision.status)
                elif new_confidence > track.confidence:
                    self.track_state.set_decision(track.track_id, result.decision.status, track.alerted)
                    track.confidence = new_confidence
                    if result.decision.should_alert:
                        logger.debug("progressive_alert_deferred_to_finalization",
                                     track_id=track.track_id,
                                     alert_level=result.decision.alert_level,
                                     status=result.decision.status)
                else:
                    logger.debug("confidence_not_upgraded",
                                 track_id=track.track_id,
                                 new=new_confidence,
                                 existing=track.confidence,
                                 status=result.decision.status)

                self.track_state.set_recognition_snapshot(
                    track.track_id, track.best_face_score, result.decision.status)

            if settings.DEBUG_RECOGNITION:
                logger.debug("recognition_timing",
                             track_id=track.track_id,
                             crop_detect_ms=result.metrics.crop_detect_ms,
                             fallback_detect_ms=result.metrics.fallback_detect_ms,
                             fallback_used=result.metrics.fallback_used,
                             person_crop_shape=result.metrics.person_crop_shape,
                             faces_on_crop=result.metrics.faces_on_crop,
                             detect_ms=result.metrics.detect_ms,
                             quality_ms=result.metrics.quality_ms,
                             embed_ms=result.metrics.embed_ms,
                             db_ms=result.metrics.db_ms,
                             memory_ms=result.metrics.memory_ms,
                             recog_policy_ms=result.metrics.recog_policy_ms,
                             total_ms=result.metrics.total_ms)

        except Exception as e:
            logger.error("progressive_recognition_failed", track_id=track.track_id, error=str(e), exc_info=True)
        finally:
            if settings.DEBUG_RECOGNITION:
                duration_ms = self._timing.get_duration_ms(track.track_id)
                current_track = self.track_state.get(track.track_id)
                track_stale = current_track is not track if current_track else True
                actual_decision = current_track.decision if current_track else "track_removed"
                logger.debug("recognition_worker_done",
                             track_id=track.track_id,
                             duration_ms=duration_ms,
                             decision=track.decision,
                             actual_decision=actual_decision,
                             confidence=track.confidence,
                             track_stale=track_stale)
            self.track_state.end_recognition(track.track_id)
            with self._track_sets_lock:
                self._recognizing_tracks.discard(track.track_id)
                if track.track_id not in self._finalized_track_ids and self.track_state.get(track.track_id) is None:
                    self._finalized_track_ids.add(track.track_id)
                    try:
                        self._recognition_executor.submit(self._finalize_track, track)
                    except RuntimeError:
                        logger.debug("finalize_submit_after_shutdown", track_id=track.track_id)

    def _finalize_track(self, track: Track):
        try:
            from agents.finalizer import retry_embedding
            retry_embedding(track, set_embedding=self.track_state.set_embedding)
        except Exception as e:
            logger.error("track_finalization_failed", track_id=track.track_id, error=str(e), exc_info=True)
        finally:
            if self.on_track_finalized:
                self.on_track_finalized(track)
