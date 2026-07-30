import cv2
import time
import structlog
import threading
import concurrent.futures
import numpy as np
from config import settings
from pipeline.tracker import track_persons
from pipeline.track_state import TrackState
from pipeline.quality_agent import compute_quality
from pipeline.face import compute_face_ratio
from pipeline.models import Track, QualityResult
from utils.image_utils import crop_person, resize_image, draw_annotations, save_image, upload_to_cloudinary, resolve_track_image_url
from utils.embedding_utils import get_insightface
from agents.recognition import RecognitionAgent
from agents.policy import RESOLVED_STATUSES
from agents.memory import MemoryAgent

logger = structlog.get_logger(__name__)

# If new embedding's cosine distance from cached is below this threshold,
# skip the Atlas vector search round-trip and reuse the last match result.
EMBEDDING_CACHE_COSINE_THRESHOLD = 0.005


class CameraAgent:
    def __init__(self, on_track_finalized=None, on_frame_annotated=None):
        self.track_state = TrackState()
        self.on_track_finalized = on_track_finalized
        self.on_frame_annotated = on_frame_annotated
        self._running = False
        self._cap = None
        self._frame_count = 0
        self._stop_event = threading.Event()
        self.recognition_agent = RecognitionAgent()
        self.memory_agent = MemoryAgent()
        self._recognizing_tracks = set()
        self._finalized_track_ids = set()
        self._track_sets_lock = threading.Lock()
        self._recognition_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="recognition"
        )
        self._recognition_submit_times: dict = {}
        self._recognition_submit_lock = threading.Lock()
        self._recognition_start_times: dict = {}
        self._recognition_start_lock = threading.Lock()

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
                                submit_time = time.perf_counter()
                                with self._recognition_submit_lock:
                                    self._recognition_submit_times[track.track_id] = submit_time
                                    pending = len(self._recognition_submit_times)
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

            annotated = draw_annotations(frame, all_tracks)
            if self.on_frame_annotated:
                self.on_frame_annotated(annotated)

    def _progressive_recognition(self, frame: np.ndarray, track: Track):
        worker_start = time.perf_counter()
        if settings.DEBUG_RECOGNITION:
            with self._recognition_submit_lock:
                submit_time = self._recognition_submit_times.pop(track.track_id, None)
            queue_delay = (worker_start - submit_time) if submit_time is not None else None
            with self._recognition_start_lock:
                self._recognition_start_times[track.track_id] = worker_start
            logger.debug("recognition_worker_start",
                         track_id=track.track_id,
                         queue_delay_ms=round(queue_delay * 1000, 1) if queue_delay is not None else None,
                         thread=threading.current_thread().name)
        with self._track_sets_lock:
            self._recognizing_tracks.add(track.track_id)
        self.track_state.begin_recognition(track.track_id)
        try:
            t_total = time.perf_counter()

            # Skip InsightFace if already matched with high confidence
            if track.pending_match_result and track.pending_match_result.similarity_score > settings.HIGH_CONFIDENCE_SIMILARITY:
                logger.debug("skip_recognition_high_confidence",
                           track_id=track.track_id,
                           similarity=track.pending_match_result.similarity_score)
                return

            t_detect = time.perf_counter()
            person_crop = crop_person(frame, track.person_box)
            if person_crop.size == 0:
                logger.debug("empty_person_crop", track_id=track.track_id)
                return

            # Guarantee every track gets at least one photo, independent of face quality
            self.track_state.ensure_fallback_frame(track.track_id, frame)

            app = get_insightface()

            # Two-stage detection: person crop first (more focused), then full frame
            # when the crop has no faces with embedding-grade quality.
            t_before_crop_detect = time.perf_counter()
            crop_faces = app.detect_faces_raw(person_crop, min_score=settings.DET_SCORE_RELAXED)
            t_after_crop_detect = time.perf_counter()

            crop_has_embedding_quality = any(f["det_score"] >= settings.EMBEDDING_DET_SCORE_MIN for f in crop_faces)
            fallback_used = not crop_has_embedding_quality
            if fallback_used:
                frame_faces = app.detect_faces_raw(frame, min_score=settings.DET_SCORE_RELAXED)
            else:
                frame_faces = []
            t_after_fallback_detect = time.perf_counter()

            crop_detect_ms = round((t_after_crop_detect - t_before_crop_detect) * 1000, 1)
            fallback_detect_ms = round((t_after_fallback_detect - t_after_crop_detect) * 1000, 1) if fallback_used else 0.0
            person_crop_shape = f"{person_crop.shape[1]}x{person_crop.shape[0]}" if person_crop.size > 0 else "empty"
            faces_on_crop = len(crop_faces)

            best = None
            detected_in_person_crop = False
            if crop_faces:
                best = crop_faces[0]
                detected_in_person_crop = True
                logger.debug("face_found_crop", track_id=track.track_id, score=best["det_score"])
            elif frame_faces:
                best = frame_faces[0]
                logger.debug("face_found_in_full_frame", track_id=track.track_id, score=best["det_score"])

            if not best:
                logger.debug("no_face_anywhere", track_id=track.track_id)
                return

            # Convert bbox to frame coordinates if detected in person_crop
            frame_bbox = None
            if best["bbox"]:
                fx1, fy1, fx2, fy2 = best["bbox"]
                if detected_in_person_crop:
                    px1, py1, _, _ = track.person_box
                    frame_bbox = (fx1 + int(px1), fy1 + int(py1), fx2 + int(px1), fy2 + int(py1))
                else:
                    frame_bbox = (fx1, fy1, fx2, fy2)

            face_ratio = 0.0
            if frame_bbox:
                face_ratio = compute_face_ratio(frame_bbox, track.person_box)

            self.track_state.update_face_visibility(track.track_id, True, face_ratio)

            if best["det_score"] < settings.EMBEDDING_DET_SCORE_MIN:
                logger.debug("embedding_score_below_threshold",
                           track_id=track.track_id,
                           score=best["det_score"],
                           threshold=settings.EMBEDDING_DET_SCORE_MIN)
                return

            if frame_bbox:
                fx1, fy1, fx2, fy2 = frame_bbox
                face_crop = frame[fy1:fy2, fx1:fx2]
            else:
                face_crop = person_crop

            t_quality = time.perf_counter()
            if face_crop.size > 0:
                quality = compute_quality(face_crop)
                logger.debug("face_quality",
                           track_id=track.track_id,
                           score=quality.overall_score,
                           valid=quality.is_valid,
                           blur=quality.blur_score,
                           brightness=quality.brightness,
                           area=quality.face_area)
            else:
                quality = QualityResult.invalid()

            if quality.is_valid:
                self.track_state.set_best_face(
                    track.track_id,
                    face_crop,
                    quality.overall_score,
                    frame,
                    face_ratio
                )
                if getattr(settings, 'DEBUG_FACE_CROPS', False):
                    crop_path = f"captures/debug/face_crops/{track.track_id}/{self._frame_count}.jpg"
                    save_image(face_crop, crop_path)
                    self.track_state.set_face_crop_path(track.track_id, crop_path)
                    logger.info("face_crop_saved", track_id=track.track_id, path=crop_path,
                                url=f"http://localhost:8000/{crop_path.replace(chr(92), '/')}")

            else:
                # Do not generate or search embeddings from low-quality faces.
                # Wait for a later recognition attempt with a better-quality frame.
                logger.debug(
                    "skip_recognition_low_quality",
                    track_id=track.track_id,
                    blur=quality.blur_score,
                    brightness=round(quality.brightness, 1),
                    area=quality.face_area,
                )
                return

            t_embed = time.perf_counter()
            embedding_list = best["embedding"].tolist()
            self.track_state.set_embedding(
                track.track_id,
                embedding_list,
                best["is_masked"],
                det_score=best["det_score"]
            )

            # Check if embedding is nearly identical to last searched one
            match_result = None
            if track.cached_embedding is not None and track.pending_match_result is not None:
                a = np.array(track.cached_embedding, dtype=np.float32)
                b = np.array(embedding_list, dtype=np.float32)
                norm_a = np.linalg.norm(a)
                norm_b = np.linalg.norm(b)
                if norm_a > 0 and norm_b > 0:
                    cos_dist = 1.0 - float(np.dot(a, b) / (norm_a * norm_b))
                    if cos_dist < EMBEDDING_CACHE_COSINE_THRESHOLD:
                        match_result = track.pending_match_result
                        logger.debug("embedding_cache_hit", track_id=track.track_id, distance=cos_dist)

            t_db = time.perf_counter()
            if match_result is None:
                from agents.matching_agent import run_matching_from_embedding
                match_result = run_matching_from_embedding(embedding_list, track_id=track.track_id)
                self.track_state.set_cached_embedding(track.track_id, embedding_list)

            t_memory = time.perf_counter()
            # Phase 2.2: Use Memory Agent for context (skip for high-confidence matches)
            memory_context = {}
            if match_result.matched and match_result.person_id:
                if match_result.similarity_score > settings.HIGH_CONFIDENCE_SIMILARITY:
                    logger.debug("skip_memory_high_confidence",
                               track_id=track.track_id,
                               similarity=match_result.similarity_score)
                    memory_context = {"skip_reason": "high_confidence", "confidence_boost": 0}
                else:
                    memory_context = self.memory_agent.run({
                        "person_id": match_result.person_id,
                        "camera_id": settings.CAMERA_ID,
                        "similarity": match_result.similarity_score,
                        "status": "known" if match_result.matched else "unknown",
                    })

            t_recog = time.perf_counter()
            # Phase 2.1: Use Recognition Agent with memory context
            track_duration = time.time() - track.first_seen
            recognition_result = self.recognition_agent.run({
                "similarity": match_result.similarity_score if match_result.matched else 0.0,
                "is_masked": best["is_masked"],
                "face_quality": quality.overall_score if hasattr(quality, 'overall_score') and quality.overall_score > 0 else None,
                "track_duration": track_duration,
                "memory_context": memory_context,
                "top2": match_result.second_best_similarity if match_result else None,
                "margin": match_result.margin if match_result else None,
                "name": match_result.name if match_result and match_result.matched else None,
                "track_id": track.track_id,
            })

            # Store recognition result in track for later use
            self.track_state.set_pending_recognition_data(track.track_id, recognition_result)

            # Phase 2.3: Use Policy Agent with all context
            from agents.decision_agent import decide
            decision = decide(track, match_result, recognition_result, memory_context)

            # Store person name on track for bounding box display
            if match_result.matched and match_result.name:
                self.track_state.set_person_name(track.track_id, match_result.name)

            # Only dispatch CRITICAL alerts (blacklist) during progressive recognition.
            # All other alerts are deferred to finalization to avoid premature alerts
            # for verified/known users when early recognition attempts produce low similarity.
            new_confidence = int(recognition_result.get("confidence", 0))

            if settings.DEBUG_RECOGNITION:
                logger.debug("decision_check",
                             track_id=track.track_id,
                             new_confidence=new_confidence,
                             existing_confidence=track.confidence,
                             decision_status=decision.status,
                             should_alert=decision.should_alert,
                             alert_level=decision.alert_level)

            if decision.should_alert and decision.alert_level == "critical" and not track.alerted:
                with self._track_sets_lock:
                    already_finalized = track.track_id in self._finalized_track_ids
                if not already_finalized:
                    from agents.alert_agent import dispatch
                    image_url = resolve_track_image_url(track)
                    dispatch(track, decision, image_url)
                    self.track_state.set_decision(track.track_id, decision.status, True)
                    track.confidence = new_confidence
                    logger.info("progressive_critical_alert",
                                track_id=track.track_id,
                                alert_level=decision.alert_level,
                                status=decision.status)
            elif new_confidence > track.confidence:
                # Only upgrade — never downgrade confidence across recognition passes
                self.track_state.set_decision(track.track_id, decision.status, track.alerted)
                track.confidence = new_confidence
                if decision.should_alert:
                    logger.debug("progressive_alert_deferred_to_finalization",
                                 track_id=track.track_id,
                                 alert_level=decision.alert_level,
                                 status=decision.status)
            else:
                logger.debug("confidence_not_upgraded",
                             track_id=track.track_id,
                             new=new_confidence,
                             existing=track.confidence,
                             status=decision.status)

            # Record quality snapshot for throttle decisions
            self.track_state.set_recognition_snapshot(
                track.track_id,
                track.best_face_score,
                decision.status
            )

            # Store full match result for finalization to reuse
            self.track_state.set_pending_match_result(track.track_id, match_result)

            # Store memory context for finalization to reuse
            if memory_context:
                self.track_state.set_pending_memory_context(track.track_id, memory_context)

            if settings.DEBUG_RECOGNITION:
                t_policy_ms = round((time.perf_counter() - t_recog) * 1000, 1)
                t_memory_ms = round((t_recog - t_memory) * 1000, 1)
                t_db_ms = round((t_memory - t_db) * 1000, 1)
                t_embed_ms = round((t_db - t_embed) * 1000, 1)
                t_quality_ms = round((t_embed - t_quality) * 1000, 1)
                t_detect_ms = round((t_quality - t_detect) * 1000, 1)
                t_total_ms = round((time.perf_counter() - t_total) * 1000, 1)
                logger.debug("recognition_timing",
                             track_id=track.track_id,
                             crop_detect_ms=crop_detect_ms,
                             fallback_detect_ms=fallback_detect_ms,
                             fallback_used=fallback_used,
                             person_crop_shape=person_crop_shape,
                             faces_on_crop=faces_on_crop,
                             detect_ms=t_detect_ms,
                             quality_ms=t_quality_ms,
                             embed_ms=t_embed_ms,
                             db_ms=t_db_ms,
                             memory_ms=t_memory_ms,
                             recog_policy_ms=t_policy_ms,
                             total_ms=t_total_ms)

        except Exception as e:
            logger.error("progressive_recognition_failed", track_id=track.track_id, error=str(e), exc_info=True)
        finally:
            if settings.DEBUG_RECOGNITION:
                worker_end = time.perf_counter()
                with self._recognition_start_lock:
                    start_time = self._recognition_start_times.pop(track.track_id, None)
                duration_ms = round((worker_end - start_time) * 1000, 1) if start_time is not None else None
                # Check if track still exists in TrackState
                current_track = self.track_state.get(track.track_id)
                track_stale = current_track is not track if current_track else True
                # Verify if decision None is logging artifact or functional bug
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
                # If the track expired while recognition was running, it was skipped by
                # _loop (line 128) because it was in _recognizing_tracks. Now that recognition
                # is done, we must finalize it to ensure an event is always logged.
                if track.track_id not in self._finalized_track_ids and self.track_state.get(track.track_id) is None:
                    self._finalized_track_ids.add(track.track_id)
                    try:
                        self._recognition_executor.submit(self._finalize_track, track)
                    except RuntimeError:
                        logger.debug("finalize_submit_after_shutdown", track_id=track.track_id)

    def _finalize_track(self, track: Track):
        try:
            # Visibility already classified in get_expired_tracks() before removal

            if track.embedding is None:
                app = get_insightface()

                # Run detection once per image instead of cascading 4 times
                crop_faces = []
                if track.best_face_crop is not None:
                    try:
                        crop_faces = app.detect_faces_raw(track.best_face_crop, min_score=settings.DET_SCORE_RELAXED)
                    except Exception as e:
                        logger.debug("best_face_crop_detect_failed", track_id=track.track_id, error=str(e))

                # Skip full frame detection if crop already found a face
                frame_faces = []
                if not crop_faces and track.best_full_frame is not None:
                    try:
                        frame_faces = app.detect_faces_raw(track.best_full_frame, min_score=settings.DET_SCORE_RELAXED)
                    except Exception as e:
                        logger.debug("best_full_frame_detect_failed", track_id=track.track_id, error=str(e))

                # Pick best face: standard threshold first, then relaxed
                best = None
                source = None
                for faces, src in [(crop_faces, "crop"), (frame_faces, "frame")]:
                    if not faces:
                        continue
                    hit = next((f for f in faces if f["det_score"] >= settings.EMBEDDING_DET_SCORE_MIN), None)
                    if hit:
                        best, source = hit, src
                        break
                    # Relaxed fallback
                    if faces[0]["det_score"] >= settings.DET_SCORE_RELAXED:
                        best, source = faces[0], f"relaxed_{src}"
                        break

                if best:
                    accepted = self.track_state.set_embedding(
                        track.track_id,
                        best["embedding"].tolist(),
                        is_masked=best["is_masked"],
                        det_score=best["det_score"],
                    )
                    if accepted:
                        logger.debug("final_embed_done", track_id=track.track_id, source=source, score=best["det_score"])
                    else:
                        logger.debug("embedding_rejected_by_quality_gate",
                                     track_id=track.track_id,
                                     current_det_score=round(track.embedding_det_score, 3),
                                     new_det_score=round(best["det_score"], 3))
                else:
                    logger.info("no_embedding_after_retries",
                              track_id=track.track_id,
                              visibility=track.visibility,
                              frames_seen=track.total_frames_seen,
                              face_detected=track.face_detected_once)

        except Exception as e:
            logger.error("track_finalization_failed", track_id=track.track_id, error=str(e), exc_info=True)
        finally:
            # Always finalize the track so at least an event is logged
            if self.on_track_finalized:
                self.on_track_finalized(track)
