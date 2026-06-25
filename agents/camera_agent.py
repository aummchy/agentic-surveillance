import cv2
import time
import structlog
import threading
import queue
import concurrent.futures
import numpy as np
from config import settings
from pipeline.tracker import track_persons
from pipeline.track_state import TrackState
from pipeline.quality_agent import compute_quality
from pipeline.face import compute_face_ratio
from pipeline.models import Track
from utils.image_utils import crop_person, resize_image, draw_annotations
from utils.embedding_utils import get_insightface
from agents.recognition import RecognitionAgent
from agents.memory import MemoryAgent

logger = structlog.get_logger(__name__)


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

    def start(self):
        self._running = True

        # Validate camera index
        test_cap = cv2.VideoCapture(settings.CAMERA_INDEX)
        if not test_cap.isOpened():
            logger.error("camera_index_invalid", index=settings.CAMERA_INDEX)
            test_cap.release()
            return
        test_cap.release()

        self._cap = cv2.VideoCapture(settings.CAMERA_INDEX)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.FRAME_WIDTH)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.FRAME_HEIGHT)

        if not self._cap.isOpened():
            logger.error("camera_open_failed")
            return

        logger.info("camera_started", index=settings.CAMERA_INDEX)

        try:
            self._loop()
        except KeyboardInterrupt:
            logger.info("Camera stopped by user")
        finally:
            self.stop()

    def stop(self):
        self._running = False
        self._stop_event.set()
        self._recognition_executor.shutdown(wait=False)
        if self._cap:
            self._cap.release()
        logger.info("camera_stopped")

    def _loop(self):
        consecutive_failures = 0
        max_failures = 30  # 3 seconds at 100ms sleep
        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                consecutive_failures += 1
                if consecutive_failures >= max_failures:
                    logger.error("camera_disconnected", failures=consecutive_failures)
                    # Try to reconnect
                    self._cap.release()
                    time.sleep(1.0)
                    self._cap = cv2.VideoCapture(settings.CAMERA_INDEX)
                    if self._cap.isOpened():
                        logger.info("camera_reconnected")
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

            active_ids = set()
            for t in tracks:
                composite_id = self.track_state._make_composite_id(settings.CAMERA_ID, t["track_id"])
                active_ids.add(composite_id)
                track = self.track_state.update(settings.CAMERA_ID, t["track_id"], t["box"])

                if track and self._frame_count % settings.RECOGNITION_INTERVAL_FRAMES == 0:
                    # Skip recognition if already matched with high confidence
                    if track.pending_match_result and track.pending_match_result.similarity_score > 0.85:
                        logger.debug("skip_recognition_already_matched",
                                   track_id=track.track_id,
                                   similarity=track.pending_match_result.similarity_score)
                    else:
                        logger.debug("progressive_recognition_scheduled",
                                   track_id=track.track_id,
                                   frame=self._frame_count)
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
        with self._track_sets_lock:
            self._recognizing_tracks.add(track.track_id)
        try:
            # Skip InsightFace if already matched with high confidence
            if track.pending_match_result and track.pending_match_result.similarity_score > 0.80:
                logger.debug("skip_recognition_high_confidence",
                           track_id=track.track_id,
                           similarity=track.pending_match_result.similarity_score)
                return

            person_crop = crop_person(frame, track.person_box)
            if person_crop.size == 0:
                logger.debug("empty_person_crop", track_id=track.track_id)
                return

            app = get_insightface()

            # Run detection on crop with reasonable threshold first
            crop_faces = app.detect_faces_raw(person_crop, min_score=settings.DET_SCORE_RELAXED)
            frame_faces = [] if crop_faces else app.detect_faces_raw(frame, min_score=settings.DET_SCORE_RELAXED)

            best = None
            detected_in_person_crop = False
            if crop_faces:
                # Try standard threshold on crop
                best = next((f for f in crop_faces if f["det_score"] >= settings.DET_SCORE_MIN), None)
                if best:
                    detected_in_person_crop = True
                    logger.debug("face_found_crop", track_id=track.track_id, score=best["det_score"])
                else:
                    # Try relaxed threshold on crop
                    best = crop_faces[0] if crop_faces[0]["det_score"] >= settings.DET_SCORE_RELAXED else None
                    if best:
                        detected_in_person_crop = True
                        logger.info("face_found_relaxed_crop", track_id=track.track_id, score=best["det_score"])

            if not best and frame_faces:
                # Try standard threshold on full frame
                best = next((f for f in frame_faces if f["det_score"] >= settings.DET_SCORE_MIN), None)
                if best:
                    logger.debug("face_found_in_full_frame", track_id=track.track_id, score=best["det_score"])
                else:
                    # Try relaxed threshold on full frame
                    best = frame_faces[0] if frame_faces[0]["det_score"] >= settings.DET_SCORE_RELAXED else None
                    if best:
                        logger.info("face_found_relaxed_frame", track_id=track.track_id, score=best["det_score"])

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
                quality = type('Q', (), {'is_valid': False, 'overall_score': 0.0})()

            if quality.is_valid:
                self.track_state.set_best_face(
                    track.track_id,
                    face_crop,
                    quality.overall_score,
                    frame,
                    face_ratio
                )

            embedding_list = best["embedding"].tolist()
            self.track_state.set_embedding(
                track.track_id,
                embedding_list,
                best["is_masked"]
            )

            from agents.matching_agent import run_matching_from_embedding
            match_result = run_matching_from_embedding(embedding_list)

            # Phase 2.2: Use Memory Agent for context (skip for high-confidence matches)
            memory_context = {}
            if match_result.matched and match_result.person_id:
                if match_result.similarity_score > 0.80:
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

            # Phase 2.1: Use Recognition Agent with memory context
            track_duration = time.time() - track.first_seen
            recognition_result = self.recognition_agent.run({
                "similarity": match_result.similarity_score if match_result.matched else 0.0,
                "is_masked": best["is_masked"],
                "face_quality": quality.overall_score if hasattr(quality, 'overall_score') else 0.0,
                "track_duration": track_duration,
                "memory_context": memory_context,
            })

            # Store recognition result in track for later use
            track.pending_recognition = recognition_result

            # Phase 2.3: Use Policy Agent with all context
            from agents.decision_agent import decide
            decision = decide(track, match_result, recognition_result, memory_context)

            # Store person name on track for bounding box display
            if match_result.matched and match_result.name:
                self.track_state.set_person_name(track.track_id, match_result.name)

            # Only dispatch CRITICAL alerts (blacklist) during progressive recognition.
            # All other alerts are deferred to finalization to avoid premature alerts
            # for verified/known users when early recognition attempts produce low similarity.
            if decision.should_alert and decision.alert_level == "critical" and not track.alerted:
                with self._track_sets_lock:
                    already_finalized = track.track_id in self._finalized_track_ids
                if not already_finalized:
                    from agents.alert_agent import dispatch
                    from utils.image_utils import upload_to_cloudinary, save_image
                    image_url = track.image_url
                    if not image_url and track.best_full_frame is not None:
                        save_image(track.best_full_frame, f"captures/{track.track_id}.jpg")
                        image_url = upload_to_cloudinary(track.best_full_frame)
                        if not image_url:
                            image_url = f"captures/{track.track_id}.jpg"
                        track.image_url = image_url
                    dispatch(track, decision, image_url)
                    self.track_state.set_decision(track.track_id, decision.status, True)
                    logger.info("progressive_critical_alert",
                                track_id=track.track_id,
                                alert_level=decision.alert_level,
                                status=decision.status)
            else:
                self.track_state.set_decision(track.track_id, decision.status, track.alerted)
                if decision.should_alert:
                    logger.debug("progressive_alert_deferred_to_finalization",
                                 track_id=track.track_id,
                                 alert_level=decision.alert_level,
                                 status=decision.status)

            self.track_state.set_pending_recognition(
                track.track_id,
                self._frame_count,
                embedding_list,
                {"matched": match_result.matched, "person_id": match_result.person_id}
            )

            # Store full match result for finalization to reuse
            self.track_state.set_pending_match_result(track.track_id, match_result)

            # Store memory context for finalization to reuse
            if memory_context:
                self.track_state.set_pending_memory_context(track.track_id, memory_context)

        except Exception as e:
            logger.error("progressive_recognition_failed", track_id=track.track_id, error=str(e), exc_info=True)
        finally:
            with self._track_sets_lock:
                self._recognizing_tracks.discard(track.track_id)

    def _finalize_track(self, track: Track):
        try:
            # Visibility already classified in get_expired_tracks() before removal

            if track.embedding is None:
                app = get_insightface()

                # Run detection once per image instead of cascading 4 times
                crop_faces = []
                if track.best_face_crop is not None:
                    try:
                        resized = resize_image(track.best_face_crop)
                        crop_faces = app.detect_faces_raw(resized, min_score=settings.DET_SCORE_RELAXED)
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
                    track.embedding = best["embedding"].tolist()
                    track.is_masked = best["is_masked"]
                    logger.debug("final_embed_done", track_id=track.track_id, source=source, score=best["det_score"])
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
