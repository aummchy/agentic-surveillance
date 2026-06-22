import cv2
import time
import structlog
import threading
import queue
import numpy as np
from config import settings
from pipeline.tracker import track_persons
from pipeline.track_state import TrackState
from pipeline.quality_agent import compute_quality
from pipeline.face import detect_and_embed, compute_face_ratio
from pipeline.models import Track
from utils.image_utils import crop_person, crop_face_region, resize_image, draw_annotations
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

    def start(self):
        self._running = True
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
        if self._cap:
            self._cap.release()
        logger.info("camera_stopped")

    def _loop(self):
        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                logger.warning("frame_read_failed")
                time.sleep(0.1)
                continue

            self._frame_count += 1

            tracks = track_persons(frame)

            active_ids = set()
            for t in tracks:
                composite_id = self.track_state._make_composite_id(settings.CAMERA_ID, t["track_id"])
                active_ids.add(composite_id)
                track = self.track_state.update(settings.CAMERA_ID, t["track_id"], t["box"])

                if track and self._frame_count % settings.RECOGNITION_INTERVAL_FRAMES == 0:
                    logger.debug("progressive_recognition",
                               track_id=track.track_id,
                               frame=self._frame_count)
                    self._progressive_recognition(frame, track)

            expired = self.track_state.get_expired_tracks()
            for track in expired:
                self._finalize_track(track)

            annotated = draw_annotations(frame, self.track_state.get_all())
            if self.on_frame_annotated:
                self.on_frame_annotated(annotated)

    def _progressive_recognition(self, frame: np.ndarray, track: Track):
        try:
            person_crop = crop_person(frame, track.person_box)
            if person_crop.size == 0:
                logger.debug("empty_person_crop", track_id=track.track_id)
                return

            app = get_insightface()

            embedding_result = app.detect_and_embed(person_crop)

            if not embedding_result.face_detected or embedding_result.embedding is None:
                logger.debug("no_face_in_crop",
                           track_id=track.track_id,
                           error=embedding_result.error)
                embedding_result = app.detect_and_embed(frame)

                if embedding_result.face_detected and embedding_result.embedding is not None:
                    logger.debug("face_found_in_full_frame", track_id=track.track_id)
                else:
                    logger.debug("no_face_in_full_frame",
                               track_id=track.track_id,
                               error=embedding_result.error)
                    embedding_result = app.detect_and_embed_relaxed(person_crop, min_score=0.20)
                    if embedding_result.face_detected and embedding_result.embedding is not None:
                        logger.info("face_found_relaxed_crop", track_id=track.track_id)
                    else:
                        embedding_result = app.detect_and_embed_relaxed(frame, min_score=0.20)
                        if embedding_result.face_detected and embedding_result.embedding is not None:
                            logger.info("face_found_relaxed_frame", track_id=track.track_id)
                        else:
                            logger.debug("no_face_relaxed", track_id=track.track_id)
                            return

            if embedding_result.embedding is None:
                return

            face_ratio = 0.0
            if embedding_result.bbox:
                face_ratio = compute_face_ratio(embedding_result.bbox, track.person_box)

            self.track_state.update_face_visibility(track.track_id, True, face_ratio)

            if embedding_result.embedding_score < settings.EMBEDDING_DET_SCORE_MIN:
                logger.debug("embedding_score_below_threshold",
                           track_id=track.track_id,
                           score=embedding_result.embedding_score,
                           threshold=settings.EMBEDDING_DET_SCORE_MIN)
                return

            if embedding_result.bbox:
                fx1, fy1, fx2, fy2 = embedding_result.bbox
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

            embedding_list = embedding_result.embedding.tolist()
            self.track_state.set_embedding(
                track.track_id,
                embedding_list,
                embedding_result.is_masked
            )

            from agents.matching_agent import run_matching_from_embedding
            match_result = run_matching_from_embedding(embedding_list)

            # Phase 2.2: Use Memory Agent for context
            memory_context = {}
            if match_result.matched and match_result.person_id:
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
                "is_masked": embedding_result.is_masked,
                "face_quality": quality.overall_score if hasattr(quality, 'overall_score') else 0.0,
                "track_duration": track_duration,
                "memory_context": memory_context,
            })

            # Store recognition result in track for later use
            track.pending_recognition = recognition_result

            # Phase 2.3: Use Policy Agent with all context
            from agents.decision_agent import decide
            decision = decide(track, match_result, recognition_result, memory_context)

            if decision.should_alert and not track.alerted:
                from agents.alert_agent import dispatch
                dispatch(track, decision)
                self.track_state.set_decision(track.track_id, decision.status, True)
            else:
                self.track_state.set_decision(track.track_id, decision.status, False)

            self.track_state.set_pending_recognition(
                track.track_id,
                self._frame_count,
                embedding_list,
                {"matched": match_result.matched, "person_id": match_result.person_id}
            )

        except Exception as e:
            logger.error("progressive_recognition_failed", track_id=track.track_id, error=str(e))

    def _finalize_track(self, track: Track):
        try:
            self.track_state.classify_visibility(track.track_id)

            if track.embedding is None:
                app = get_insightface()

                if track.best_face_crop is not None:
                    try:
                        resized = resize_image(track.best_face_crop)
                        embedding_result = app.detect_and_embed(resized)
                        if embedding_result.face_detected and embedding_result.embedding is not None:
                            track.embedding = embedding_result.embedding.tolist()
                            track.is_masked = embedding_result.is_masked
                            logger.debug("final_embed_best_face_crop", track_id=track.track_id)
                    except Exception as e:
                        logger.debug("best_face_crop_embed_failed", track_id=track.track_id, error=str(e))

                if track.embedding is None and track.best_full_frame is not None:
                    try:
                        embedding_result = app.detect_and_embed(track.best_full_frame)
                        if embedding_result.face_detected and embedding_result.embedding is not None:
                            track.embedding = embedding_result.embedding.tolist()
                            track.is_masked = embedding_result.is_masked
                            logger.debug("final_embed_best_full_frame", track_id=track.track_id)
                    except Exception as e:
                        logger.debug("best_full_frame_embed_failed", track_id=track.track_id, error=str(e))

                if track.embedding is None and track.best_face_crop is not None:
                    try:
                        resized = resize_image(track.best_face_crop)
                        embedding_result = app.detect_and_embed_relaxed(resized, min_score=0.15)
                        if embedding_result.face_detected and embedding_result.embedding is not None:
                            track.embedding = embedding_result.embedding.tolist()
                            track.is_masked = embedding_result.is_masked
                            logger.info("final_embed_relaxed_crop", track_id=track.track_id)
                    except Exception as e:
                        logger.debug("relaxed_crop_embed_failed", track_id=track.track_id, error=str(e))

                if track.embedding is None and track.best_full_frame is not None:
                    try:
                        embedding_result = app.detect_and_embed_relaxed(track.best_full_frame, min_score=0.15)
                        if embedding_result.face_detected and embedding_result.embedding is not None:
                            track.embedding = embedding_result.embedding.tolist()
                            track.is_masked = embedding_result.is_masked
                            logger.info("final_embed_relaxed_frame", track_id=track.track_id)
                    except Exception as e:
                        logger.debug("relaxed_frame_embed_failed", track_id=track.track_id, error=str(e))

                if track.embedding is None:
                    logger.info("no_embedding_after_retries",
                              track_id=track.track_id,
                              visibility=track.visibility,
                              frames_seen=track.total_frames_seen,
                              face_detected=track.face_detected_once)

            if self.on_track_finalized:
                self.on_track_finalized(track)

        except Exception as e:
            logger.error("track_finalization_failed", track_id=track.track_id, error=str(e))
