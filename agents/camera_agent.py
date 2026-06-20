import cv2
import time
import logging
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

logger = logging.getLogger(__name__)


class CameraAgent:
    def __init__(self, on_track_finalized=None, on_frame_annotated=None):
        self.track_state = TrackState()
        self.on_track_finalized = on_track_finalized
        self.on_frame_annotated = on_frame_annotated
        self._running = False
        self._cap = None
        self._frame_count = 0
        self._stop_event = threading.Event()

    def start(self):
        self._running = True
        self._cap = cv2.VideoCapture(settings.CAMERA_INDEX)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.FRAME_WIDTH)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.FRAME_HEIGHT)

        if not self._cap.isOpened():
            logger.error("Failed to open camera")
            return

        logger.info(f"Camera started: index={settings.CAMERA_INDEX}")

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
        logger.info("Camera stopped")

    def _loop(self):
        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                logger.warning("Failed to read frame, retrying...")
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
                return

            app = get_insightface()
            embedding_result = app.detect_and_embed(person_crop)

            if embedding_result.face_detected and embedding_result.embedding is not None:
                face_ratio = 0.0
                if embedding_result.bbox:
                    face_ratio = compute_face_ratio(embedding_result.bbox, track.person_box)

                self.track_state.update_face_visibility(track.track_id, True, face_ratio)

                if embedding_result.embedding_score >= settings.EMBEDDING_DET_SCORE_MIN:
                    quality = compute_quality(person_crop)
                    if quality.is_valid:
                        self.track_state.set_best_face(
                            track.track_id,
                            person_crop,
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

                        from agents.decision_agent import decide
                        decision = decide(track, match_result)

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
            logger.error(f"Progressive recognition failed for {track.track_id}: {e}")

    def _finalize_track(self, track: Track):
        try:
            self.track_state.classify_visibility(track.track_id)

            if track.embedding is None and track.best_face_crop is not None:
                try:
                    app = get_insightface()
                    resized = resize_image(track.best_face_crop)
                    embedding_result = app.detect_and_embed(resized)

                    if embedding_result.face_detected and embedding_result.embedding is not None:
                        track.embedding = embedding_result.embedding.tolist()
                        track.is_masked = embedding_result.is_masked
                except Exception as e:
                    logger.error(f"Final embedding generation failed: {e}")

            if self.on_track_finalized:
                self.on_track_finalized(track)

        except Exception as e:
            logger.error(f"Track finalization failed for {track.track_id}: {e}")
