import threading
import time
import cv2
import numpy as np
import structlog
from typing import Dict, Optional
from config import settings
from pipeline.models import Track

logger = structlog.get_logger(__name__)


class TrackState:
    def __init__(self):
        self._tracks: Dict[str, Track] = {}
        self._lock = threading.Lock()
        self._session_epoch = int(time.time())
        self._in_flight: Dict[str, int] = {}  # track_id → active recognition count

    @staticmethod
    def _compute_iou(box1: tuple, box2: tuple) -> float:
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        if inter == 0:
            return 0.0
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0.0

    def make_composite_id(self, camera_id: str, byte_track_id: int) -> str:
        return f"{camera_id}_{self._session_epoch}_{byte_track_id}"

    def update(self, camera_id: str, track_id: int, box: tuple,
               frame: np.ndarray = None) -> Optional[Track]:
        composite_id = self.make_composite_id(camera_id, track_id)

        with self._lock:
            if composite_id in self._tracks:
                track = self._tracks[composite_id]
                track.last_seen = time.time()
                track.person_box = box
                track.total_frames_seen += 1
                return track
            else:
                track = Track(
                    track_id=composite_id,
                    first_seen=time.time(),
                    last_seen=time.time(),
                    person_box=box,
                    max_track_secs=settings.MAX_TRACK_SECS
                )
                self._tracks[composite_id] = track
                if settings.DEBUG_RECOGNITION:
                    # Find the nearest existing track by IoU for fragmentation detection
                    nearest_id = None
                    nearest_iou = 0.0
                    gap_ms = None
                    now = time.time()
                    for cid, existing in self._tracks.items():
                        if cid == composite_id:
                            continue
                        iou_val = self._compute_iou(box, existing.person_box)
                        if iou_val > nearest_iou:
                            nearest_iou = iou_val
                            nearest_id = existing.track_id
                            gap_ms = round((now - existing.last_seen) * 1000, 1)
                    log_fields = dict(
                        track_id=composite_id,
                        active_count=len(self._tracks),
                        bt_track_id=track_id)
                    if nearest_id and nearest_iou > 0.01:
                        log_fields["nearest_track_id"] = nearest_id
                        log_fields["nearest_iou"] = round(nearest_iou, 3)
                        log_fields["time_gap_ms"] = gap_ms
                    logger.debug("track_created", **log_fields)
                return track

    def get(self, composite_id: str) -> Optional[Track]:
        with self._lock:
            return self._tracks.get(composite_id)

    def get_all(self) -> list:
        with self._lock:
            return list(self._tracks.values())

    def debug_snapshot(self) -> list[dict]:
        with self._lock:
            return [
                {"id": t.track_id,
                 "last_seen": t.last_seen,
                 "age_secs": round(time.time() - t.last_seen, 2),
                 "expired_reported": t.expired_reported,
                 "visibility": t.visibility,
                 "person_box": t.person_box,
                 "total_frames": t.total_frames_seen,
                 "face_detected": t.face_detected_once,
                 "decision": t.decision,
                 "is_masked": t.is_masked}
                for t in self._tracks.values()
            ]

    def remove(self, composite_id: str) -> Optional[Track]:
        with self._lock:
            return self._tracks.pop(composite_id, None)

    def get_expired_tracks(self) -> list:
        expired = []
        with self._lock:
            to_remove = []
            for cid, track in self._tracks.items():
                if track.is_expired(settings.TRACK_TIMEOUT_SECS) or track.is_max_lifetime_exceeded():
                    if track.expired_reported:
                        continue
                    self._classify_visibility_inplace(track)
                    expired.append(track)
                    track.expired_reported = True
                    # Defer actual removal while a recognition thread is still
                    # writing to the track (set_best_face, set_embedding, etc.).
                    if self._in_flight.get(cid, 0) == 0:
                        to_remove.append(cid)
            for cid in to_remove:
                if settings.DEBUG_RECOGNITION:
                    track_obj = self._tracks.get(cid)
                    lifetime = round(time.time() - track_obj.first_seen, 1) if track_obj else None
                    logger.debug("track_removed",
                                 track_id=cid,
                                 reason="expired",
                                 active_count=len(self._tracks) - 1,
                                 lifetime_secs=lifetime,
                                 total_frames=track_obj.total_frames_seen if track_obj else None,
                                 face_detected=track_obj.face_detected_once if track_obj else None)
                del self._tracks[cid]
        return expired

    def _classify_visibility_inplace(self, track: Track):
        """Classify visibility directly on the track object (no dict lookup)."""
        if track.max_face_ratio >= settings.VISIBLE_FACE_RATIO:
            track.visibility = "visible"
        elif track.max_face_ratio >= settings.PARTIAL_FACE_RATIO:
            track.visibility = "partial"
        elif track.is_masked or track.face_detected_once:
            track.visibility = "partial"
        elif not track.face_detected_once and track.total_frames_seen >= settings.MIN_TRACK_FRAMES:
            track.visibility = "hidden"
        else:
            track.visibility = "unknown"

    def update_face_visibility(self, composite_id: str, face_detected: bool, face_ratio: float):
        with self._lock:
            track = self._tracks.get(composite_id)
            if track:
                if face_detected:
                    track.frames_with_detectable_face += 1
                    track.face_detected_once = True
                if face_ratio > track.max_face_ratio:
                    track.max_face_ratio = face_ratio

    def begin_recognition(self, composite_id: str):
        """Mark a track as having an in-flight recognition task."""
        with self._lock:
            self._in_flight[composite_id] = self._in_flight.get(composite_id, 0) + 1

    def end_recognition(self, composite_id: str):
        """Clear one in-flight recognition reference for a track.

        If the track has expired and no other recognition threads hold a
        reference, it is removed from _tracks here so finalization can proceed.
        """
        with self._lock:
            count = self._in_flight.get(composite_id, 0)
            if count <= 1:
                self._in_flight.pop(composite_id, None)
                # If the track expired while recognition was running it was
                # reported by get_expired_tracks() but NOT removed because
                # _in_flight > 0.  Now that we are the last reference, clean
                # it up so the next get_expired_tracks() call (or the
                # progressive-recognition finally block) can finalize it.
                track = self._tracks.get(composite_id)
                if track and (track.is_expired(settings.TRACK_TIMEOUT_SECS)
                              or track.is_max_lifetime_exceeded()):
                    if settings.DEBUG_RECOGNITION:
                        lifetime = round(time.time() - track.first_seen, 1)
                        logger.debug("track_removed",
                                     track_id=composite_id,
                                     reason="expired_after_recognition",
                                     active_count=len(self._tracks) - 1,
                                     lifetime_secs=lifetime,
                                     total_frames=track.total_frames_seen,
                                     face_detected=track.face_detected_once)
                    del self._tracks[composite_id]
            else:
                self._in_flight[composite_id] = count - 1

    def set_best_face(self, composite_id: str, face_crop: np.ndarray,
                      face_score: float, full_frame: np.ndarray, face_ratio: float):
        with self._lock:
            track = self._tracks.get(composite_id)
            if not track or face_score <= track.best_face_score + 0.03:
                return

        # Encode JPEG outside the lock (expensive operation) — only if score improved significantly
        success, jpeg_buf = cv2.imencode(".jpg", full_frame, [cv2.IMWRITE_JPEG_QUALITY, settings.JPEG_QUALITY_STORE])
        if not success:
            logger.warning("jpeg_encode_failed", composite_id=composite_id, context="set_best_face")
            return
        jpeg_bytes = jpeg_buf.tobytes()

        with self._lock:
            track = self._tracks.get(composite_id)
            if track and face_score > track.best_face_score:
                track.best_face_crop = face_crop
                track.best_face_score = face_score
                track.best_full_frame = full_frame
                track.best_face_ratio = face_ratio
                track.best_frame_jpeg = jpeg_bytes

    def ensure_fallback_frame(self, composite_id: str, frame: np.ndarray):
        """Guarantee every track gets at least one photo, independent of face quality.

        Called unconditionally on the first recognition pass. Uses double-checked
        locking: cheap read under lock → expensive JPEG encode outside lock →
        re-check and write under lock.
        """
        with self._lock:
            track = self._tracks.get(composite_id)
            if not track or track.fallback_frame_jpeg is not None:
                return

        success, jpeg_buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, settings.JPEG_QUALITY_STORE])
        if not success:
            logger.warning("jpeg_encode_failed", composite_id=composite_id, context="ensure_fallback_frame")
            return
        jpeg_bytes = jpeg_buf.tobytes()

        with self._lock:
            track = self._tracks.get(composite_id)
            if track and track.fallback_frame_jpeg is None:
                track.fallback_frame_jpeg = jpeg_bytes

    def set_embedding(self, composite_id: str, embedding: list, is_masked: bool = False, det_score: float = 0.0) -> tuple[bool, str]:
        with self._lock:
            track = self._tracks.get(composite_id)
            if track:
                if track.embedding is None or det_score > track.embedding_det_score + 0.05:
                    track.embedding = embedding
                    track.is_masked = is_masked
                    track.embedding_det_score = det_score
                    return (True, "ok")
                return (False, "rejected_quality")
            return (False, "track_removed")

    def set_decision(self, composite_id: str, decision: str, alerted: bool = False):
        with self._lock:
            track = self._tracks.get(composite_id)
            if track:
                track.decision = decision
                track.alerted = alerted

    def set_person_name(self, composite_id: str, name: str):
        with self._lock:
            track = self._tracks.get(composite_id)
            if track:
                track.person_name = name

    def set_face_crop_path(self, composite_id: str, path: str):
        with self._lock:
            track = self._tracks.get(composite_id)
            if track:
                track.best_face_crop_path = path

    def set_pending_match_result(self, composite_id: str, match_result):
        with self._lock:
            track = self._tracks.get(composite_id)
            if track:
                track.pending_match_result = match_result

    def set_pending_memory_context(self, composite_id: str, memory_context: dict):
        with self._lock:
            track = self._tracks.get(composite_id)
            if track:
                track.pending_memory_context = memory_context

    def set_cached_embedding(self, composite_id: str, embedding: list):
        with self._lock:
            track = self._tracks.get(composite_id)
            if track:
                track.cached_embedding = embedding

    def set_pending_recognition_data(self, composite_id: str, recognition_result: dict):
        with self._lock:
            track = self._tracks.get(composite_id)
            if track:
                track.pending_recognition = recognition_result

    def set_recognition_snapshot(self, composite_id: str, quality: float, status: str):
        with self._lock:
            track = self._tracks.get(composite_id)
            if track:
                track.last_recognition_quality = quality
                track.last_recognition_status = status
                track.last_recognition_time = time.time()
