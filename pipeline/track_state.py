import threading
import time
import cv2
import numpy as np
from typing import Dict, Optional, Callable
from config import settings
from pipeline.models import Track


class TrackState:
    def __init__(self):
        self._tracks: Dict[str, Track] = {}
        self._lock = threading.Lock()
        self._session_epoch = int(time.time())
        self._in_flight: Dict[str, int] = {}  # track_id → active recognition count

    def _make_composite_id(self, camera_id: str, byte_track_id: int) -> str:
        return f"{camera_id}_{self._session_epoch}_{byte_track_id}"

    def update(self, camera_id: str, track_id: int, box: tuple,
               frame: np.ndarray = None) -> Optional[Track]:
        composite_id = self._make_composite_id(camera_id, track_id)

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
                return track

    def get(self, composite_id: str) -> Optional[Track]:
        with self._lock:
            return self._tracks.get(composite_id)

    def get_all(self) -> list:
        with self._lock:
            return list(self._tracks.values())

    def remove(self, composite_id: str) -> Optional[Track]:
        with self._lock:
            return self._tracks.pop(composite_id, None)

    def get_expired_tracks(self) -> list:
        expired = []
        with self._lock:
            to_remove = []
            for cid, track in self._tracks.items():
                if track.is_expired(settings.TRACK_TIMEOUT_SECS) or track.is_max_lifetime_exceeded():
                    self._classify_visibility_inplace(track)
                    expired.append(track)
                    # Defer actual removal while a recognition thread is still
                    # writing to the track (set_best_face, set_embedding, etc.).
                    if self._in_flight.get(cid, 0) == 0:
                        to_remove.append(cid)
            for cid in to_remove:
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
                    del self._tracks[composite_id]
            else:
                self._in_flight[composite_id] = count - 1

    def set_best_face(self, composite_id: str, face_crop: np.ndarray,
                      face_score: float, full_frame: np.ndarray, face_ratio: float):
        with self._lock:
            track = self._tracks.get(composite_id)
            if not track or face_score <= track.best_face_score + 0.1:
                return

        # Encode JPEG outside the lock (expensive operation) — only if score improved significantly
        _, jpeg_buf = cv2.imencode(".jpg", full_frame, [cv2.IMWRITE_JPEG_QUALITY, settings.JPEG_QUALITY_STORE])
        jpeg_bytes = jpeg_buf.tobytes()

        with self._lock:
            track = self._tracks.get(composite_id)
            if track and face_score > track.best_face_score:
                track.best_face_crop = face_crop
                track.best_face_score = face_score
                track.best_full_frame = full_frame
                track.best_face_ratio = face_ratio
                track.best_frame_jpeg = jpeg_bytes

    def set_embedding(self, composite_id: str, embedding: list, is_masked: bool = False):
        with self._lock:
            track = self._tracks.get(composite_id)
            if track:
                track.embedding = embedding
                track.is_masked = is_masked

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

    def set_pending_recognition(self, composite_id: str, frame_num: int,
                                embedding: list = None, match: dict = None):
        with self._lock:
            track = self._tracks.get(composite_id)
            if track:
                track.last_recognition_frame = frame_num
                if embedding:
                    track.pending_embedding = embedding
                if match:
                    track.pending_match = match

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
