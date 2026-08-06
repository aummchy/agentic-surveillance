import threading
import time
import cv2
import numpy as np
import structlog
from typing import Dict, List, Optional, Tuple, Any
from config import settings
from pipeline.models import Track
from config.status import Visibility
from utils.image_utils import compute_iou

logger = structlog.get_logger(__name__)


class TrackState:
    """Thread-safe state manager for active person tracks.

    Coordinates between the camera thread (update, get_expired_tracks) and
    recognition workers (set_best_face, set_embedding, etc.) using per-track
    locks and an in-flight reference counter to prevent premature removal.
    """

    def __init__(self) -> None:
        self._tracks: Dict[str, Track] = {}
        self._lock = threading.Lock()
        self._session_epoch = int(time.time())
        self._in_flight: Dict[str, int] = {}  # track_id → active recognition count
        self._bt_active: Dict[int, int] = {}   # bt_id → current active generation
        self._bt_next_gen: Dict[int, int] = {} # bt_id → next generation to allocate

    def allocate_generation(self, bt_id: int) -> int:
        """Allocate a generation for a new track with the given bt_id.

        Returns 0 if first time, or the next generation after the previous
        one was released.  Must be called under self._lock.
        """
        if bt_id in self._bt_active:
            logger.warning("bytetrack_id_already_active",
                           bt_id=bt_id, gen=self._bt_active[bt_id])
            return self._bt_active[bt_id]
        gen = self._bt_next_gen.get(bt_id, 0)
        self._bt_active[bt_id] = gen
        return gen

    def release_generation(self, bt_id: int) -> None:
        """Release the generation for a removed track.

        Increments the next generation for this bt_id.
        Must be called under self._lock.
        """
        gen = self._bt_active.pop(bt_id, None)
        if gen is not None:
            self._bt_next_gen[bt_id] = gen + 1

    @staticmethod
    def make_composite_id(camera_id: str, bt_id: int, generation: int, session_epoch: int) -> str:
        """Pure formatter — no mutable state."""
        return f"{camera_id}_{session_epoch}_{bt_id}_{generation}"

    def update(self, camera_id: str, track_id: int, box: tuple,
               frame: np.ndarray = None) -> Optional[Track]:
        """Update an existing track or create a new one.

        If a track with the given byte_track_id already exists, update its
        bounding box and last_seen time. Otherwise, allocate a new generation
        and create a fresh Track object.

        Returns the updated or newly created Track, or None if creation failed.
        """
        with self._lock:
            # Check if this bt_id already has an active track
            for cid, existing in self._tracks.items():
                if existing.byte_track_id == track_id:
                    with existing._lock:
                        existing.last_seen = time.time()
                        existing.person_box = box
                        existing.total_frames_seen += 1
                    return existing

            # New track — allocate generation
            generation = self.allocate_generation(track_id)
            composite_id = self.make_composite_id(camera_id, track_id, generation, self._session_epoch)

            if generation > 0:
                logger.info("bytetrack_id_reused",
                            bt_id=track_id, generation=generation,
                            composite_id=composite_id)

            track = Track(
                track_id=composite_id,
                first_seen=time.time(),
                last_seen=time.time(),
                person_box=box,
                byte_track_id=track_id,
                generation=generation,
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
                    iou_val = compute_iou(box, existing.person_box)
                    if iou_val > nearest_iou:
                        nearest_iou = iou_val
                        nearest_id = existing.track_id
                        gap_ms = round((now - existing.last_seen) * 1000, 1)
                log_fields = dict(
                    track_id=composite_id,
                    active_count=len(self._tracks),
                    bt_track_id=track_id,
                    generation=generation)
                if nearest_id and nearest_iou > 0.01:
                    log_fields["nearest_track_id"] = nearest_id
                    log_fields["nearest_iou"] = round(nearest_iou, 3)
                    log_fields["time_gap_ms"] = gap_ms
                logger.debug("track_created", **log_fields)
            return track

    def get(self, composite_id: str) -> Optional[Track]:
        """Retrieve a track by its composite ID.

        Returns the Track object if found, None otherwise.
        """
        with self._lock:
            return self._tracks.get(composite_id)

    def get_all(self) -> List[Track]:
        """Return a snapshot of all active tracks.

        Returns a new list (safe to iterate while other threads modify _tracks).
        """
        with self._lock:
            return list(self._tracks.values())

    def debug_snapshot(self) -> List[dict]:
        """Return a debug-friendly snapshot of all tracks.

        Each dict contains key track properties for logging and diagnostics.
        """
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
        """Remove a track by its composite ID.

        Releases the byte_track_id generation for reuse.
        Returns the removed Track, or None if not found.
        """
        with self._lock:
            track = self._tracks.pop(composite_id, None)
            if track:
                self.release_generation(track.byte_track_id)
            return track

    def get_expired_tracks(self) -> List[Track]:
        """Identify and return tracks that have exceeded their timeout.

        Tracks with in-flight recognition tasks are reported but NOT removed
        until end_recognition() clears the last reference.
        """
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
                track_obj = self._tracks.get(cid)
                if settings.DEBUG_RECOGNITION and track_obj:
                    lifetime = round(time.time() - track_obj.first_seen, 1)
                    logger.debug("track_removed",
                                 track_id=cid,
                                 reason="expired",
                                 active_count=len(self._tracks) - 1,
                                 lifetime_secs=lifetime,
                                 total_frames=track_obj.total_frames_seen,
                                 face_detected=track_obj.face_detected_once)
                if track_obj:
                    self.release_generation(track_obj.byte_track_id)
                del self._tracks[cid]
        return expired

    def _classify_visibility_inplace(self, track: Track) -> None:
        """Classify visibility directly on the track object (no dict lookup)."""
        if track.max_face_ratio >= settings.VISIBLE_FACE_RATIO:
            track.visibility = Visibility.VISIBLE
        elif track.max_face_ratio >= settings.PARTIAL_FACE_RATIO:
            track.visibility = Visibility.PARTIAL
        elif track.is_masked or track.face_detected_once:
            track.visibility = Visibility.PARTIAL
        elif not track.face_detected_once and track.total_frames_seen >= settings.MIN_TRACK_FRAMES:
            track.visibility = Visibility.HIDDEN
        else:
            track.visibility = Visibility.UNKNOWN

    def update_face_visibility(self, composite_id: str, face_detected: bool, face_ratio: float) -> None:
        """Update face detection stats for a track.

        Increments face detection count and updates max_face_ratio.
        """
        with self._lock:
            track = self._tracks.get(composite_id)
            if track:
                if face_detected:
                    track.frames_with_detectable_face += 1
                    track.face_detected_once = True
                if face_ratio > track.max_face_ratio:
                    track.max_face_ratio = face_ratio

    def begin_recognition(self, composite_id: str) -> None:
        """Mark a track as having an in-flight recognition task."""
        with self._lock:
            self._in_flight[composite_id] = self._in_flight.get(composite_id, 0) + 1

    def end_recognition(self, composite_id: str) -> None:
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
                    self.release_generation(track.byte_track_id)
                    del self._tracks[composite_id]
            else:
                self._in_flight[composite_id] = count - 1

    def set_best_face(self, track: 'Track', face_crop: np.ndarray,
                      face_score: float, full_frame: np.ndarray, face_ratio: float,
                      person_crop: np.ndarray = None) -> None:
        """Update the best face data for a track.

        Accepts the Track object directly (not composite_id lookup) so that
        data is written even if end_recognition() removed the track from
        _tracks before this call.  The caller always holds a valid reference.
        """
        composite_id = track.track_id
        with track._lock:
            if face_score <= track.best_face_score + settings.FACE_SCORE_IMPROVEMENT_MIN:
                return

        # Encode JPEGs outside the lock (expensive operations) — only if score improved significantly
        success, jpeg_buf = cv2.imencode(".jpg", full_frame, [cv2.IMWRITE_JPEG_QUALITY, settings.JPEG_QUALITY_STORE])
        if not success:
            logger.warning("jpeg_encode_failed", composite_id=composite_id, context="set_best_face")
            return
        jpeg_bytes = jpeg_buf.tobytes()

        person_crop_jpeg = None
        if person_crop is not None and person_crop.size > 0:
            ok, pc_buf = cv2.imencode(".jpg", person_crop, [cv2.IMWRITE_JPEG_QUALITY, settings.JPEG_QUALITY_STORE])
            if ok:
                person_crop_jpeg = pc_buf.tobytes()

        with track._lock:
            if face_score > track.best_face_score:
                track.best_face_crop = face_crop.copy()
                track.best_face_score = face_score
                track.best_full_frame = full_frame
                track.best_face_ratio = face_ratio
                track.best_frame_jpeg = jpeg_bytes
                if person_crop_jpeg is not None:
                    track.best_person_crop_jpeg = person_crop_jpeg

    def ensure_fallback_frame(self, track: 'Track', frame: np.ndarray) -> None:
        """Guarantee every track gets at least one photo, independent of face quality.

        Called unconditionally on the first recognition pass. Uses double-checked
        locking: cheap read under lock → expensive JPEG encode outside lock →
        re-check and write under lock.  Accepts Track directly so that the
        fallback is saved even if end_recognition() removed the track from
        _tracks before this call.
        """
        with track._lock:
            if track.fallback_frame_jpeg is not None:
                return

        success, jpeg_buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, settings.JPEG_QUALITY_STORE])
        if not success:
            logger.warning("jpeg_encode_failed", composite_id=track.track_id, context="ensure_fallback_frame")
            return
        jpeg_bytes = jpeg_buf.tobytes()

        with track._lock:
            if track.fallback_frame_jpeg is None:
                track.fallback_frame_jpeg = jpeg_bytes

    def set_embedding(self, composite_id: str, embedding: list, is_masked: bool = False, det_score: float = 0.0) -> tuple[bool, str]:
        """Store an embedding for a track if quality is sufficient.

        Only overwrites if det_score exceeds the existing score by the
        configured improvement threshold. Returns (success, reason).
        """
        with self._lock:
            track = self._tracks.get(composite_id)
            if track:
                with track._lock:
                    if track.embedding is None or det_score > track.embedding_det_score + settings.EMBEDDING_DET_SCORE_IMPROVEMENT_MIN:
                        track.embedding = embedding
                        track.is_masked = is_masked
                        track.embedding_det_score = det_score
                        return (True, "ok")
                    return (False, "rejected_quality")
            return (False, "track_removed")

    def set_decision(self, composite_id: str, decision: int) -> None:
        """Set the decision status for a track.

        Never overwrites alerted=True; use mark_alerted_once() for that.
        """
        with self._lock:
            track = self._tracks.get(composite_id)
            if track:
                with track._lock:
                    track.decision = decision
                    # Never overwrite alerted=True — mark_alerted_once() is the sole writer

    def set_person_name(self, composite_id: str, name: str, similarity: float = 0.0) -> None:
        """Set the identified person name for a track.

        Only upgrades: new similarity must be >= existing similarity.
        """
        with self._lock:
            track = self._tracks.get(composite_id)
            if track:
                with track._lock:
                    existing_sim = track.person_name_similarity
                    if existing_sim == 0.0 or similarity >= existing_sim:
                        track.person_name = name
                        track.person_name_similarity = similarity

    def set_face_crop_path(self, track: 'Track', path: str) -> None:
        """Set the face crop path on the track directly.

        Accepts the Track object so that the path is written even if
        end_recognition() removed the track from _tracks before this call.
        """
        with track._lock:
            track.best_face_crop_path = path

    def get_active_track_ids_and_boxes(self) -> Dict[str, Tuple[int, tuple]]:
        """Return a snapshot of active tracks for IoU dedup (lock-safe).

        Returns dict mapping composite_id -> (byte_track_id, person_box).
        """
        with self._lock:
            return {cid: (t.track_id, t.person_box) for cid, t in self._tracks.items()}

    def set_pending_match_result(self, composite_id: str, match_result: Any) -> None:
        """Store the matching agent result for a track.

        Only upgrades: new similarity >= existing similarity.
        Never lets a no-match (sim=0) clobber a real match.
        """
        with self._lock:
            track = self._tracks.get(composite_id)
            if track:
                existing = track.pending_match_result
                # Never let a no-match (sim=0) clobber a real match.
                # Only upgrade: new sim >= existing sim.
                if existing is None or match_result.similarity_score >= existing.similarity_score:
                    track.pending_match_result = match_result

    def set_pending_memory_context(self, composite_id: str, memory_context: dict) -> None:
        """Store the memory agent result for a track.

        Only upgrades: is_known True > False; visit_count higher > lower.
        """
        with self._lock:
            track = self._tracks.get(composite_id)
            if track:
                existing = track.pending_memory_context
                new_known = memory_context.get("is_known", False)
                existing_known = existing.get("is_known", False) if existing else False
                # Only upgrade: is_known True > False; visit_count higher > lower.
                if new_known or not existing_known:
                    if new_known and existing_known:
                        # Both known — keep the one with higher visit count
                        if memory_context.get("visit_count", 0) >= existing.get("visit_count", 0):
                            track.pending_memory_context = memory_context
                    else:
                        track.pending_memory_context = memory_context

    def set_pending_recognition_data(self, composite_id: str, recognition_result: dict) -> None:
        """Store the recognition agent result for a track.

        Only upgrades: new confidence > existing confidence.
        """
        with self._lock:
            track = self._tracks.get(composite_id)
            if track:
                existing = track.pending_recognition
                new_conf = recognition_result.get("confidence", 0)
                existing_conf = existing.get("confidence", 0) if existing else 0
                # Only upgrade: new confidence > existing confidence.
                if new_conf > existing_conf:
                    track.pending_recognition = recognition_result

    def set_recognition_snapshot(self, composite_id: str, quality: float, status: int) -> None:
        """Store a snapshot of recognition results for debugging.

        Records the face quality, status, and timestamp of the last recognition pass.
        """
        with self._lock:
            track = self._tracks.get(composite_id)
            if track:
                track.last_recognition_quality = quality
                track.last_recognition_status = status
                track.last_recognition_time = time.time()
