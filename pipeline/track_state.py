"""Thread-safe registry of live person tracks.

TrackState owns the composite_id -> Track mapping and is the only place that
adds to or removes from it. It does not run recognition itself; it exists so
that the camera thread and the recognition worker pool can safely touch the
same Track objects concurrently.

Two locks, fixed order:

    self._lock  ->  track._lock

self._lock guards the _tracks dict and the other bookkeeping dicts below.
track._lock guards an individual Track's fields. Any method that needs both
acquires self._lock first and track._lock second. No code path in this module
acquires self._lock while holding track._lock.

Three setters deliberately take only track._lock (set_best_face,
ensure_fallback_frame, set_face_crop_path). They accept the Track object
itself rather than a composite_id lookup so that a write still lands after
end_recognition() has removed the track from _tracks.

Recognition in-flight protocol: begin_recognition() increments a per-track
counter before a worker starts, end_recognition() decrements it. An expired
track is reported to the finalizer but not removed from _tracks while the
counter is non-zero, so a worker never writes into a dropped dict entry.
"""

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

    Coordinates between the camera thread (update, get_expired_tracks,
    get_all, get_active_track_ids_and_boxes) and recognition workers
    (set_best_face, set_embedding, set_pending_*, begin/end_recognition)
    using a two-level lock: self._lock for the registry, track._lock for an
    individual Track's fields, always acquired in that order.

    An in-flight reference counter (_in_flight) prevents premature removal:
    a track that expires while a worker is writing to it stays in _tracks
    until the last end_recognition() call.
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

        ByteTrack reuses its integer ids across the session, so a generation
        suffix keeps a new track distinct from an earlier one that had the
        same bt_id. Only update() calls this.

        If the bt_id is already marked active this means the caller saw a
        track it should have matched instead; returns the live generation
        unchanged and logs, rather than allocating a competing one.
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

        Called from remove() and from both removal paths in the expiry flow
        (get_expired_tracks and end_recognition). Popping a bt_id that is not
        active is a no-op, so repeated releases are safe.
        """
        gen = self._bt_active.pop(bt_id, None)
        if gen is not None:
            self._bt_next_gen[bt_id] = gen + 1

    @staticmethod
    def make_composite_id(camera_id: str, bt_id: int, generation: int, session_epoch: int) -> str:
        """Pure formatter — no mutable state.

        Produces the composite track id {camera_id}_{session_epoch}_{bt_id}_{generation}.
        session_epoch is captured once in __init__, so ids stay unique across
        camera restarts within a process; the generation handles ByteTrack
        id reuse within a session.
        """
        return f"{camera_id}_{session_epoch}_{bt_id}_{generation}"

    def update(self, camera_id: str, track_id: int, box: tuple,
               frame: np.ndarray = None) -> Optional[Track]:
        """Update an existing track or create a new one.

        If a track with the given byte_track_id already exists, update its
        bounding box and last_seen time. Otherwise, allocate a new generation
        and create a fresh Track object.

        Returns the updated or newly created Track, or None if creation failed.

        The existing-track lookup scans _tracks and compares byte_track_id
        only — camera_id does not participate in the match, which is correct
        for the single-camera deployment this system targets. The scan is
        linear in the number of active tracks and runs under self._lock.

        The optional frame argument is unused by this implementation; it is
        kept for signature compatibility with callers.
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

        Returns the live object, not a copy — callers read its fields and may
        hold track._lock. A None result means the track has already been
        removed (expired and not in-flight), which callers use as a
        "still here?" check.
        """
        with self._lock:
            return self._tracks.get(composite_id)

    def get_all(self) -> List[Track]:
        """Return a snapshot of all active tracks.

        Returns a new list (safe to iterate while other threads modify _tracks).

        The list is a copy; the Track objects inside it are live references.
        """
        with self._lock:
            return list(self._tracks.values())

    def debug_snapshot(self) -> List[dict]:
        """Return a debug-friendly snapshot of all tracks.

        Each dict contains key track properties for logging and diagnostics.
        Only read when DEBUG_RECOGNITION is on; builds plain dicts so no
        mutable Track state escapes into the logs.
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

        Not called by any current production or test code — the expiry flow
        in get_expired_tracks()/end_recognition() performs removal inline so
        it can also handle logging and generation release together. Kept as
        part of the public surface; note that it does not clear _in_flight,
        so it must not be used on a track with a live recognition task.
        """
        with self._lock:
            track = self._tracks.pop(composite_id, None)
            if track:
                self.release_generation(track.byte_track_id)
            return track

    def get_expired_tracks(self) -> List[Track]:
        """Identify and return tracks that have exceeded their timeout.

        A track qualifies when TRACK_TIMEOUT_SECS of inactivity has passed or
        its MAX_TRACK_SECS lifetime is up.

        Tracks with in-flight recognition tasks are reported but NOT removed
        until end_recognition() clears the last reference.

        Two-phase under a single self._lock hold: first mark and collect
        (setting expired_reported so a track is only ever reported once),
        then delete the subset whose _in_flight count is already zero. Tracks
        left in _tracks by this call are the ones end_recognition() is
        responsible for cleaning up.
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
        """Classify visibility directly on the track object (no dict lookup).

        Called only from get_expired_tracks() while self._lock is already
        held, which is sufficient on its own here: every field this reads
        (max_face_ratio, face_detected_once, is_masked, total_frames_seen)
        is written by paths that also hold self._lock — update(),
        update_face_visibility(), set_embedding(). track._lock is therefore
        not required. Keep it that way; if any of those fields later gets
        written under track._lock alone, this method must take both locks.

        Ratios: face area relative to the person box, so a face that fills
        at least VISIBLE_FACE_RATIO counts as visible, PARTIAL_FACE_RATIO as
        partial. A track with a detected-but-too-small face counts as partial;
        one old enough to have shown a face but never did counts as hidden.
        """
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

        Only max_face_ratio's high-water mark is retained; the visibility
        classification itself is deferred to expiry (see
        _classify_visibility_inplace).
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
        """Mark a track as having an in-flight recognition task.

        Called by a recognition worker before it starts writing to the track.
        Increments _in_flight under self._lock; pairs with end_recognition()
        in the worker's finally block. A track whose count is above zero is
        exempt from removal in get_expired_tracks().
        """
        with self._lock:
            self._in_flight[composite_id] = self._in_flight.get(composite_id, 0) + 1

    def end_recognition(self, composite_id: str) -> None:
        """Clear one in-flight recognition reference for a track.

        If the track has expired and no other recognition threads hold a
        reference, it is removed from _tracks here so finalization can proceed.

        This is the second half of the expiry protocol: get_expired_tracks()
        defers removal while a worker is active and relies on this method —
        running in that worker's finally block — to do the removal once the
        last reference drops. Removal releases the ByteTrack generation the
        same way the inline path does.

        A counter already at zero (a stray end_recognition for an unknown or
        long-finished track) leaves the entry deleted rather than going
        negative.
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

        Three phases, and the thresholds differ between them on purpose:
        1. under track._lock, reject early unless face_score improves on the
           stored one by more than FACE_SCORE_IMPROVEMENT_MIN — a cheap gate
           that keeps marginal frames from paying the encode cost;
        2. JPEG-encode the full frame (and optional person crop) *outside*
           the lock, since encoding a 1080p frame is the expensive part;
        3. under track._lock again, re-check face_score > best_face_score
           before writing. The weaker re-check is a race guard, not the
           filter: another worker may have raised the score while we were
           encoding, and this write must never lower what is stored.
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

        Exists because set_best_face() only stores a photo when the face
        beats the quality gate — a track that never produced an acceptable
        face would otherwise reach finalization with no image at all.
        First caller wins; concurrent callers may both encode but only one
        result is kept.
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

        Reasons: "ok", "rejected_quality" (a better detection score is
        already stored, so this embedding would be a downgrade),
        "track_removed" (no live track for this id).

        Two callers, only one of which inspects the result:
        camera_agent._handle_pipeline_result ignores it, while
        finalizer.retry_embedding unpacks it to decide whether a retry is
        worth attempting.
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

        The decision argument is a config.status.Status member, which is an
        IntEnum — the Track field and this parameter are both typed int for
        that reason. Accepts any Status value; no validation happens here.
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

        person_name_similarity doubles as "unset" detection: it starts at
        0.0, so a stored 0.0 is indistinguishable from never having been
        set, and the first write always lands regardless of its similarity.
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

        Path only — the crop image bytes already live on the track from
        set_best_face(). Written under track._lock alone, by design.
        """
        with track._lock:
            track.best_face_crop_path = path

    def get_active_track_ids_and_boxes(self) -> Dict[str, Tuple[str, tuple]]:
        """Return a snapshot of active tracks for IoU dedup (lock-safe).

        Returns dict mapping composite_id -> (composite_id, person_box).

        The value's first element is the track's own composite id — the same
        string as its key — not the ByteTrack id. Consumers unpack it as
        (other_track_id, other_box) and compare it against a set of
        composite ids, which is what makes the duplicate correct for them.
        """
        with self._lock:
            return {cid: (t.track_id, t.person_box) for cid, t in self._tracks.items()}

    def set_pending_match_result(self, composite_id: str, match_result: Any) -> None:
        """Store the matching agent result for a track.

        Only upgrades: new similarity >= existing similarity.
        Never lets a no-match (sim=0) clobber a real match.

        Any MatchResult is accepted; the upgrade rule reads its
        similarity_score. Consumed at finalization by
        TrackProcessor._run_matching(), which uses it instead of re-querying.
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

        The rule is deliberately not a simple max: a known context always
        beats an unknown one, and when both are known the one with more
        visits wins. A lower visit_count from a later pass is treated as
        stale and dropped rather than overwriting the better record.
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

        Strictly greater — an equal-confidence pass is dropped, keeping the
        first result that reached that level. This is the guard that stops a
        later weaker pass from replacing a stronger one at finalization.
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

        Unlike the pending_* setters this one has no upgrade rule — it always
        overwrites, because its job is to describe the most recent pass so
        the throttle in CameraAgent._should_skip_recognition can compare the
        new face quality against the last one attempted.
        """
        with self._lock:
            track = self._tracks.get(composite_id)
            if track:
                track.last_recognition_quality = quality
                track.last_recognition_status = status
                track.last_recognition_time = time.time()
