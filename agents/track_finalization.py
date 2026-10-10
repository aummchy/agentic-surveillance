"""Track finalization: ending a person's track exactly once.

Extracted from CameraAgent (Phase 4 split, step 4). Two routes reach a
finalization claim — idle expiry (finalize_expired, called from the camera
thread) and worker release (cleanup_after_recognition, called from a
recognition worker's finally block) — both atomic through TrackWorkGate.
finalize_track is the single close-out point: embedding retry, the
on_track_finalized callback (which enqueues TrackProcessor work), then the
gate release that frees the ByteTrack id for reuse.

All dependencies are injected (gate, track_state, executor, callback,
timing) so the class holds no camera-loop state of its own.
"""

import structlog
from typing import Callable, List, Optional

from config import settings
from pipeline.models import Track
from agents.finalizer import retry_embedding
from agents.track_work_gate import TrackWorkGate

logger = structlog.get_logger(__name__)


class TrackFinalizer:
    """Closes out expired tracks exactly once; composes gate + TrackState."""

    def __init__(self, gate: TrackWorkGate, track_state, executor,
                 on_track_finalized: Optional[Callable[[Track], None]] = None,
                 timing=None) -> None:
        self._gate = gate
        self._track_state = track_state
        self._executor = executor
        self._on_track_finalized = on_track_finalized
        self._timing = timing

    def finalize_expired(self, expired: List[Track]) -> None:
        """Submit expired tracks for finalization if not already processing.

        Idle route to finalization: claims each track through
        TrackWorkGate.try_claim_finalize_if_idle (no worker holding it and
        not already claimed — atomic check-and-add), so a track is scheduled
        once. The worker-held route is handled instead by
        cleanup_after_recognition.
        """
        for track in expired:
            if self._gate.try_claim_finalize_if_idle(track.track_id):
                self._executor.submit(self.finalize_track, track)

    def cleanup_after_recognition(self, track: Track) -> None:
        """Final cleanup after recognition: end recognition, schedule finalization.

        Runs from the recognition worker's finally block, so it executes
        exactly once per pass whether the pass succeeded or raised. It
        releases the in-flight reference (which may itself remove an expired
        track from TrackState), releases the gate's worker-busy mark, and
        then — only if that track has already disappeared from the registry —
        claims finalization via the gate. A track still present here will be
        picked up later by finalize_expired instead.

        Submitting can race with stop() closing the pool; the RuntimeError
        that produces is expected and logged at debug level.
        """
        if settings.DEBUG_RECOGNITION:
            duration_ms = self._timing.get_duration_ms(track.track_id)
            current_track = self._track_state.get(track.track_id)
            track_stale = current_track is not track if current_track else True
            actual_decision = current_track.decision if current_track else "track_removed"
            logger.debug("recognition_worker_done",
                         track_id=track.track_id,
                         duration_ms=duration_ms,
                         decision=track.decision,
                         actual_decision=actual_decision,
                         confidence=track.confidence,
                         track_stale=track_stale)
        self._track_state.end_recognition(track.track_id)
        self._gate.release_recognizing(track.track_id)
        if self._track_state.get(track.track_id) is None and self._gate.try_claim_finalize(track.track_id):
            try:
                self._executor.submit(self.finalize_track, track)
            except RuntimeError:
                logger.debug("finalize_submit_after_shutdown", track_id=track.track_id)

    def finalize_track(self, track: Track) -> None:
        """Finalize one track: retry its embedding, then notify the caller.

        Third and last duplicate guard — Track.mark_finalized_once() — covers
        the case where both scheduling routes somehow reached this point.
        retry_embedding() gives a track that never produced a usable
        embedding one more attempt using the stored best frame; failures are
        logged rather than raised so finalization still completes.

        on_track_finalized fires from this worker thread regardless of the
        embedding outcome, so the caller always sees the track. The id is
        released via TrackWorkGate.release_finalize at the end so ByteTrack
        may reuse it for a later track.
        """
        if not track.mark_finalized_once():
            logger.debug("finalize_skipped_duplicate", track_id=track.track_id)
            return
        try:
            retry_embedding(track, set_embedding=self._track_state.set_embedding)
        except Exception as e:
            logger.error("track_finalization_failed", track_id=track.track_id, error=str(e), exc_info=True)
        finally:
            if self._on_track_finalized:
                self._on_track_finalized(track)
            # Allow ByteTrack ID reuse — release after finalization completes
            # so a new track with the same ID can finalize.
            self._gate.release_finalize(track.track_id)
