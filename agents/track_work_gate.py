"""Exactly-once work gate for track recognition and finalization.

Owns the pair of id sets CameraAgent used to keep inline
(_recognizing_tracks / _finalized_track_ids + _track_sets_lock), extracted
in the Phase 4 split (step 3) so the anti-duplicate invariant lives in one
documented, testable place.

Invariant, stated once:
  a track id is held by at most one live recognition worker, and finalization
  is claimed exactly once by either route (idle expiry or worker release).
  release_finalize() frees the id afterwards so ByteTrack may reuse it.

Every check-and-add below happens under a single lock — that atomicity is
the whole point; do not split a claim into "check then add" at call sites.

Consumed by agents/camera_agent.py (scheduling, worker start, both finalize
routes, the critical-alert finalized check).
"""

import threading


class TrackWorkGate:
    """Lock-protected registry of which track ids are busy / finalized."""

    def __init__(self) -> None:
        self._recognizing: set[str] = set()
        self._finalized: set[str] = set()
        self._lock = threading.Lock()

    # ── recognition (worker-busy) side ─────────────────────────────

    def is_recognizing(self, track_id: str) -> bool:
        """True while a recognition worker holds this track."""
        with self._lock:
            return track_id in self._recognizing

    def begin_recognizing(self, track_id: str) -> None:
        """Mark the track as worker-held (called at worker start, not submit)."""
        with self._lock:
            self._recognizing.add(track_id)

    def release_recognizing(self, track_id: str) -> None:
        """Drop the worker-held mark (idempotent; absent ids are ignored)."""
        with self._lock:
            self._recognizing.discard(track_id)

    def recognizing_count(self) -> int:
        """Number of tracks currently held by workers (debug logging only)."""
        with self._lock:
            return len(self._recognizing)

    # ── finalization (exactly-once) side ───────────────────────────

    def is_finalized(self, track_id: str) -> bool:
        """True once finalization has been claimed for this track."""
        with self._lock:
            return track_id in self._finalized

    def try_claim_finalize_if_idle(self, track_id: str) -> bool:
        """Idle route: claim finalization only if no worker holds the track
        and it is not already claimed. Atomic check-and-add; True = caller
        now owns finalization and must submit TrackFinalizer.finalize_track.
        """
        with self._lock:
            if track_id in self._recognizing or track_id in self._finalized:
                return False
            self._finalized.add(track_id)
            return True

    def try_claim_finalize(self, track_id: str) -> bool:
        """Worker-release route: claim finalization if not already claimed.

        Unlike try_claim_finalize_if_idle this does not consult the
        recognizing set — the caller has just released its own hold and any
        remaining registry-empty precondition is its own to check.
        """
        with self._lock:
            if track_id in self._finalized:
                return False
            self._finalized.add(track_id)
            return True

    def release_finalize(self, track_id: str) -> None:
        """Free the id after finalization completes so ByteTrack may reuse it."""
        with self._lock:
            self._finalized.discard(track_id)
