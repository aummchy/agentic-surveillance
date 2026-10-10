"""Recognition throttle: decide whether a track may be re-recognized.

Extracted verbatim from CameraAgent._should_skip_recognition (Phase 4 split,
step 2). Pure function of (track, settings, time) — no CameraAgent state.

Cameras call it on every recognition-interval tick via
CameraAgent._maybe_schedule_recognition; the throttle reason strings appear in
debug logs ("resolved", "high_confidence", "rescan_interval", "quality_throttle").
"""

import time

from config import settings
from config.status import Status, RESOLVED_STATUSES
from pipeline.models import Track


def should_skip_recognition(track: Track) -> tuple[bool, str]:
    """Determine if recognition should be skipped for this track.

    Returns (should_skip: bool, reason: str).

    Reasons, in order of evaluation:
      "resolved"          decision already reached — nothing left to learn
      "high_confidence"   a prior pass matched above HIGH_CONFIDENCE_SIMILARITY
      "rescan_interval"   UNKNOWN/UNCERTAIN but the rescan backoff has not elapsed
      "quality_throttle"  face quality has not improved on the last attempt
                          by MIN_QUALITY_IMPROVEMENT

    Not a pure predicate: when it decides to allow a rescan it increments
    track.rescan_attempts as a side effect, spending one of the
    MAX_RESCAN_ATTEMPTS budget. Callers therefore must not invoke it
    speculatively.

    The last_recognition_status it branches on is written from the policy
    decision status (see CameraAgent._handle_decision_and_alert), not from the
    recognition agent's own classification — the two can differ.
    """
    already_resolved = track.decision in RESOLVED_STATUSES
    high_confidence = (
        track.pending_match_result
        and track.pending_match_result.similarity_score > settings.HIGH_CONFIDENCE_SIMILARITY
    )

    if already_resolved or high_confidence:
        return True, "resolved" if already_resolved else "high_confidence"

    if track.last_recognition_status in (Status.UNKNOWN, Status.UNCERTAIN):
        has_similarity = (
            track.pending_match_result
            and track.pending_match_result.similarity_score > 0
        )
        if has_similarity and track.rescan_attempts < settings.MAX_RESCAN_ATTEMPTS:
            time_since_last = time.time() - track.last_recognition_time
            if time_since_last >= settings.RESCAN_INTERVAL_SECS:
                track.rescan_attempts += 1
                return False, ""
            return True, "rescan_interval"
        # Throttle if face was previously detected and quality didn't improve
        if track.best_face_score > 0:
            quality_improved = track.best_face_score > (
                track.last_recognition_quality + settings.MIN_QUALITY_IMPROVEMENT
            )
            if not quality_improved:
                return True, "quality_throttle"

    return False, ""
