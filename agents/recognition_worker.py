"""Progressive recognition worker path: schedule -> run -> write-back -> decide.

Split out of agents/camera_agent.py in the readability refactor (step 5).
All four functions take the owning CameraAgent as their first argument and
read its members (track_state, gate, timing, pipeline, executor, frame
counter) at call time - exactly the lookups the code performed against
``self`` when it lived in that class.

Thread map:

  main thread        maybe_schedule_recognition - runs inside
                     CameraAgent._process_tracks on a recognition-interval
                     tick, only submits to the pool, never blocks.
  recognition pool   progressive_recognition - the worker entry: resolved
  worker threads     check, gate/timing marks, pipeline run, then the
                     write-back (handle_pipeline_result, the sole write-back
                     site) and the decision/alert branch
                     (handle_decision_and_alert). Every exit path lands in
                     TrackFinalizer.cleanup_after_recognition via finally.

CameraAgent keeps a one-line delegate per function, so external references
to the CameraAgent methods - tests that call or replace them, and the
docstrings in pipeline/recognition_pipeline.py, pipeline/track_state.py and
agents/recognition_throttle.py - stay true.
"""

import time
import threading
from typing import TYPE_CHECKING

import numpy as np
import structlog

from config import settings
from config.status import RESOLVED_STATUSES, SkipReason, AlertLevel
from pipeline.models import Track
from pipeline.recognition_pipeline import PipelineResult
from utils.image_utils import save_image, resolve_track_image_url
from agents.alert_agent import dispatch as alert_dispatch

if TYPE_CHECKING:
    from agents.camera_agent import CameraAgent

logger = structlog.get_logger(__name__)


def maybe_schedule_recognition(agent: "CameraAgent", frame: np.ndarray, track: Track) -> None:
    """Schedule recognition for a track if it should be processed.

    Called only on recognition-interval ticks. Runs the throttle checks,
    then submits to the recognition pool unless this track already has a
    worker active.

    Timing caveat worth knowing: the gate's worker-busy check runs here
    but begin_recognizing() happens inside the worker
    (_progressive_recognition), not at submit time.
    A second tick can therefore pass the check before the first worker
    starts; the interval between ticks (RECOGNITION_INTERVAL_FRAMES)
    makes that unlikely, and the worker-start resolved-check is the
    backstop that prevents redundant work from running to completion.

    The submitted frame is a copy — the camera loop keeps mutating its
    own array while the worker reads it.
    """
    if settings.DEBUG_RECOGNITION:
        logger.debug("recognition_tick",
                     track_id=track.track_id,
                     frame=agent._frame_count,
                     decision=track.decision,
                     best_score=round(track.best_face_score, 3),
                     rescan_attempts=track.rescan_attempts)

    should_skip, skip_reason = agent._should_skip_recognition(track)
    if should_skip:
        if settings.DEBUG_RECOGNITION:
            logger.debug("recognition_skipped",
                         track_id=track.track_id,
                         reason=skip_reason,
                         decision=track.decision,
                         current_score=round(track.best_face_score, 3),
                         previous_best=round(track.last_recognition_quality, 3),
                         required_delta=round(track.last_recognition_quality + settings.MIN_QUALITY_IMPROVEMENT, 3),
                         delta=round(track.best_face_score - track.last_recognition_quality, 3))
        logger.debug("skip_recognition_throttled",
                   track_id=track.track_id,
                   decision=track.decision,
                   quality=round(track.best_face_score, 3),
                   last_quality=round(track.last_recognition_quality, 3),
                   last_status=track.last_recognition_status)
        return

    if not agent._gate.is_recognizing(track.track_id):
        logger.debug("progressive_recognition_scheduled",
                   track_id=track.track_id,
                   frame=agent._frame_count)
        if settings.DEBUG_RECOGNITION:
            submit_time = agent._timing.record_submit(track.track_id)
            pending = agent._timing.submit_pending_count()
            logger.debug("recognition_scheduled",
                         track_id=track.track_id,
                         pending=pending)
        agent._recognition_executor.submit(
            agent._progressive_recognition, frame.copy(), track
        )


def progressive_recognition(agent: "CameraAgent", frame: np.ndarray, track: Track) -> None:
    """Run recognition pipeline for a single track in a worker thread.

    Entry point for work submitted by _maybe_schedule_recognition. Runs in
    order: re-check that the track has not already been resolved by
    another worker (the queue-time check cannot see this), record timing,
    mark the track as recognizing, register the in-flight reference with
    TrackState, capture a fallback photo, then run the pipeline.

    Every exit path lands in TrackFinalizer.cleanup_after_recognition via
    finally, which releases the in-flight reference and decides whether
    finalization should be scheduled — including the exception path, so a
    failed pass never strands a track as permanently in-flight.
    """
    worker_start = time.perf_counter()

    # Bail out if the track was already resolved by a previous worker.
    current_track = agent.track_state.get(track.track_id)
    if current_track and current_track.decision in RESOLVED_STATUSES:
        if settings.DEBUG_RECOGNITION:
            logger.debug("recognition_worker_skip_resolved",
                         track_id=track.track_id,
                         decision=current_track.decision)
        return

    if settings.DEBUG_RECOGNITION:
        submit_time = agent._timing.pop_submit(track.track_id)
        queue_delay = (worker_start - submit_time) if submit_time is not None else None
        agent._timing.record_start(track.track_id)
        logger.debug("recognition_worker_start",
                     track_id=track.track_id,
                     queue_delay_ms=round(queue_delay * 1000, 1) if queue_delay is not None else None,
                     thread=threading.current_thread().name)
    agent._gate.begin_recognizing(track.track_id)
    agent.track_state.begin_recognition(track.track_id)
    agent.track_state.ensure_fallback_frame(track, frame)

    try:
        result = agent._pipeline.run(frame, track)
        agent._handle_pipeline_result(result, track, frame)
    except Exception as e:
        logger.error("progressive_recognition_failed", track_id=track.track_id, error=str(e), exc_info=True)
    finally:
        agent._finalizer.cleanup_after_recognition(track)


def handle_pipeline_result(agent: "CameraAgent", result: PipelineResult, track: Track, frame: np.ndarray) -> None:
    """Process recognition pipeline results and update track state.

    This is the single site where a progressive pass writes its findings
    back onto the Track, and it branches on skip_reason first:

      high_confidence     nothing to do, a prior pass already settled it
      no_face/low_quality/embedding_failed
                          store the face only if quality was valid, then stop
      success             run the full write-back in order:
                          visibility -> best face (+ optional debug crop)
                          -> embedding -> person name + pending match
                          -> pending recognition -> pending memory
                          -> decision/alert -> timing log

    Each write goes through a TrackState setter that applies its own
    upgrade rule, so a later weaker pass cannot overwrite a stronger one.
    """
    # Handle skip results (early return)
    if result.skip_reason == SkipReason.HIGH_CONFIDENCE:
        return
    if result.skip_reason in (SkipReason.NO_FACE, SkipReason.LOW_QUALITY, SkipReason.EMBEDDING_FAILED):
        if result.quality and result.quality.is_valid and result.face_crop is not None:
            agent.track_state.set_best_face(
                track, result.face_crop,
                result.quality.overall_score, frame, result.face_ratio,
                person_crop=result.person_crop)
        return

    agent.track_state.update_face_visibility(track.track_id, True, result.face_ratio)

    # Store face crop and debug image
    if result.face_crop is not None:
        if result.quality is None:
            logger.error("quality_none_on_success",
                         track_id=track.track_id,
                         skip_reason=result.skip_reason)
            return
        agent.track_state.set_best_face(
            track, result.face_crop,
            result.quality.overall_score, frame, result.face_ratio,
            person_crop=result.person_crop)
        if settings.DEBUG_FACE_CROPS:
            crop_path = f"captures/debug/face_crops/{track.track_id}/{agent._frame_count}.jpg"
            save_image(result.face_crop, crop_path)
            agent.track_state.set_face_crop_path(track, crop_path)
            logger.info("face_crop_saved", track_id=track.track_id, path=crop_path,
                        url=f"{settings.API_BASE_URL}/{crop_path.replace(chr(92), '/')}")

    # Store embedding, match, recognition, and memory results
    if result.embedding is not None:
        agent.track_state.set_embedding(
            track.track_id, result.embedding,
            result.is_masked, det_score=result.det_score)

    if result.match is not None:
        if result.match.matched and result.match.name:
            agent.track_state.set_person_name(track.track_id, result.match.name, result.match.similarity_score)
        agent.track_state.set_pending_match_result(track.track_id, result.match)

    if result.recognition is not None:
        agent.track_state.set_pending_recognition_data(track.track_id, result.recognition)

    if result.memory:
        agent.track_state.set_pending_memory_context(track.track_id, result.memory)

    # Handle decision and alert logic
    if result.decision is not None and result.recognition is not None:
        agent._handle_decision_and_alert(result, track)

    if settings.DEBUG_RECOGNITION:
        logger.debug("recognition_timing",
                     track_id=track.track_id,
                     crop_detect_ms=result.metrics.crop_detect_ms,
                     fallback_detect_ms=result.metrics.fallback_detect_ms,
                     fallback_used=result.metrics.fallback_used,
                     person_crop_shape=result.metrics.person_crop_shape,
                     faces_on_crop=result.metrics.faces_on_crop,
                     detect_ms=result.metrics.detect_ms,
                     quality_ms=result.metrics.quality_ms,
                     embed_ms=result.metrics.embed_ms,
                     db_ms=result.metrics.db_ms,
                     memory_ms=result.metrics.memory_ms,
                     recog_policy_ms=result.metrics.recog_policy_ms,
                     total_ms=result.metrics.total_ms)


def handle_decision_and_alert(agent: "CameraAgent", result: PipelineResult, track: Track) -> None:
    """Process decision result: upgrade confidence, dispatch alerts.

    Three mutually exclusive branches:

    1. CRITICAL + mark_alerted_once() succeeded: dispatch immediately,
       unless finalization already happened or ALERT_COOLDOWN_SECS has
       not elapsed since the last alert. Only blacklist-level alerts are
       dispatched mid-track; everything else is deferred on purpose so
       an early uncertain pass cannot alert on a verified person.
    2. Confidence upgraded: record the new decision status (unless the
       track is already resolved) and note that a non-critical alert is
       deferred to finalization.
    3. Confidence did not upgrade: log and change nothing — this is the
       max-confidence gate, recognition never downgrades.

    The recognition snapshot is written on every path, including 3,
    because the throttle in _should_skip_recognition needs the latest
    reading regardless of whether the confidence moved.
    """
    new_confidence = int(result.recognition.get("confidence", 0))

    if settings.DEBUG_RECOGNITION:
        logger.debug("decision_check",
                     track_id=track.track_id,
                     new_confidence=new_confidence,
                     existing_confidence=track.confidence,
                     decision_status=result.decision.status,
                     should_alert=result.decision.should_alert,
                     alert_level=result.decision.alert_level)

    alert_cooldown_active = (
        result.decision.should_alert
        and result.decision.alert_level in (AlertLevel.HIGH, AlertLevel.CRITICAL)
        and track.last_alert_time > 0
        and (time.time() - track.last_alert_time) < settings.ALERT_COOLDOWN_SECS
    )

    if result.decision.should_alert and result.decision.alert_level == AlertLevel.CRITICAL and track.mark_alerted_once():
        already_finalized = agent._gate.is_finalized(track.track_id)
        if not already_finalized and not alert_cooldown_active:
            image_url = resolve_track_image_url(track)
            alert_dispatch(track, result.decision, image_url)
            track.last_alert_time = time.time()
            agent.track_state.set_decision(track.track_id, result.decision.status)
            track.update_confidence_if_higher(new_confidence)
            logger.info("progressive_critical_alert",
                        track_id=track.track_id,
                        alert_level=result.decision.alert_level,
                        status=result.decision.status)
    elif track.update_confidence_if_higher(new_confidence):
        if track.decision not in RESOLVED_STATUSES:
            agent.track_state.set_decision(track.track_id, result.decision.status)
        if result.decision.should_alert:
            logger.debug("progressive_alert_deferred_to_finalization",
                         track_id=track.track_id,
                         alert_level=result.decision.alert_level,
                         status=result.decision.status)
    else:
        logger.debug("confidence_not_upgraded",
                     track_id=track.track_id,
                     new=new_confidence,
                     existing=track.confidence,
                     status=result.decision.status)

    agent.track_state.set_recognition_snapshot(
        track.track_id, track.best_face_score, result.decision.status)
