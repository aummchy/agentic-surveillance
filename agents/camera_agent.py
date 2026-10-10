"""Camera capture loop, tracking, and recognition scheduling.

CameraAgent owns the live half of the system: it opens the capture device,
runs person detection and ByteTrack on every frame, feeds the resulting
tracks into TrackState, schedules recognition work onto a worker pool, and
hands annotated frames to the dashboard callback.

Two threads meet in this module:

  main thread   start() -> _loop(): read, resize, track, update tracks,
                expire/finalize, annotate, publish. Never blocks on MongoDB
                or alerts — those run on the recognition workers or their
                own executors.
  recognition   ThreadPoolExecutor of RECOGNITION_MAX_WORKERS threads running
  workers       _progressive_recognition -> _pipeline.run -> write-back ->
                cleanup -> possibly TrackFinalizer.finalize_track.

Coordination between them is the TrackWorkGate (exactly-once claims for
recognition and finalization) plus the lock-protected registry inside
TrackState. Callbacks fire on whichever thread reaches them:
on_track_finalized runs on a recognition worker, on_frame_annotated on the
main thread.
"""

import cv2
import time
import structlog
import threading
import concurrent.futures
import numpy as np
from typing import Callable, List, Optional, Tuple
from config import settings
from config.status import Status, RESOLVED_STATUSES, SkipReason, AlertLevel
from pipeline.tracker import track_persons
from pipeline.track_state import TrackState
from pipeline.models import Track
from pipeline.recognition_pipeline import RecognitionPipeline, PipelineResult
from utils.image_utils import draw_annotations, save_image, resolve_track_image_url, compute_iou
from agents.timing import TimingCollector
from agents.alert_agent import dispatch as alert_dispatch
from agents.capture import camera_source, open_capture, apply_frame_props
from agents.recognition_throttle import should_skip_recognition
from agents.track_work_gate import TrackWorkGate
from agents.track_finalization import TrackFinalizer

logger = structlog.get_logger(__name__)


class CameraAgent:
    """Drives the per-frame pipeline and the progressive recognition pool.

    Frame path (main thread), once per captured frame:
        resize -> track_persons -> _process_tracks (update, IoU dedup,
        schedule recognition every RECOGNITION_INTERVAL_FRAMES) ->
        get_expired_tracks + TrackFinalizer.finalize_expired ->
        draw_annotations -> on_frame_annotated -> FPS stats.

    Recognition path (worker thread), per scheduled track:
        resolved-check -> mark recognizing -> begin_recognition ->
        ensure_fallback_frame -> _pipeline.run -> _handle_pipeline_result
        (the sole write-back site) -> TrackFinalizer.cleanup_after_recognition.

    A track reaches finalization by either of two routes, both claimed
    exactly once via TrackWorkGate: finalize_expired when the track expires
    while idle, or cleanup_after_recognition when it expires while a worker
    still holds it. TrackFinalizer.finalize_track adds a third, per-object
    guard via Track.mark_finalized_once().
    """

    def __init__(self, on_track_finalized: Optional[Callable[[Track], None]] = None,
                 on_frame_annotated: Optional[Callable[[np.ndarray], None]] = None,
                 recognition_pipeline: Optional[RecognitionPipeline] = None) -> None:
        self.track_state = TrackState()
        self.on_track_finalized = on_track_finalized
        self.on_frame_annotated = on_frame_annotated
        self._running = False
        self._cap = None
        self._frame_count = 0
        self._stop_event = threading.Event()
        self._pipeline = recognition_pipeline or RecognitionPipeline()
        # Exactly-once invariant: worker-busy set + finalized-claimed set,
        # each check-and-add atomic under the gate's lock. See
        # agents/track_work_gate.py for the full statement.
        self._gate = TrackWorkGate()
        self._recognition_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=settings.RECOGNITION_MAX_WORKERS, thread_name_prefix="recognition"
        )
        self._timing = TimingCollector()
        # Finalization routes (both idle-expiry and worker-release), sharing
        # the gate, the track registry, and the recognition pool above.
        self._finalizer = TrackFinalizer(
            gate=self._gate, track_state=self.track_state,
            executor=self._recognition_executor,
            on_track_finalized=self.on_track_finalized, timing=self._timing,
        )

        # FPS counter state (guarded by settings.PERFORMANCE_STATS)
        self._fps_last_time = time.perf_counter()
        self._fps_frame_count = 0
        self._fps_tracker_ms_total = 0.0
        self._fps_tracker_ms_max = 0.0

    def start(self) -> None:
        """Open the capture and run the frame loop until it stops.

        Blocks the calling thread for the life of the session: _loop() only
        returns on a file source reaching EOF or self._running going false.
        KeyboardInterrupt is treated as a normal shutdown, and stop() runs
        either way via finally.
        """
        self._running = True

        source = camera_source()

        # Open camera once (no test-then-reopen)
        self._cap = open_capture()
        if not self._cap.isOpened():
            logger.error("camera_open_failed", source=source)
            self._cap.release()
            return

        resolution = apply_frame_props(self._cap)

        logger.info("camera_started", source=source, backend=getattr(settings, "CAMERA_BACKEND", "auto"), resolution=resolution,
                    yolo_model=settings.YOLO_MODEL, yolo_device=settings.YOLO_DEVICE, face_model=settings.INSIGHTFACE_MODEL)

        try:
            self._loop()
        except KeyboardInterrupt:
            logger.info("Camera stopped by user")
        finally:
            self.stop()

    def stop(self) -> None:
        """Stop the loop and release everything the session holds.

        Shuts the recognition pool down with wait=True so queued workers
        finish before the capture is released. A worker that is still in
        TrackFinalizer.cleanup_after_recognition may then find the pool
        already closed; that path expects the RuntimeError and logs it
        rather than failing.
        """
        self._running = False
        self._stop_event.set()
        self._recognition_executor.shutdown(wait=True)
        if self._cap:
            self._cap.release()
        logger.info("camera_stopped")

    def _loop(self) -> None:
        """Read frames and run the per-frame pipeline until stopped.

        Failure handling depends on the source type: a local file (string
        source with no "://") treats the first short read as normal EOF and
        exits, while a live camera counts consecutive failures and, once
        CAMERA_MAX_FAILURES is reached, releases the device, waits, and
        reopens it with fixed-delay retries.

        Every accepted frame is resized to FRAME_WIDTH x FRAME_HEIGHT before
        tracking so detection sees a consistent input size regardless of what
        the device actually delivered. The tracker call is timed for the
        optional FPS stats; nothing after it blocks on external I/O.
        """
        consecutive_failures = 0
        source = camera_source()
        is_file_source = isinstance(source, str) and "://" not in source
        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                if is_file_source:
                    logger.info("video_complete", source=source, frames=self._frame_count)
                    break
                consecutive_failures += 1
                if consecutive_failures >= settings.CAMERA_MAX_FAILURES:
                    logger.error("camera_disconnected", failures=consecutive_failures)
                    self._cap.release()
                    time.sleep(settings.CAMERA_RECONNECT_DELAY_SECS)
                    self._cap = open_capture()
                    if self._cap.isOpened():
                        resolution = apply_frame_props(self._cap)
                        logger.info("camera_reconnected", resolution=resolution)
                        consecutive_failures = 0
                    else:
                        logger.error("camera_reconnect_failed")
                        time.sleep(settings.CAMERA_RECONNECT_DELAY_SECS * 2)
                else:
                    time.sleep(0.1)
                continue

            consecutive_failures = 0
            self._frame_count += 1
            frame = cv2.resize(frame, (settings.FRAME_WIDTH, settings.FRAME_HEIGHT))

            _track_start = time.perf_counter()
            tracks = track_persons(frame)
            _tracker_ms = (time.perf_counter() - _track_start) * 1000

            if settings.DEBUG_RECOGNITION:
                logger.debug("recognition_cycle",
                             frame=self._frame_count,
                             track_count=len(tracks),
                             recognizing=self._gate.recognizing_count())

            self._process_tracks(frame, tracks)

            expired = self.track_state.get_expired_tracks()
            self._finalizer.finalize_expired(expired)

            all_tracks = self.track_state.get_all()
            self._log_duplicate_diagnostics(tracks, all_tracks)

            annotated = draw_annotations(frame, all_tracks)
            if self.on_frame_annotated:
                self.on_frame_annotated(annotated)

            if settings.PERFORMANCE_STATS:
                self._update_fps_stats(_tracker_ms, len(tracks), len(all_tracks))

    def _process_tracks(self, frame: np.ndarray, tracks: List[dict]) -> None:
        """Update tracks, dedup overlaps, and schedule recognition.

        For each detection from the tracker: register it in TrackState (or
        update the existing one), suppress it if it overlaps an already-active
        track above OVERLAP_IOU_THRESHOLD (two boxes describing the same
        person), then schedule recognition on the recognition-interval tick.

        Note: get_active_track_ids_and_boxes() rebuilds the whole snapshot
        once per detection, so this is quadratic in the number of active
        tracks within a frame.
        """
        active_ids = set()
        skipped_overlap = set()

        for t in tracks:
            track = self.track_state.update(settings.CAMERA_ID, t["track_id"], t["box"])
            if track:
                active_ids.add(track.track_id)

            # IoU overlap dedup
            if track and track.track_id not in skipped_overlap:
                active_tracks = self.track_state.get_active_track_ids_and_boxes()
                for other_id, (other_track_id, other_box) in active_tracks.items():
                    if other_id == track.track_id:
                        continue
                    if other_track_id in active_ids and other_track_id != track.track_id:
                        iou = compute_iou(track.person_box, other_box)
                        if iou > settings.OVERLAP_IOU_THRESHOLD:
                            skipped_overlap.add(track.track_id)
                            logger.debug("track_skipped_overlap",
                                         track_id=track.track_id,
                                         overlaps_with=other_track_id,
                                         iou=round(iou, 3))
                            break

            if track and track.track_id in skipped_overlap:
                continue

            if track and self._frame_count % settings.RECOGNITION_INTERVAL_FRAMES == 0:
                self._maybe_schedule_recognition(frame, track)

    def _maybe_schedule_recognition(self, frame: np.ndarray, track: Track) -> None:
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
                         frame=self._frame_count,
                         decision=track.decision,
                         best_score=round(track.best_face_score, 3),
                         rescan_attempts=track.rescan_attempts)

        should_skip, skip_reason = self._should_skip_recognition(track)
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

        if not self._gate.is_recognizing(track.track_id):
            logger.debug("progressive_recognition_scheduled",
                       track_id=track.track_id,
                       frame=self._frame_count)
            if settings.DEBUG_RECOGNITION:
                submit_time = self._timing.record_submit(track.track_id)
                pending = self._timing.submit_pending_count()
                logger.debug("recognition_scheduled",
                             track_id=track.track_id,
                             pending=pending)
            self._recognition_executor.submit(
                self._progressive_recognition, frame.copy(), track
            )

    def _should_skip_recognition(self, track: Track) -> Tuple[bool, str]:
        """Delegate to agents/recognition_throttle.should_skip_recognition.

        Kept as a method so existing references to
        CameraAgent._should_skip_recognition (e.g. pipeline/track_state.py)
        stay true; the implementation and its full contract live in the
        throttle module.
        """
        return should_skip_recognition(track)

    def _log_duplicate_diagnostics(self, raw_tracks: List[dict], all_tracks: List[Track]) -> None:
        """Log duplicate box diagnostics when debug is enabled.

        Compares every pair of active tracks and reports those overlapping
        above 0.3 IoU — evidence of the tracker splitting one person into
        two tracks. Gated on DEBUG_DUPLICATE_BOXES; cheap no-op otherwise.
        """
        if not getattr(settings, "DEBUG_DUPLICATE_BOXES", False):
            return
        snapshot = self.track_state.debug_snapshot()
        logger.debug("duplicate_boxes_diag",
                     raw_track_count=len(raw_tracks),
                     state_track_count=len(snapshot),
                     all_tracks_count=len(all_tracks),
                     snapshot=snapshot)
        high_iou_pairs = []
        for i, a in enumerate(all_tracks):
            for b in all_tracks[i + 1:]:
                iou = compute_iou(a.person_box, b.person_box)
                if iou > 0.3:
                    high_iou_pairs.append({
                        "pair": (a.track_id, b.track_id),
                        "iou": round(iou, 3),
                        "boxes": [a.person_box, b.person_box]
                    })
        if high_iou_pairs:
            logger.debug("high_iou_track_pairs",
                         pairs=high_iou_pairs,
                         frame=self._frame_count)

    def _update_fps_stats(self, tracker_ms: float, track_count: int, drawn_count: int) -> None:
        """Update and log FPS performance stats.

        Accumulates over a one-second window, emits a single pipeline_stats
        line when the window closes, then zeroes every accumulator including
        the max so each report describes only the window just finished.
        Called only when PERFORMANCE_STATS is on.
        """
        self._fps_frame_count += 1
        self._fps_tracker_ms_total += tracker_ms
        if tracker_ms > self._fps_tracker_ms_max:
            self._fps_tracker_ms_max = tracker_ms
        now = time.perf_counter()
        if now - self._fps_last_time >= 1.0:
            elapsed = now - self._fps_last_time
            fps = self._fps_frame_count / elapsed
            avg = (
                self._fps_tracker_ms_total / self._fps_frame_count
                if self._fps_frame_count
                else 0.0
            )
            logger.info("pipeline_stats",
                        fps=round(fps, 1),
                        tracker_avg=round(avg, 1),
                        tracker_max=round(self._fps_tracker_ms_max, 1),
                        n_bt=track_count,
                        n_drawn=drawn_count)
            self._fps_last_time = now
            self._fps_frame_count = 0
            self._fps_tracker_ms_total = 0.0
            self._fps_tracker_ms_max = 0.0

    def _progressive_recognition(self, frame: np.ndarray, track: Track) -> None:
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
        current_track = self.track_state.get(track.track_id)
        if current_track and current_track.decision in RESOLVED_STATUSES:
            if settings.DEBUG_RECOGNITION:
                logger.debug("recognition_worker_skip_resolved",
                             track_id=track.track_id,
                             decision=current_track.decision)
            return

        if settings.DEBUG_RECOGNITION:
            submit_time = self._timing.pop_submit(track.track_id)
            queue_delay = (worker_start - submit_time) if submit_time is not None else None
            self._timing.record_start(track.track_id)
            logger.debug("recognition_worker_start",
                         track_id=track.track_id,
                         queue_delay_ms=round(queue_delay * 1000, 1) if queue_delay is not None else None,
                         thread=threading.current_thread().name)
        self._gate.begin_recognizing(track.track_id)
        self.track_state.begin_recognition(track.track_id)
        self.track_state.ensure_fallback_frame(track, frame)

        try:
            result = self._pipeline.run(frame, track)
            self._handle_pipeline_result(result, track, frame)
        except Exception as e:
            logger.error("progressive_recognition_failed", track_id=track.track_id, error=str(e), exc_info=True)
        finally:
            self._finalizer.cleanup_after_recognition(track)

    def _handle_pipeline_result(self, result: PipelineResult, track: Track, frame: np.ndarray) -> None:
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
                self.track_state.set_best_face(
                    track, result.face_crop,
                    result.quality.overall_score, frame, result.face_ratio,
                    person_crop=result.person_crop)
            return

        self.track_state.update_face_visibility(track.track_id, True, result.face_ratio)

        # Store face crop and debug image
        if result.face_crop is not None:
            if result.quality is None:
                logger.error("quality_none_on_success",
                             track_id=track.track_id,
                             skip_reason=result.skip_reason)
                return
            self.track_state.set_best_face(
                track, result.face_crop,
                result.quality.overall_score, frame, result.face_ratio,
                person_crop=result.person_crop)
            if settings.DEBUG_FACE_CROPS:
                crop_path = f"captures/debug/face_crops/{track.track_id}/{self._frame_count}.jpg"
                save_image(result.face_crop, crop_path)
                self.track_state.set_face_crop_path(track, crop_path)
                logger.info("face_crop_saved", track_id=track.track_id, path=crop_path,
                            url=f"{settings.API_BASE_URL}/{crop_path.replace(chr(92), '/')}")

        # Store embedding, match, recognition, and memory results
        if result.embedding is not None:
            self.track_state.set_embedding(
                track.track_id, result.embedding,
                result.is_masked, det_score=result.det_score)

        if result.match is not None:
            if result.match.matched and result.match.name:
                self.track_state.set_person_name(track.track_id, result.match.name, result.match.similarity_score)
            self.track_state.set_pending_match_result(track.track_id, result.match)

        if result.recognition is not None:
            self.track_state.set_pending_recognition_data(track.track_id, result.recognition)

        if result.memory:
            self.track_state.set_pending_memory_context(track.track_id, result.memory)

        # Handle decision and alert logic
        if result.decision is not None and result.recognition is not None:
            self._handle_decision_and_alert(result, track)

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

    def _handle_decision_and_alert(self, result: PipelineResult, track: Track) -> None:
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
            already_finalized = self._gate.is_finalized(track.track_id)
            if not already_finalized and not alert_cooldown_active:
                image_url = resolve_track_image_url(track)
                alert_dispatch(track, result.decision, image_url)
                track.last_alert_time = time.time()
                self.track_state.set_decision(track.track_id, result.decision.status)
                track.update_confidence_if_higher(new_confidence)
                logger.info("progressive_critical_alert",
                            track_id=track.track_id,
                            alert_level=result.decision.alert_level,
                            status=result.decision.status)
        elif track.update_confidence_if_higher(new_confidence):
            if track.decision not in RESOLVED_STATUSES:
                self.track_state.set_decision(track.track_id, result.decision.status)
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

        self.track_state.set_recognition_snapshot(
            track.track_id, track.best_face_score, result.decision.status)
