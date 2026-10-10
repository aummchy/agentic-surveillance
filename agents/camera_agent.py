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
  workers       _progressive_recognition (body: agents/recognition_worker.py)
                -> _pipeline.run -> write-back -> cleanup -> possibly
                TrackFinalizer.finalize_track.

Coordination between them is the TrackWorkGate (exactly-once claims for
recognition and finalization) plus the lock-protected registry inside
TrackState. Callbacks fire on whichever thread reaches them:
on_track_finalized runs on a recognition worker, on_frame_annotated on the
main thread.

The four worker-path bodies - scheduling, worker entry, result write-back,
decision/alert - live in agents/recognition_worker.py; every method name
kept here is a one-line delegate to it, so existing references to
CameraAgent.<method> stay true.
"""

import cv2
import time
import structlog
import threading
import concurrent.futures
import numpy as np
from typing import Callable, List, Optional, Tuple
from config import settings
from pipeline.tracker import track_persons
from pipeline.track_state import TrackState
from pipeline.models import Track
from pipeline.recognition_pipeline import RecognitionPipeline, PipelineResult
from utils.image_utils import draw_annotations, compute_iou
from agents.timing import TimingCollector
from agents.capture import camera_source, open_capture, apply_frame_props
from agents.recognition_throttle import should_skip_recognition
from agents.track_work_gate import TrackWorkGate
from agents.track_finalization import TrackFinalizer
from agents import recognition_worker

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
        """Delegate to agents/recognition_worker.maybe_schedule_recognition.

        Kept as a method so references to
        CameraAgent._maybe_schedule_recognition (tests that replace it on the
        instance, agents/recognition_throttle.py) stay true; the
        implementation and its full contract live in the worker module.
        """
        return recognition_worker.maybe_schedule_recognition(self, frame, track)

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
        """Delegate to agents/recognition_worker.progressive_recognition.

        Kept as a method so the recognition pool still submits
        self._progressive_recognition and tests can call it directly; the
        implementation and its full contract live in the worker module.
        """
        return recognition_worker.progressive_recognition(self, frame, track)

    def _handle_pipeline_result(self, result: PipelineResult, track: Track, frame: np.ndarray) -> None:
        """Delegate to agents/recognition_worker.handle_pipeline_result.

        Kept as a method so references to
        CameraAgent._handle_pipeline_result (pipeline/recognition_pipeline.py,
        pipeline/track_state.py) stay true; the implementation and its full
        contract live in the worker module.
        """
        return recognition_worker.handle_pipeline_result(self, result, track, frame)

    def _handle_decision_and_alert(self, result: PipelineResult, track: Track) -> None:
        """Delegate to agents/recognition_worker.handle_decision_and_alert.

        Kept as a method so references to
        CameraAgent._handle_decision_and_alert (agents/recognition_throttle.py)
        stay true; the implementation and its full contract live in the
        worker module.
        """
        return recognition_worker.handle_decision_and_alert(self, result, track)
