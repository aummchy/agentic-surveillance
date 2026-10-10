"""Characterization tests for CameraAgent (agents/camera_agent.py).

Purpose: pin the CURRENT behaviour of the five concerns inside CameraAgent
before any structural split (Phase 4 camera_agent refactor, step 0).
These tests must pass unchanged after code is moved between modules -
they describe WHAT the code does, not WHERE it lives.

Covered concerns:
  1. _should_skip_recognition   - 4 skip reasons, 2 allowed paths, rescan side effect
  2. _process_tracks            - track registration, IoU overlap dedup, 20-frame tick
  3. _handle_pipeline_result    - 4 skip branches + success write-back order
  4. _handle_decision_and_alert - critical dispatch / upgrade / no-upgrade
  5. finalization guards        - exactly-once via TrackWorkGate + mark_finalized_once
  6. _progressive_recognition   - recognizing marked at worker start (not submit time)

No camera, MongoDB, Cloudinary, or alert channel is touched: the recognition
pipeline is injected as a mock, TrackState stays real (in-memory), and the
alert/IO helpers are patched at their import site in agents.recognition_worker.

Run: python -m pytest tests/test_camera_agent.py -v
"""

import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from agents.camera_agent import CameraAgent
from config import settings
from config.status import Status, SkipReason, AlertLevel, RESOLVED_STATUSES
from pipeline.models import (
    Track, QualityResult, MatchResult, DecisionResult,
)
from pipeline.recognition_pipeline import PipelineResult


# ───────────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────────

def _track(**kw) -> Track:
    defaults = dict(
        track_id="cam_01_123456_42", first_seen=time.time() - 2.0,
        last_seen=time.time(), person_box=(10, 10, 100, 300),
    )
    defaults.update(kw)
    return Track(**defaults)


def _frame(h=480, w=640) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def _face_crop() -> np.ndarray:
    return np.zeros((50, 40, 3), dtype=np.uint8)


def _valid_quality(score=0.8) -> QualityResult:
    return QualityResult(blur_score=150, brightness=140, face_area=5000,
                         is_valid=True, overall_score=score)


def _make_match(similarity=0.66, name="Alice") -> MatchResult:
    return MatchResult(matched=True, person_id="p_alice", name=name,
                       similarity_score=similarity, verified=True)


@pytest.fixture
def agent() -> CameraAgent:
    """CameraAgent with a mocked pipeline; never opens a capture device."""
    a = CameraAgent(
        on_track_finalized=MagicMock(),
        on_frame_annotated=MagicMock(),
        recognition_pipeline=MagicMock(),
    )
    yield a
    a._recognition_executor.shutdown(wait=False)


# ═══════════════════════════════════════════════════════════════════
# 1. _should_skip_recognition
# ═══════════════════════════════════════════════════════════════════

class TestShouldSkipRecognition:
    def test_fresh_track_is_allowed(self, agent):
        """A brand-new track with no history is allowed (no skip reason)."""
        track = _track()
        assert agent._should_skip_recognition(track) == (False, "")

    def test_resolved_decision_skips(self, agent):
        """Any RESOLVED status stops further recognition."""
        track = _track(decision=Status.VERIFIED)
        assert agent._should_skip_recognition(track) == (True, "resolved")

    def test_high_similarity_skips(self, agent):
        """A prior match above HIGH_CONFIDENCE_SIMILARITY stops recognition."""
        track = _track(pending_match_result=_make_match(
            similarity=settings.HIGH_CONFIDENCE_SIMILARITY + 0.07))
        assert agent._should_skip_recognition(track) == (True, "high_confidence")

    def test_similarity_exactly_at_threshold_is_not_high_confidence(self, agent):
        """Boundary: the check is strict '>' - exactly 0.85 is NOT high_confidence.

        Falls through to the rescan branch instead (allowed, spending an attempt).
        """
        track = _track(pending_match_result=_make_match(
            similarity=settings.HIGH_CONFIDENCE_SIMILARITY),
            last_recognition_time=0.0)
        allowed, reason = agent._should_skip_recognition(track)
        assert (allowed, reason) == (False, "")
        assert track.rescan_attempts == 1   # side effect: attempt spent

    def test_rescan_allowed_when_interval_elapsed_spends_attempt(self, agent):
        """UNKNOWN + mid similarity + budget + elapsed interval -> allowed,
        and rescan_attempts increments by exactly 1 (documented side effect)."""
        track = _track(pending_match_result=_make_match(similarity=0.55),
                       last_recognition_status=Status.UNKNOWN,
                       last_recognition_time=0.0,
                       rescan_attempts=0)
        assert agent._should_skip_recognition(track) == (False, "")
        assert track.rescan_attempts == 1

    def test_rescan_interval_skips_when_too_soon(self, agent):
        """UNKNOWN + budget left but interval not elapsed -> rescan_interval,
        and the attempt is NOT spent."""
        track = _track(pending_match_result=_make_match(similarity=0.55),
                       last_recognition_status=Status.UNKNOWN,
                       last_recognition_time=time.time())
        assert agent._should_skip_recognition(track) == (True, "rescan_interval")
        assert track.rescan_attempts == 0

    def test_uncertain_treated_like_unknown(self, agent):
        """UNCERTAIN follows the same rescan branch as UNKNOWN."""
        track = _track(pending_match_result=_make_match(similarity=0.55),
                       last_recognition_status=Status.UNCERTAIN,
                       last_recognition_time=time.time())
        assert agent._should_skip_recognition(track) == (True, "rescan_interval")

    def test_budget_exhausted_without_quality_history_is_allowed(self, agent):
        """Characterization of a fall-through: rescan budget exhausted but no
        prior face quality (best_face_score == 0) -> the quality gate cannot
        apply, so recognition is allowed again."""
        track = _track(pending_match_result=_make_match(similarity=0.55),
                       last_recognition_status=Status.UNKNOWN,
                       rescan_attempts=settings.MAX_RESCAN_ATTEMPTS,
                       best_face_score=0.0)
        assert agent._should_skip_recognition(track) == (False, "")

    def test_quality_not_improved_throttles(self, agent):
        """Budget exhausted + quality delta below MIN_QUALITY_IMPROVEMENT -> throttle."""
        track = _track(pending_match_result=_make_match(similarity=0.55),
                       last_recognition_status=Status.UNKNOWN,
                       rescan_attempts=settings.MAX_RESCAN_ATTEMPTS,
                       best_face_score=0.50,
                       last_recognition_quality=0.45)   # delta 0.05 < 0.10
        assert agent._should_skip_recognition(track) == (True, "quality_throttle")

    def test_quality_improved_allows_recognition(self, agent):
        """Same setup, but the face improved by >= MIN_QUALITY_IMPROVEMENT."""
        track = _track(pending_match_result=_make_match(similarity=0.55),
                       last_recognition_status=Status.UNKNOWN,
                       rescan_attempts=settings.MAX_RESCAN_ATTEMPTS,
                       best_face_score=0.60,
                       last_recognition_quality=0.45)   # delta 0.15 >= 0.10
        assert agent._should_skip_recognition(track) == (False, "")

    def test_trusted_status_skips_throttle_branch(self, agent):
        """last_recognition_status KNOWN is outside the UNKNOWN/UNCERTAIN
        branch, so no rescan/quality throttling applies at all."""
        track = _track(pending_match_result=_make_match(similarity=0.55),
                       last_recognition_status=Status.KNOWN,
                       best_face_score=0.50,
                       last_recognition_quality=0.50)
        assert agent._should_skip_recognition(track) == (False, "")


# ═══════════════════════════════════════════════════════════════════
# 2. _process_tracks
# ═══════════════════════════════════════════════════════════════════

class TestProcessTracks:
    def _setup(self, agent, frame_count):
        agent._frame_count = frame_count
        agent._maybe_schedule_recognition = MagicMock()

    def test_new_detection_registers_track(self, agent):
        self._setup(agent, 1)
        agent.track_state.update(settings.CAMERA_ID, 42, (10, 10, 100, 300))
        agent._process_tracks(_frame(), [{"track_id": 42, "box": (10, 10, 100, 300)}])
        active = agent.track_state.get_all()
        assert len(active) == 1
        assert active[0].person_box == (10, 10, 100, 300)

    def test_scheduling_only_on_interval_tick(self, agent):
        """_maybe_schedule_recognition fires only when frame_count is a
        multiple of RECOGNITION_INTERVAL_FRAMES."""
        det = [{"track_id": 7, "box": (10, 10, 100, 300)}]
        self._setup(agent, settings.RECOGNITION_INTERVAL_FRAMES)
        agent._process_tracks(_frame(), det)
        assert agent._maybe_schedule_recognition.call_count == 1

        agent2 = CameraAgent(recognition_pipeline=MagicMock())
        try:
            self._setup(agent2, settings.RECOGNITION_INTERVAL_FRAMES + 1)
            agent2._process_tracks(_frame(), det)
            assert agent2._maybe_schedule_recognition.call_count == 0
        finally:
            agent2._recognition_executor.shutdown(wait=False)

    def test_overlapping_boxes_schedule_only_once(self, agent):
        """Two detections with IoU > OVERLAP_IOU_THRESHOLD are treated as one
        person: both are registered, but only the first is scheduled."""
        self._setup(agent, settings.RECOGNITION_INTERVAL_FRAMES)
        dets = [
            {"track_id": 10, "box": (10, 10, 100, 300)},
            {"track_id": 11, "box": (10, 10, 100, 300)},   # identical box, IoU=1.0
        ]
        agent._process_tracks(_frame(), dets)
        assert len(agent.track_state.get_all()) == 2          # both registered
        assert agent._maybe_schedule_recognition.call_count == 1  # one scheduled

    def test_distant_boxes_both_scheduled(self, agent):
        """IoU below the threshold: two people, two scheduled recognitions."""
        self._setup(agent, settings.RECOGNITION_INTERVAL_FRAMES)
        dets = [
            {"track_id": 20, "box": (0, 0, 100, 300)},
            {"track_id": 21, "box": (400, 0, 500, 300)},
        ]
        agent._process_tracks(_frame(), dets)
        assert agent._maybe_schedule_recognition.call_count == 2


# ═══════════════════════════════════════════════════════════════════
# 3. _handle_pipeline_result
# ═══════════════════════════════════════════════════════════════════

class TestHandlePipelineResult:
    def test_high_confidence_is_noop(self, agent):
        """skip_reason=high_confidence: nothing is written back."""
        agent.track_state = MagicMock()
        result = PipelineResult(skip_reason=SkipReason.HIGH_CONFIDENCE)
        agent._handle_pipeline_result(result, _track(), _frame())
        assert agent.track_state.mock_calls == []

    def test_no_face_with_valid_quality_stores_face_only(self, agent):
        """NO_FACE with a valid quality result still stores the best face."""
        agent.track_state = MagicMock()
        result = PipelineResult(
            skip_reason=SkipReason.NO_FACE,
            quality=_valid_quality(), face_crop=_face_crop(), face_ratio=0.5)
        agent._handle_pipeline_result(result, _track(), _frame())
        agent.track_state.set_best_face.assert_called_once()
        agent.track_state.set_embedding.assert_not_called()
        agent.track_state.update_face_visibility.assert_not_called()

    def test_no_face_with_invalid_quality_stores_nothing(self, agent):
        """NO_FACE with invalid quality: the face is discarded entirely."""
        agent.track_state = MagicMock()
        result = PipelineResult(
            skip_reason=SkipReason.NO_FACE, quality=QualityResult.invalid(),
            face_crop=_face_crop())
        agent._handle_pipeline_result(result, _track(), _frame())
        agent.track_state.set_best_face.assert_not_called()

    def test_embedding_failed_uses_same_branch_as_no_face(self, agent):
        """EMBEDDING_FAILED shares the store-face-only branch."""
        agent.track_state = MagicMock()
        result = PipelineResult(
            skip_reason=SkipReason.EMBEDDING_FAILED,
            quality=_valid_quality(), face_crop=_face_crop(), face_ratio=0.5)
        agent._handle_pipeline_result(result, _track(), _frame())
        agent.track_state.set_best_face.assert_called_once()

    def _success_result(self, **over):
        base = dict(
            skip_reason="",        # empty = success (only enum values early-return)
            quality=_valid_quality(), face_crop=_face_crop(), face_ratio=0.6,
            embedding=[0.1] * 8, is_masked=False, det_score=0.9,
            match=_make_match(), recognition={"confidence": 75, "status": Status.KNOWN},
            memory={"visit_count": 1},
            decision=DecisionResult(status=Status.KNOWN, alert_level="none"),
        )
        base.update(over)
        return PipelineResult(**base)

    def test_success_writes_back_in_documented_order(self, agent):
        """Order: visibility -> best face -> embedding -> match -> recognition
        -> memory -> decision/alert (snapshot at the end)."""
        agent.track_state = MagicMock()
        order = []
        ts = agent.track_state
        ts.update_face_visibility.side_effect = lambda *a, **k: order.append("visibility")
        ts.set_best_face.side_effect = lambda *a, **k: order.append("best_face")
        ts.set_embedding.side_effect = lambda *a, **k: order.append("embedding")
        ts.set_pending_match_result.side_effect = lambda *a, **k: order.append("match")
        ts.set_pending_recognition_data.side_effect = lambda *a, **k: order.append("recognition")
        ts.set_pending_memory_context.side_effect = lambda *a, **k: order.append("memory")
        ts.set_recognition_snapshot.side_effect = lambda *a, **k: order.append("snapshot")

        agent._handle_pipeline_result(self._success_result(), _track(), _frame())
        assert order == ["visibility", "best_face", "embedding", "match",
                         "recognition", "memory", "snapshot"]

    def test_success_without_embedding_skips_set_embedding(self, agent):
        agent.track_state = MagicMock()
        result = self._success_result(embedding=None)
        agent._handle_pipeline_result(result, _track(), _frame())
        agent.track_state.set_embedding.assert_not_called()
        agent.track_state.set_pending_match_result.assert_called_once()

    def test_unmatched_result_does_not_set_person_name(self, agent):
        agent.track_state = MagicMock()
        result = self._success_result(
            match=MatchResult(matched=False, similarity_score=0.2))
        agent._handle_pipeline_result(result, _track(), _frame())
        agent.track_state.set_person_name.assert_not_called()
        agent.track_state.set_pending_match_result.assert_called_once()


# ═══════════════════════════════════════════════════════════════════
# 4. _handle_decision_and_alert
# ═══════════════════════════════════════════════════════════════════

class TestHandleDecisionAndAlert:
    def _result(self, confidence=80, status=Status.BLACKLIST,
                alert_level=AlertLevel.CRITICAL, should_alert=True):
        return PipelineResult(
            decision=DecisionResult(status=status, alert_level=alert_level,
                                    should_alert=should_alert),
            recognition={"confidence": confidence})

    def _patched(self):
        return (
            patch("agents.recognition_worker.alert_dispatch"),
            patch("agents.recognition_worker.resolve_track_image_url", return_value=None),
        )

    def test_critical_dispatches_immediately(self, agent):
        dispatch_p, url_p = self._patched()
        with dispatch_p as dispatch, url_p:
            # track must be registered: set_decision writes via TrackState lookup
            track = agent.track_state.update(settings.CAMERA_ID, 42, (10, 10, 100, 300))
            agent._handle_decision_and_alert(self._result(confidence=90), track)
        dispatch.assert_called_once()
        assert track.last_alert_time > 0
        assert track.confidence == 90
        assert track.decision == Status.BLACKLIST

    def test_critical_only_dispatches_once(self, agent):
        """mark_alerted_once() is one-shot: a second CRITICAL pass falls
        through to the confidence-upgrade branch instead of re-dispatching."""
        dispatch_p, url_p = self._patched()
        with dispatch_p as dispatch, url_p:
            track = _track(alerted=True, last_alert_time=time.time())
            agent._handle_decision_and_alert(self._result(confidence=90), track)
        dispatch.assert_not_called()

    def test_critical_skips_if_already_finalized(self, agent):
        dispatch_p, url_p = self._patched()
        with dispatch_p as dispatch, url_p:
            track = _track()
            agent._gate.try_claim_finalize(track.track_id)
            agent._handle_decision_and_alert(self._result(), track)
        dispatch.assert_not_called()

    def test_critical_respects_alert_cooldown(self, agent):
        """A recent HIGH/CRITICAL alert on the same track suppresses dispatch."""
        dispatch_p, url_p = self._patched()
        with dispatch_p as dispatch, url_p:
            track = _track(last_alert_time=time.time() - 1.0)  # 1s ago < 60s
            agent._handle_decision_and_alert(self._result(), track)
        dispatch.assert_not_called()

    def test_non_critical_upgrade_defers_alert(self, agent):
        """Confidence upgrade records the decision but does NOT dispatch -
        non-critical alerts wait for finalization."""
        dispatch_p, _ = self._patched()
        with dispatch_p as dispatch:
            track = agent.track_state.update(settings.CAMERA_ID, 42, (10, 10, 100, 300))
            track.confidence = 40
            agent._handle_decision_and_alert(
                self._result(confidence=70, status=Status.KNOWN,
                             alert_level=AlertLevel.HIGH), track)
        dispatch.assert_not_called()
        assert track.confidence == 70
        assert track.decision == Status.KNOWN

    def test_set_decision_on_unregistered_track_is_noop(self, agent):
        """Characterization: set_decision looks the track up in TrackState -
        a track object not in the registry is silently not updated."""
        dispatch_p, url_p = self._patched()
        with dispatch_p, url_p:
            track = _track()   # never registered
            agent._handle_decision_and_alert(self._result(confidence=90), track)
        assert track.last_alert_time > 0       # alert path mutates the object
        assert track.decision is None          # but the registry write was a no-op

    def test_no_upgrade_changes_nothing_but_snapshots(self, agent):
        """The max-confidence gate: a weaker pass records nothing but still
        writes the recognition snapshot (the throttle needs the latest read)."""
        agent.track_state = MagicMock()
        dispatch_p, _ = self._patched()
        with dispatch_p as dispatch:
            track = _track(confidence=80)
            agent._handle_decision_and_alert(
                self._result(confidence=50, status=Status.UNCERTAIN,
                             alert_level=AlertLevel.NONE, should_alert=False),
                track)
        dispatch.assert_not_called()
        assert track.confidence == 80
        agent.track_state.set_decision.assert_not_called()
        agent.track_state.set_recognition_snapshot.assert_called_once()


# ═══════════════════════════════════════════════════════════════════
# 5. Finalization guards (exactly-once)
# ═══════════════════════════════════════════════════════════════════

class TestFinalizationGuards:
    """The three routes CameraAgent delegates to TrackFinalizer."""

    def test_expired_idle_track_is_scheduled(self, agent):
        agent._finalizer._executor = MagicMock()
        track = _track()
        agent._finalizer.finalize_expired([track])
        agent._finalizer._executor.submit.assert_called_once_with(
            agent._finalizer.finalize_track, track)
        assert agent._gate.is_finalized(track.track_id)

    def test_recognizing_track_is_not_scheduled(self, agent):
        """The worker-held route owns finalization for tracks being recognized."""
        agent._finalizer._executor = MagicMock()
        track = _track()
        agent._gate.begin_recognizing(track.track_id)
        agent._finalizer.finalize_expired([track])
        agent._finalizer._executor.submit.assert_not_called()

    def test_already_scheduled_track_is_not_scheduled_twice(self, agent):
        agent._finalizer._executor = MagicMock()
        track = _track()
        agent._gate.try_claim_finalize(track.track_id)
        agent._finalizer.finalize_expired([track])
        agent._finalizer._executor.submit.assert_not_called()

    def test_finalize_track_runs_once_and_frees_the_id(self, agent):
        """mark_finalized_once() is the third guard; the id is released at
        the end so ByteTrack may reuse it."""
        track = _track()
        agent._gate.try_claim_finalize(track.track_id)
        with patch("agents.track_finalization.retry_embedding") as retry:
            agent._finalizer.finalize_track(track)
            agent._finalizer.finalize_track(track)   # duplicate route arrives late
        assert retry.call_count == 1
        assert agent.on_track_finalized.call_count == 1
        assert not agent._gate.is_finalized(track.track_id)

    def test_finalize_reports_even_if_embedding_retry_fails(self, agent):
        """A retry_embedding exception is logged; the callback still fires."""
        track = _track()
        with patch("agents.track_finalization.retry_embedding",
                   side_effect=RuntimeError("boom")):
            agent._finalizer.finalize_track(track)
        agent.on_track_finalized.assert_called_once_with(track)

    def test_cleanup_schedules_finalize_only_when_track_is_gone(self, agent):
        agent._finalizer._executor = MagicMock()

        # still in the registry -> no finalize, but the busy mark is released
        live = agent.track_state.update(settings.CAMERA_ID, 42, (10, 10, 100, 300))
        agent._gate.begin_recognizing(live.track_id)
        agent._finalizer.cleanup_after_recognition(live)
        agent._finalizer._executor.submit.assert_not_called()
        assert not agent._gate.is_recognizing(live.track_id)

        # gone from the registry (never registered) -> finalize now
        gone = _track()
        agent._gate.begin_recognizing(gone.track_id)
        agent._finalizer.cleanup_after_recognition(gone)
        agent._finalizer._executor.submit.assert_called_once_with(
            agent._finalizer.finalize_track, gone)

    def test_cleanup_survives_executor_shutdown_race(self, agent):
        """Submitting after stop() closed the pool raises RuntimeError - that
        path is expected and must not propagate."""
        agent._finalizer._executor = MagicMock()
        agent._finalizer._executor.submit.side_effect = RuntimeError("shutdown")
        gone = _track()
        agent._gate.begin_recognizing(gone.track_id)
        agent._finalizer.cleanup_after_recognition(gone)   # must not raise
        assert not agent._gate.is_recognizing(gone.track_id)


# ═══════════════════════════════════════════════════════════════════
# 6. _progressive_recognition (worker entry)
# ═══════════════════════════════════════════════════════════════════

class TestProgressiveRecognition:
    def test_marks_recognizing_at_worker_start(self, agent):
        """The gate's worker-busy mark is set inside the WORKER, not at
        submit time: the pipeline must observe the track as already-recognizing."""
        track = agent.track_state.update(settings.CAMERA_ID, 42, (10, 10, 100, 300))
        observed = {}

        def fake_run(frame, t):
            observed["recognizing"] = agent._gate.is_recognizing(t.track_id)
            observed["in_flight"] = agent.track_state._in_flight.get(t.track_id, 0)
            return PipelineResult(skip_reason=SkipReason.HIGH_CONFIDENCE)

        agent._pipeline.run.side_effect = fake_run
        agent._progressive_recognition(_frame(), track)

        assert observed == {"recognizing": True, "in_flight": 1}
        assert not agent._gate.is_recognizing(track.track_id)   # released
        assert track.track_id not in agent.track_state._in_flight  # released

    def test_resolved_track_bails_before_pipeline(self, agent):
        """A track resolved by an earlier worker never reaches the pipeline."""
        track = agent.track_state.update(settings.CAMERA_ID, 42, (10, 10, 100, 300))
        agent.track_state.set_decision(track.track_id, Status.VERIFIED)
        track.decision = Status.VERIFIED
        agent._progressive_recognition(_frame(), track)
        agent._pipeline.run.assert_not_called()
        assert track.track_id not in agent.track_state._in_flight
        assert not agent._gate.is_recognizing(track.track_id)
