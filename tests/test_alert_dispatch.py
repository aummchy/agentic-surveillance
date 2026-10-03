"""Tests for the alert dispatch chain.

Covers the Phase 3 review findings (C1, H5, M4, H6):
- dispatch() works when the caller marks first (mark-before-dispatch contract)
- AlertLevel membership check no longer raises NameError (H5)
- should_send_alert() stamps at allow-time (M4 semantics)
- H6: dispatch() and should_send_alert() now have direct coverage
"""

import time

import pytest
from unittest.mock import patch, MagicMock

from pipeline.models import Track, DecisionResult
from agents.alert_agent import dispatch, should_send_alert, _alert_timestamps
from config.status import Status, AlertLevel


@pytest.fixture(autouse=True)
def clean_alert_state():
    """Isolate cooldown state between tests."""
    _alert_timestamps.clear()
    yield
    _alert_timestamps.clear()


@pytest.fixture
def channels():
    """Patch all alert channels + LLM so no I/O leaves the process."""
    with patch("agents.alert_agent._alert_console") as console, \
         patch("agents.alert_agent._alert_email"), \
         patch("agents.alert_agent._alert_sms"), \
         patch("agents.alert_agent._alert_webhook"), \
         patch("agents.alert_agent.llm_client.generate_nl_summary", return_value=None):
        yield console


def make_track(track_id="trk_dispatch_1"):
    return Track(
        track_id=track_id,
        first_seen=time.time() - 5,
        last_seen=time.time(),
        person_box=(0, 0, 100, 200),
        max_track_secs=300,
    )


def make_decision(should_alert=True, alert_level=AlertLevel.CRITICAL,
                  status=Status.BLACKLIST):
    return DecisionResult(
        status=status,
        alert_level=alert_level,
        should_alert=should_alert,
        person_id="p_test",
        name="Test Person",
        reason="unit test",
    )


def test_mark_before_dispatch_succeeds(channels):
    """C1: caller marks first (track_processor/camera_agent order), then dispatches."""
    track = make_track("trk_c1")
    decision = make_decision()
    console = channels

    assert track.mark_alerted_once() is True
    assert dispatch(track, decision) is True
    console.assert_called_once()


def test_dispatch_skips_when_should_alert_false(channels):
    track = make_track("trk_no_alert")
    decision = make_decision(should_alert=False)

    assert dispatch(track, decision) is False
    channels.assert_not_called()
    assert track.alerted is False


def test_cooldown_suppresses_second_dispatch(channels):
    """M4: first call stamps at allow-time, second is suppressed by cooldown."""
    track = make_track("trk_cooldown")
    decision = make_decision(alert_level=AlertLevel.HIGH, status=Status.MASKED_UNKNOWN)

    assert dispatch(track, decision) is True
    assert dispatch(track, decision) is False
    channels.assert_called_once()


def test_valid_alert_level_reaches_payload(channels):
    """H5: dispatch reaches the AlertLevel membership check without NameError
    and builds a payload with the expected keys."""
    track = make_track("trk_payload")
    decision = make_decision(alert_level=AlertLevel.HIGH, status=Status.MASKED_UNKNOWN)

    assert dispatch(track, decision) is True
    payload, _ = channels.call_args[0]
    assert payload["track_id"] == track.track_id
    assert payload["alert_level"] == "high"
    assert payload["status"] == int(Status.MASKED_UNKNOWN)
    for key in ("timestamp", "camera_id", "reason", "is_masked"):
        assert key in payload


def test_should_send_alert_allow_then_suppress():
    assert should_send_alert("t1", "high") is True
    assert should_send_alert("t1", "high") is False
    assert should_send_alert("t2", "high") is True


def test_mark_alerted_once_is_oneshot():
    """Caller-side one-shot contract: first mark wins, later marks fail."""
    track = make_track("trk_oneshot")
    assert track.mark_alerted_once() is True
    assert track.mark_alerted_once() is False
    assert track.alerted is True
