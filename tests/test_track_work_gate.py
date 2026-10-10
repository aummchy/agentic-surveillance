"""Unit tests for TrackWorkGate (agents/track_work_gate.py).

The gate owns the exactly-once invariant that used to live inline in
CameraAgent as _recognizing_tracks / _finalized_track_ids + _track_sets_lock
(Phase 4 split, step 3). These tests pin the primitives the camera agent
now composes: atomic claims, release semantics, and ByteTrack id reuse.

Run: python -m pytest tests/test_track_work_gate.py -v
"""

import threading

from agents.track_work_gate import TrackWorkGate


class TestRecognitionSide:
    def test_begin_and_is_recognizing(self):
        g = TrackWorkGate()
        assert not g.is_recognizing("t1")
        g.begin_recognizing("t1")
        assert g.is_recognizing("t1")

    def test_release_is_idempotent_for_absent_ids(self):
        g = TrackWorkGate()
        g.release_recognizing("ghost")   # must not raise
        g.begin_recognizing("t1")
        g.release_recognizing("t1")
        g.release_recognizing("t1")      # second release still fine
        assert not g.is_recognizing("t1")

    def test_recognizing_count(self):
        g = TrackWorkGate()
        assert g.recognizing_count() == 0
        g.begin_recognizing("a")
        g.begin_recognizing("b")
        assert g.recognizing_count() == 2
        g.release_recognizing("a")
        assert g.recognizing_count() == 1


class TestFinalizeClaimSide:
    def test_first_claim_wins_second_loses(self):
        g = TrackWorkGate()
        assert g.try_claim_finalize("t1") is True
        assert g.try_claim_finalize("t1") is False
        assert g.is_finalized("t1")

    def test_if_idle_claim_refused_while_worker_holds_track(self):
        g = TrackWorkGate()
        g.begin_recognizing("t1")
        assert g.try_claim_finalize_if_idle("t1") is False
        g.release_recognizing("t1")
        assert g.try_claim_finalize_if_idle("t1") is True
        assert g.try_claim_finalize_if_idle("t1") is False   # already claimed

    def test_worker_route_claim_does_not_consult_recognizing(self):
        """The worker-release route claims even if the id is (still) marked
        recognizing — the caller has just released its own hold."""
        g = TrackWorkGate()
        g.begin_recognizing("t1")
        assert g.try_claim_finalize("t1") is True

    def test_release_finalize_allows_id_reuse(self):
        """ByteTrack may reuse an id: release frees it for a fresh claim."""
        g = TrackWorkGate()
        assert g.try_claim_finalize("t1") is True
        g.release_finalize("t1")
        assert not g.is_finalized("t1")
        assert g.try_claim_finalize("t1") is True


class TestConcurrency:
    def test_exactly_one_winner_under_contention(self):
        """N threads race try_claim_finalize on one id; exactly one wins."""
        g = TrackWorkGate()
        winners = []
        lock = threading.Lock()
        n = 16

        def contend():
            if g.try_claim_finalize("hot"):
                with lock:
                    winners.append(1)

        threads = [threading.Thread(target=contend) for _ in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(winners) == 1

    def test_idle_claims_never_race_with_begin_recognizing(self):
        """Under interleaved begin/claim traffic no exception is raised and
        the gate stays consistent."""
        g = TrackWorkGate()
        stop = threading.Event()

        def churn():
            i = 0
            while not stop.is_set():
                tid = f"t{i % 8}"
                g.begin_recognizing(tid)
                g.is_recognizing(tid)
                g.try_claim_finalize_if_idle(tid)
                g.release_recognizing(tid)
                g.release_finalize(tid)
                i += 1

        threads = [threading.Thread(target=churn) for _ in range(4)]
        for t in threads:
            t.start()
        stop.wait(0.3)
        stop.set()
        for t in threads:
            t.join()
        # after full churn + releases everything is clean
        assert g.recognizing_count() == 0
        assert not g.is_finalized("t0")
