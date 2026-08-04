"""
Phase 4.3 + 4.4 — Thread-Safety Regression Tests

Phase 4.3: Proves the old read-modify-write update_visit_memory lost concurrent
           updates.  The new atomic version must count every call.
Phase 4.4: Proves get_expired_tracks() used to silently drop in-flight track
           data.  The new version defers removal while recognition is active.

Run: python -m pytest tests/test_thread_safety.py -v
"""

import threading
import time
from unittest.mock import patch, MagicMock
from datetime import datetime

import numpy as np
import pytest

from pipeline.track_state import TrackState
from pipeline.models import Track
from config import settings


# ═══════════════════════════════════════════════════════════════════
# Phase 4.3 — Atomic visit-memory updates
# ═══════════════════════════════════════════════════════════════════

class FakeMemoryCollection:
    """In-memory stand-in for the MongoDB visit_memory collection.

    Simulates the old (non-atomic) read-modify-write behaviour when
    ``atomic=False`` and the new atomic behaviour when ``atomic=True``,
    so we can demonstrate the race condition and its fix without a
    real database.
    """

    def __init__(self, atomic=False):
        self._store = {}
        self._atomic = atomic
        self._lock = threading.Lock()

    # -- non-atomic (old) path -------------------------------------------------

    def _get(self, person_id):
        return self._store.get(person_id)

    def _upsert(self, person_id, doc):
        self._store[person_id] = doc

    # -- simulated MongoDB operations ------------------------------------------

    def find_one_and_update(self, filter_doc, update_doc, upsert=False,
                             return_document=False):
        """Simulates find_one_and_update — atomic or not."""
        person_id = filter_doc.get("person_id")

        if self._atomic:
            return self._atomic_update(person_id, update_doc)
        else:
            return self._legacy_update(person_id, update_doc)

    def find_one(self, filter_doc, projection=None):
        """Simulates find_one — returns the stored doc or None."""
        person_id = filter_doc.get("person_id")
        doc = self._store.get(person_id)
        if doc is None:
            return None
        return dict(doc)

    def _legacy_update(self, person_id, update_doc):
        """Non-atomic read-modify-write (the old buggy path)."""
        with self._lock:
            doc = self._store.get(person_id)
            if doc is None:
                doc = {
                    "person_id": person_id,
                    "visit_count": 0,
                    "similarity_history": [],
                    "status_history": [],
                    "typical_hours": [],
                    "typical_cameras": [],
                }
                self._store[person_id] = doc

            # Simulate the TOCTOU: another thread could overwrite between
            # our read and write.  We force the race by yielding here.
            time.sleep(0.001)

            # Apply the same logic the OLD code used
            ops = update_doc.get("$set", {})
            if "visit_count" in ops:
                doc["visit_count"] = ops["visit_count"]
            if "similarity_history" in ops:
                doc["similarity_history"] = ops["similarity_history"]
            if "status_history" in ops:
                doc["status_history"] = ops["status_history"]
            if "typical_hours" in ops:
                doc["typical_hours"] = ops["typical_hours"]
            if "typical_cameras" in ops:
                doc["typical_cameras"] = ops["typical_cameras"]

            return doc

    def _atomic_update(self, person_id, update_doc):
        """Simulates the new single-round-trip atomic update."""
        with self._lock:
            doc = self._store.get(person_id)
            if doc is None:
                doc = {
                    "person_id": person_id,
                    "visit_count": 0,
                    "similarity_history": [],
                    "status_history": [],
                    "typical_hours": [],
                    "typical_cameras": [],
                }
                self._store[person_id] = doc

            # $inc
            if "$inc" in update_doc:
                for k, v in update_doc["$inc"].items():
                    doc[k] = doc.get(k, 0) + v

            # $push (with $slice)
            if "$push" in update_doc:
                for k, push_spec in update_doc["$push"].items():
                    items = push_spec.get("$each", [])
                    slice_n = push_spec.get("$slice", None)
                    lst = doc.get(k, [])
                    lst.extend(items)
                    if slice_n is not None and slice_n < 0:
                        lst = lst[slice_n:]
                    doc[k] = lst

            # $addToSet
            if "$addToSet" in update_doc:
                for k, v in update_doc["$addToSet"].items():
                    lst = doc.get(k, [])
                    if v not in lst:
                        lst.append(v)
                    doc[k] = lst

            # $set
            if "$set" in update_doc:
                for k, v in update_doc["$set"].items():
                    doc[k] = v

            return dict(doc)  # return a copy


# ── 4.3a: Old pattern LOSES concurrent visits ──────────────────────

def _old_update_visit(person_id, collection, similarity):
    """Reproduces the pre-fix read-modify-write pattern."""
    now = datetime.utcnow()
    doc = collection._get(person_id)
    if doc is None:
        doc = {
            "person_id": person_id,
            "visit_count": 0,
            "similarity_history": [],
            "status_history": [],
            "typical_hours": [],
            "typical_cameras": [],
        }

    time.sleep(0.001)  # widen TOCTOU window

    new_count = doc["visit_count"] + 1
    sim_hist = doc["similarity_history"] + [similarity]
    doc["visit_count"] = new_count
    doc["similarity_history"] = sim_hist[-10:]
    collection._upsert(person_id, doc)


class TestPhase4_3_VisitMemoryRace:
    """Proves the old read-modify-write loses concurrent updates."""

    def test_old_pattern_loses_visits(self):
        """Spawn N threads updating the same person concurrently.
        The old pattern will record fewer visits than N because
        threads overwrite each other's visit_count."""
        collection = FakeMemoryCollection(atomic=False)
        person_id = "race_test_person"
        n_threads = 20
        barrier = threading.Barrier(n_threads)

        def worker(similarity):
            barrier.wait()
            _old_update_visit(person_id, collection, similarity)

        threads = [
            threading.Thread(target=worker, args=(i * 0.1,))
            for i in range(n_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        doc = collection._get(person_id)
        # The old pattern loses visits — count < n_threads.
        # This is the bug we are proving exists.
        assert doc["visit_count"] < n_threads, (
            f"Old pattern unexpectedly kept all visits: "
            f"{doc['visit_count']} == {n_threads}"
        )

    def test_new_pattern_counts_every_visit(self):
        """Spawn N threads — the atomic version must record exactly N visits."""
        from utils.db_utils import update_visit_memory

        collection = FakeMemoryCollection(atomic=True)
        person_id = "atomic_test_person"
        n_threads = 20
        barrier = threading.Barrier(n_threads)

        def worker(similarity):
            barrier.wait()
            with patch("utils.db_utils.get_memory_collection",
                       return_value=collection), \
                 patch("utils.db_utils.settings") as mock_settings:
                mock_settings.MIN_VISIT_GAP_SECS = 0
                update_visit_memory(person_id, "cam1", "known", similarity)

        threads = [
            threading.Thread(target=worker, args=(i * 0.1,))
            for i in range(n_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        doc = collection._get(person_id)
        assert doc["visit_count"] == n_threads, (
            f"Atomic version lost visits: got {doc['visit_count']}, "
            f"expected {n_threads}"
        )

    def test_similarity_history_capped_at_10(self):
        """Even with many concurrent pushes, history stays bounded."""
        from utils.db_utils import update_visit_memory

        collection = FakeMemoryCollection(atomic=True)
        person_id = "bounded_history_person"

        with patch("utils.db_utils.get_memory_collection",
                   return_value=collection), \
             patch("utils.db_utils.settings") as mock_settings:
            mock_settings.MIN_VISIT_GAP_SECS = 0
            for i in range(25):
                update_visit_memory(person_id, "cam1", "known", i * 0.01)

        doc = collection._get(person_id)
        assert len(doc["similarity_history"]) <= 10, (
            f"similarity_history should be ≤10, got "
            f"{len(doc['similarity_history'])}"
        )

    def test_typical_cameras_deduped(self):
        """Same camera_id must not appear twice in typical_cameras."""
        from utils.db_utils import update_visit_memory

        collection = FakeMemoryCollection(atomic=True)
        person_id = "dedup_cam_person"

        with patch("utils.db_utils.get_memory_collection",
                   return_value=collection), \
             patch("utils.db_utils.settings") as mock_settings:
            mock_settings.MIN_VISIT_GAP_SECS = 0
            for _ in range(10):
                update_visit_memory(person_id, "cam1", "known", 0.5)

        doc = collection._get(person_id)
        assert doc["typical_cameras"].count("cam1") == 1, (
            f"cam1 should appear once, got {doc['typical_cameras']}"
        )


# ═══════════════════════════════════════════════════════════════════
# Phase 4.4 — Track expiration vs. recognition thread
# ═══════════════════════════════════════════════════════════════════

class TestPhase4_4_TrackExpirationRace:
    """Proves get_expired_tracks() no longer kills in-flight recognition."""

    def _make_track(self, track_id, timeout=0):
        """Create a Track that is already expired (or not)."""
        now = time.time()
        # first_seen must be within max_track_secs to avoid lifetime expiry
        return Track(
            track_id=track_id,
            first_seen=now - 10,
            last_seen=now - timeout,
            person_box=(0, 0, 100, 100),
            max_track_secs=600,
        )

    def test_expired_track_removed_when_no_recognition(self):
        """Without in-flight recognition, expired tracks are removed."""
        ts = TrackState()
        track = self._make_track("cam_0_1", timeout=999)
        with ts._lock:
            ts._tracks[track.track_id] = track

        expired = ts.get_expired_tracks()
        assert len(expired) == 1
        assert expired[0].track_id == track.track_id
        # Track should be gone from _tracks
        assert ts.get(track.track_id) is None

    def test_expired_track_deferred_while_recognizing(self):
        """In-flight recognition keeps the expired track alive."""
        ts = TrackState()
        track = self._make_track("cam_0_2", timeout=999)
        with ts._lock:
            ts._tracks[track.track_id] = track

        # Simulate a recognition thread marking the track as in-flight
        ts.begin_recognition(track.track_id)

        expired = ts.get_expired_tracks()
        assert len(expired) == 1  # reported as expired
        # …but NOT removed from _tracks
        assert ts.get(track.track_id) is not None, (
            "Track should still exist while recognition is in-flight"
        )

        # Finish recognition → track is now removed
        ts.end_recognition(track.track_id)
        assert ts.get(track.track_id) is None, (
            "Track should be removed after last recognition completes"
        )

    def test_multiple_recognition_threads_defer_removal(self):
        """Two concurrent recognition threads both defer removal."""
        ts = TrackState()
        track = self._make_track("cam_0_3", timeout=999)
        with ts._lock:
            ts._tracks[track.track_id] = track

        ts.begin_recognition(track.track_id)
        ts.begin_recognition(track.track_id)  # second thread

        expired = ts.get_expired_tracks()
        assert len(expired) == 1
        assert ts.get(track.track_id) is not None, (
            "Track must survive until both recognition threads finish"
        )

        ts.end_recognition(track.track_id)  # first thread done
        assert ts.get(track.track_id) is not None, (
            "Track must survive while second thread is still active"
        )

        ts.end_recognition(track.track_id)  # second thread done
        assert ts.get(track.track_id) is None

    def test_non_expired_track_not_returned(self):
        """Tracks that haven't expired are not returned by get_expired_tracks."""
        ts = TrackState()
        track = self._make_track("cam_0_4", timeout=0)  # last_seen = now
        with ts._lock:
            ts._tracks[track.track_id] = track

        expired = ts.get_expired_tracks()
        assert len(expired) == 0

    def test_concurrent_recognition_and_expiration(self):
        """Simulates the real race: camera thread expires tracks while
        recognition threads are writing to them concurrently."""
        ts = TrackState()
        n_tracks = 10
        barrier = threading.Barrier(n_tracks + 1)  # +1 for camera thread

        for i in range(n_tracks):
            track = self._make_track(f"cam_0_{i}", timeout=999)
            with ts._lock:
                ts._tracks[track.track_id] = track

        recognition_done = threading.Event()
        writes_during_expiration = []

        def recognition_worker(track_id):
            ts.begin_recognition(track_id)
            barrier.wait()  # sync with camera thread
            # Simulate writing to the track after expiration would have removed it
            time.sleep(0.005)
            ts.set_embedding(track_id, [0.1] * 512)
            # Pass a real numpy array so cv2.imencode doesn't crash
            dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
            track_obj = ts.get(track_id)
            if track_obj:
                ts.set_best_face(track_obj, dummy_frame, 0.9, dummy_frame, 0.5)
            ts.end_recognition(track_id)

        def camera_worker():
            barrier.wait()  # sync with recognition threads
            expired = ts.get_expired_tracks()
            # All tracks should still be accessible because they're in-flight
            for t in expired:
                obj = ts.get(t.track_id)
                if obj is None:
                    writes_during_expiration.append(t.track_id)

        threads = []
        for i in range(n_tracks):
            threads.append(threading.Thread(
                target=recognition_worker, args=(f"cam_0_{i}",)
            ))
        threads.append(threading.Thread(target=camera_worker))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(writes_during_expiration) == 0, (
            f"Tracks removed while in-flight: {writes_during_expiration}"
        )

    def test_end_recognition_cleans_up_expired_track(self):
        """After end_recognition, the expired track is removed so
        finalization can proceed."""
        ts = TrackState()
        track = self._make_track("cam_0_5", timeout=999)
        with ts._lock:
            ts._tracks[track.track_id] = track

        ts.begin_recognition(track.track_id)

        # get_expired_tracks reports it but does NOT remove it
        expired = ts.get_expired_tracks()
        assert len(expired) == 1
        assert ts.get(track.track_id) is not None

        # end_recognition removes it
        ts.end_recognition(track.track_id)
        assert ts.get(track.track_id) is None


class TestByteTrackIdReuse:
    """Regression tests for ByteTrack ID reuse lifecycle tracking."""

    def test_first_track_gets_generation_zero(self):
        ts = TrackState()
        t1 = ts.update("cam_01", 2, (100, 100, 200, 200))
        assert t1.generation == 0
        assert t1.byte_track_id == 2
        assert t1.track_id.endswith("_0")

    def test_reuse_gets_unique_composite_id(self):
        ts = TrackState()
        t1 = ts.update("cam_01", 2, (100, 100, 200, 200))
        assert t1.generation == 0
        ts.remove(t1.track_id)

        t2 = ts.update("cam_01", 2, (100, 100, 200, 200))
        assert t2.generation == 1
        assert t2.byte_track_id == 2
        assert t2.track_id.endswith("_1")
        assert t1.track_id != t2.track_id

    def test_multiple_reuses_increment_generation(self):
        ts = TrackState()
        t1 = ts.update("cam_01", 5, (10, 10, 50, 50))
        assert t1.generation == 0
        ts.remove(t1.track_id)

        t2 = ts.update("cam_01", 5, (10, 10, 50, 50))
        assert t2.generation == 1
        ts.remove(t2.track_id)

        t3 = ts.update("cam_01", 5, (10, 10, 50, 50))
        assert t3.generation == 2
        assert t3.track_id.endswith("_2")

    def test_different_bt_ids_independent_generations(self):
        ts = TrackState()
        t_a = ts.update("cam_01", 2, (100, 100, 200, 200))
        t_b = ts.update("cam_01", 3, (300, 300, 400, 400))
        assert t_a.generation == 0
        assert t_b.generation == 0
        assert t_a.track_id != t_b.track_id

        ts.remove(t_a.track_id)
        t_a2 = ts.update("cam_01", 2, (100, 100, 200, 200))
        assert t_a2.generation == 1
        assert t_b.generation == 0  # bt_id=3 still generation 0

    def test_release_generation_idempotent(self):
        ts = TrackState()
        t1 = ts.update("cam_01", 2, (100, 100, 200, 200))
        ts.remove(t1.track_id)
        # Second release should be a no-op
        ts.remove(t1.track_id)

        t2 = ts.update("cam_01", 2, (100, 100, 200, 200))
        assert t2.generation == 1  # only incremented once

    def test_update_existing_track_returns_same_track(self):
        ts = TrackState()
        t1 = ts.update("cam_01", 2, (100, 100, 200, 200))
        t2 = ts.update("cam_01", 2, (105, 105, 205, 205))
        assert t1.track_id == t2.track_id  # same composite_id
        assert t1 is t2  # same object
