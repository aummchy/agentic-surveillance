import pytest
import time
from unittest.mock import patch, MagicMock, call
from pipeline.models import Track, DecisionResult
from agents.policy import decide
from agents.memory import MemoryAgent
from agents.alert_agent import dispatch, should_send_alert
from config.status import Status


class TestFinalizationDecisionChain:
    """Tests for the decision chain executed during track finalization."""

    def test_known_person_skips_registration(self, mock_vector_search, mock_get_or_create_memory):
        mock_vector_search.return_value = {
            "matches": [
                {"person_id": "p1", "name": "Alice", "role": "employee",
                 "similarity_score": 0.85, "tags": ["authorized"],
                 "image_url": None, "verified": True, "alert_level": "none"},
            ],
            "top2": None, "margin": None,
        }
        mock_get_or_create_memory.return_value = {
            "visit_count": 100, "last_seen": time.time() - 3600,
            "first_seen": time.time() - 63072000,
            "avg_similarity": 0.90, "typical_hours": [8, 9, 10],
            "typical_cameras": ["cam_01"], "last_status": Status.AUTHORIZED,
        }

        from agents.matching_agent import run_matching_from_embedding
        track = Track(track_id="test_final_auth", first_seen=time.time() - 60,
                      last_seen=time.time(), person_box=(0, 0, 100, 200),
                      max_track_secs=300, best_face_score=0.9,
                      embedding=[0.1] * 512, face_detected_once=True,
                      total_frames_seen=90)

        match = run_matching_from_embedding(track.embedding)
        memory = MemoryAgent().run({
            "person_id": match.person_id, "camera_id": "cam_01",
            "similarity": match.similarity_score,
            "status": Status.KNOWN if match.matched else Status.UNKNOWN,
        })
        from agents.recognition import RecognitionAgent
        recog = RecognitionAgent().run({
            "similarity": match.similarity_score,
            "is_masked": False, "face_quality": track.best_face_score,
            "track_duration": time.time() - track.first_seen,
            "memory_context": memory,
            "margin": match.margin, "name": match.name,
            "track_id": track.track_id,
        })
        decision = decide(track, match, recog, memory)

        assert decision.status == Status.AUTHORIZED
        assert decision.should_register is False

    def test_unknown_person_triggers_registration(self, mock_vector_search, mock_get_or_create_memory):
        mock_vector_search.return_value = {
            "matches": [], "top2": None, "margin": None,
        }
        mock_get_or_create_memory.return_value = {
            "visit_count": 0, "last_seen": None, "first_seen": None,
            "avg_similarity": 0.0, "typical_hours": [],
            "typical_cameras": [], "last_status": None,
        }

        from agents.matching_agent import run_matching_from_embedding
        track = Track(track_id="test_final_unknown", first_seen=time.time() - 15,
                      last_seen=time.time(), person_box=(0, 0, 100, 200),
                      max_track_secs=300, best_face_score=0.7,
                      embedding=[0.1] * 512, face_detected_once=True,
                      total_frames_seen=20)

        match = run_matching_from_embedding(track.embedding)
        memory = MemoryAgent().run({
            "person_id": None, "camera_id": "cam_01",
            "similarity": 0.0,             "status": Status.UNKNOWN,
        })
        from agents.recognition import RecognitionAgent
        recog = RecognitionAgent().run({
            "similarity": 0.0, "is_masked": False,
            "face_quality": track.best_face_score,
            "track_duration": time.time() - track.first_seen,
            "memory_context": memory,
            "track_id": track.track_id,
        })
        decision = decide(track, match, recog, memory)

        assert decision.status == Status.UNKNOWN
        assert decision.should_register is True


class TestRegistrationDedup:
    """Tests for the auto-registration dedup logic."""

    @patch("utils.db_utils.find_similar_unknowns")
    @patch("utils.db_utils.update_face")
    def test_finds_and_merges_similar_unknown(self, mock_update, mock_find):
        mock_find.return_value = [
            {"person_id": "existing_unknown_1", "similarity_score": 0.82},
        ]
        track = Track(track_id="test_dedup", first_seen=time.time() - 10,
                      last_seen=time.time(), person_box=(0, 0, 100, 200),
                      max_track_secs=300, best_face_score=0.75,
                      embedding=[0.1] * 512, face_detected_once=True,
                      total_frames_seen=15)

        from utils.db_utils import find_similar_unknowns, update_face
        similar = find_similar_unknowns(track.embedding)
        assert len(similar) > 0
        assert similar[0]["person_id"] == "existing_unknown_1"

        if similar:
            existing_id = similar[0]["person_id"]
            update_face(existing_id, image_url="/captures/test.jpg",
                        embedding=track.embedding,
                        quality_score=track.best_face_score)
            mock_update.assert_called_once()

    @patch("utils.db_utils.store_face")
    def test_registers_new_unknown_when_no_similar(self, mock_store):
        mock_store.return_value = "new_person_id"
        track = Track(track_id="test_new_reg", first_seen=time.time() - 10,
                      last_seen=time.time(), person_box=(0, 0, 100, 200),
                      max_track_secs=300, best_face_score=0.8,
                      embedding=[0.1] * 512, face_detected_once=True,
                      total_frames_seen=20)

        from utils.db_utils import store_face
        stored_id = store_face(
            person_id=track.track_id, name="Unknown", role="unknown",
            embedding=track.embedding, image_url="/captures/test.jpg",
            tags=["auto_registered"], camera_id="cam_01",
            quality_score=track.best_face_score,
        )
        assert stored_id == "new_person_id"
        mock_store.assert_called_once()


class TestVisitRecording:
    """Tests for visit recording during finalization."""

    @patch("agents.memory.update_visit_memory")
    def test_records_visit_for_known_person(self, mock_update):
        mock_update.return_value = {"visit_count": 11}
        agent = MemoryAgent()
        agent.record_visit(
            person_id="p1", camera_id="cam_01",
            status=Status.KNOWN_VISITOR, similarity=0.75,
            is_masked=False, visit_action="recorded",
        )
        mock_update.assert_called_once()

    @patch("agents.memory.update_visit_memory")
    def test_no_visit_for_none_person(self, mock_update):
        agent = MemoryAgent()
        agent.record_visit(
            person_id=None, camera_id="cam_01",
            status=Status.UNKNOWN, similarity=0.0,
            is_masked=False, visit_action="recorded",
        )
        assert mock_update.called is False


class TestAlertDispatchConditions:
    """Tests for alert dispatch decisions during finalization."""

    def test_alert_dispatched_for_blacklist(self):
        decision = DecisionResult(
            status=Status.BLACKLIST, alert_level="critical",
            should_alert=True, person_id="p_bad", name="Intruder",
            reason="Blacklisted person detected",
        )
        assert decision.should_alert is True
        assert decision.alert_level == "critical"

    def test_no_alert_for_authorized(self):
        decision = DecisionResult(
            status=Status.AUTHORIZED, alert_level="none",
            should_alert=False, person_id="p1", name="Alice",
            reason="Authorized person",
        )
        assert decision.should_alert is False

    def test_alert_for_masked_unknown_loitering(self):
        decision = DecisionResult(
            status=Status.MASKED_UNKNOWN, alert_level="high",
            should_alert=True, name="Masked Person",
            reason="Masked unknown person loitering",
        )
        assert decision.should_alert is True
        assert decision.alert_level == "high"
