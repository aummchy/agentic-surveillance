import pytest
import time
from unittest.mock import patch, MagicMock
from agents.matching_agent import run_matching_from_embedding, MatchResult
from agents.memory import MemoryAgent
from agents.recognition import RecognitionAgent
from agents.decision_agent import decide
from conftest import make_match_result


class TestMatchingAgentIntegration:
    """Integration tests for matching_agent with mocked vector_search."""

    def test_matched_person_propagates_all_fields(self, mock_vector_search):
        mock_vector_search.return_value = {
            "matches": [
                {"person_id": "p1", "name": "Alice", "role": "visitor",
                 "similarity_score": 0.78, "tags": ["authorized"],
                 "image_url": "http://example.com/face.jpg",
                 "verified": True, "alert_level": "low"},
                {"person_id": "p2", "name": "Bob", "role": "visitor",
                 "similarity_score": 0.45, "tags": [],
                 "image_url": None, "verified": False, "alert_level": "low"},
            ],
            "top2": 0.45,
            "margin": 0.33,
        }
        result = run_matching_from_embedding([0.1] * 512)
        assert result.matched is True
        assert result.person_id == "p1"
        assert result.name == "Alice"
        assert result.similarity_score == 0.78
        assert result.second_best_similarity == 0.45
        assert result.margin == 0.33
        assert result.candidate_count == 2
        assert result.tags == ["authorized"]

    def test_no_match_returns_unmatched(self, mock_vector_search):
        mock_vector_search.return_value = {
            "matches": [], "top2": None, "margin": None,
        }
        result = run_matching_from_embedding([0.1] * 512)
        assert result.matched is False
        assert result.candidate_count == 0
        assert result.person_id is None

    def test_single_match_no_margin(self, mock_vector_search):
        mock_vector_search.return_value = {
            "matches": [
                {"person_id": "p1", "name": "Solo", "role": "visitor",
                 "similarity_score": 0.72, "tags": [], "image_url": None,
                 "verified": False, "alert_level": "low"},
            ],
            "top2": None, "margin": None,
        }
        result = run_matching_from_embedding([0.1] * 512)
        assert result.matched is True
        assert result.candidate_count == 1
        assert result.second_best_similarity is None
        assert result.margin is None

    def test_none_embedding_returns_unmatched(self, mock_vector_search):
        result = run_matching_from_embedding(None)
        assert result.matched is False
        assert mock_vector_search.called is False


class TestMemoryAgentIntegration:
    """Integration tests for MemoryAgent with mocked DB."""

    def test_returns_empty_for_no_person_id(self, mock_get_or_create_memory):
        agent = MemoryAgent()
        result = agent.run({
            "person_id": None,
            "camera_id": "cam_01",
            "similarity": 0.0,
            "status": "unknown",
        })
        assert result["visit_count"] == 0
        assert mock_get_or_create_memory.called is False

    def test_returns_visit_data_for_known_person(self, mock_get_or_create_memory):
        mock_get_or_create_memory.return_value = {
            "visit_count": 15,
            "last_seen": time.time() - 3600,
            "first_seen": time.time() - 604800,
            "avg_similarity": 0.82,
            "typical_hours": [9, 10, 14],
            "typical_cameras": ["cam_01"],
            "last_status": "known",
        }
        agent = MemoryAgent()
        result = agent.run({
            "person_id": "p1",
            "camera_id": "cam_01",
            "similarity": 0.80,
            "status": "known",
        })
        assert result["visit_count"] == 15
        assert result["is_known"] is True
        mock_get_or_create_memory.assert_called_once_with("p1")

    def test_confidence_boost_increases_with_visits(self, mock_get_or_create_memory):
        mock_get_or_create_memory.return_value = {
            "visit_count": 10,
            "last_seen": time.time() - 7200,
            "first_seen": time.time() - 1209600,
            "avg_similarity": 0.85,
            "typical_hours": [10, 11],
            "typical_cameras": ["cam_01"],
            "last_status": "known",
        }
        agent = MemoryAgent()
        result = agent.run({
            "person_id": "p1",
            "camera_id": "cam_01",
            "similarity": 0.80,
            "status": "known",
        })
        assert result["confidence_boost"] >= 5


class TestFullPipelineIntegration:
    """End-to-end tests: matching → memory → recognition → decision."""

    def test_authorized_person_no_alert(self, mock_vector_search, mock_get_or_create_memory):
        mock_vector_search.return_value = {
            "matches": [
                {"person_id": "p1", "name": "Alice", "role": "employee",
                 "similarity_score": 0.82, "tags": ["authorized"],
                 "image_url": None, "verified": True, "alert_level": "none"},
            ],
            "top2": None, "margin": None,
        }
        mock_get_or_create_memory.return_value = {
            "visit_count": 50, "last_seen": time.time() - 1800,
            "first_seen": time.time() - 31536000,
            "avg_similarity": 0.88, "typical_hours": [9, 10, 14],
            "typical_cameras": ["cam_01"], "last_status": "authorized",
        }

        from pipeline.models import Track
        track = Track(track_id="test_auth", first_seen=time.time() - 30,
                      last_seen=time.time(), person_box=(0, 0, 100, 200),
                      max_track_secs=300, total_frames_seen=45,
                      face_detected_once=True, best_face_score=0.9)

        match = run_matching_from_embedding([0.1] * 512)
        memory = MemoryAgent().run({
            "person_id": match.person_id, "camera_id": "cam_01",
            "similarity": match.similarity_score,
            "status": "known" if match.matched else "unknown",
        })
        recog = RecognitionAgent().run({
            "similarity": match.similarity_score,
            "is_masked": False,
            "face_quality": 0.9,
            "track_duration": time.time() - track.first_seen,
            "memory_context": memory,
            "margin": match.margin,
            "name": match.name,
            "track_id": track.track_id,
        })
        decision = decide(track, match, recog, memory)

        assert decision.status == "authorized"
        assert decision.should_alert is False

    def test_unknown_person_triggers_alert(self, mock_vector_search, mock_get_or_create_memory):
        mock_vector_search.return_value = {
            "matches": [], "top2": None, "margin": None,
        }
        mock_get_or_create_memory.return_value = {
            "visit_count": 0, "last_seen": None, "first_seen": None,
            "avg_similarity": 0.0, "typical_hours": [],
            "typical_cameras": [], "last_status": None,
        }

        from pipeline.models import Track
        track = Track(track_id="test_unknown", first_seen=time.time() - 10,
                      last_seen=time.time(), person_box=(0, 0, 100, 200),
                      max_track_secs=300, total_frames_seen=15,
                      face_detected_once=True, best_face_score=0.7)

        match = run_matching_from_embedding([0.1] * 512)
        memory = MemoryAgent().run({
            "person_id": None, "camera_id": "cam_01",
            "similarity": 0.0, "status": "unknown",
        })
        recog = RecognitionAgent().run({
            "similarity": 0.0, "is_masked": False,
            "face_quality": 0.7,
            "track_duration": time.time() - track.first_seen,
            "memory_context": memory,
            "track_id": track.track_id,
        })
        decision = decide(track, match, recog, memory)

        assert decision.status == "unknown"
        assert decision.should_alert is True
        assert decision.should_register is True

    def test_blacklist_triggers_critical_alert(self, mock_vector_search, mock_get_or_create_memory):
        mock_vector_search.return_value = {
            "matches": [
                {"person_id": "p_bad", "name": "Intruder", "role": "unknown",
                 "similarity_score": 0.75, "tags": ["blacklist"],
                 "image_url": None, "verified": False, "alert_level": "critical"},
            ],
            "top2": None, "margin": None,
        }
        mock_get_or_create_memory.return_value = {
            "visit_count": 1, "last_seen": time.time() - 86400,
            "first_seen": time.time() - 86400,
            "avg_similarity": 0.75, "typical_hours": [2],
            "typical_cameras": ["cam_01"], "last_status": "blacklist",
        }

        from pipeline.models import Track
        track = Track(track_id="test_black", first_seen=time.time() - 5,
                      last_seen=time.time(), person_box=(0, 0, 100, 200),
                      max_track_secs=300, total_frames_seen=10,
                      face_detected_once=True, best_face_score=0.8)

        match = run_matching_from_embedding([0.1] * 512)
        memory = MemoryAgent().run({
            "person_id": match.person_id, "camera_id": "cam_01",
            "similarity": match.similarity_score,
            "status": "known",
        })
        recog = RecognitionAgent().run({
            "similarity": match.similarity_score,
            "is_masked": False, "face_quality": 0.8,
            "track_duration": time.time() - track.first_seen,
            "memory_context": memory,
            "margin": match.margin,
            "name": match.name,
            "track_id": track.track_id,
        })
        decision = decide(track, match, recog, memory)

        assert decision.status == "blacklist"
        assert decision.should_alert is True
        assert decision.alert_level == "critical"

    def test_masked_unknown_triggers_alert(self, mock_vector_search, mock_get_or_create_memory):
        mock_vector_search.return_value = {
            "matches": [], "top2": None, "margin": None,
        }
        mock_get_or_create_memory.return_value = {
            "visit_count": 0, "last_seen": None, "first_seen": None,
            "avg_similarity": 0.0, "typical_hours": [],
            "typical_cameras": [], "last_status": None,
        }

        from pipeline.models import Track
        track = Track(track_id="test_masked", first_seen=time.time() - 10,
                      last_seen=time.time(), person_box=(0, 0, 100, 200),
                      max_track_secs=300, total_frames_seen=20,
                      face_detected_once=True, best_face_score=0.6, is_masked=True)

        match = run_matching_from_embedding([0.1] * 512)
        memory = MemoryAgent().run({
            "person_id": None, "camera_id": "cam_01",
            "similarity": 0.0, "status": "unknown",
        })
        recog = RecognitionAgent().run({
            "similarity": 0.0, "is_masked": True,
            "face_quality": 0.6,
            "track_duration": time.time() - track.first_seen,
            "memory_context": memory,
            "track_id": track.track_id,
        })
        decision = decide(track, match, recog, memory)

        assert decision.status == "masked_unknown"
        assert decision.should_alert is True
        assert decision.should_register is True

    def test_known_visitor_with_memory(self, mock_vector_search, mock_get_or_create_memory):
        mock_vector_search.return_value = {
            "matches": [
                {"person_id": "p_visitor", "name": "Bob", "role": "visitor",
                 "similarity_score": 0.65, "tags": [],
                 "image_url": None, "verified": False, "alert_level": "low"},
            ],
            "top2": None, "margin": None,
        }
        mock_get_or_create_memory.return_value = {
            "visit_count": 8, "last_seen": time.time() - 86400,
            "first_seen": time.time() - 2592000,
            "avg_similarity": 0.70, "typical_hours": [14, 15],
            "typical_cameras": ["cam_01"], "last_status": "known",
        }

        from pipeline.models import Track
        track = Track(track_id="test_visitor", first_seen=time.time() - 60,
                      last_seen=time.time(), person_box=(0, 0, 100, 200),
                      max_track_secs=300, total_frames_seen=90,
                      face_detected_once=True, best_face_score=0.85)

        match = run_matching_from_embedding([0.1] * 512)
        memory = MemoryAgent().run({
            "person_id": match.person_id, "camera_id": "cam_01",
            "similarity": match.similarity_score,
            "status": "known",
        })
        recog = RecognitionAgent().run({
            "similarity": match.similarity_score,
            "is_masked": False, "face_quality": 0.85,
            "track_duration": time.time() - track.first_seen,
            "memory_context": memory,
            "margin": match.margin,
            "name": match.name,
            "track_id": track.track_id,
        })
        decision = decide(track, match, recog, memory)

        assert decision.status == "known_visitor"
        assert decision.should_alert is False


class TestConfidenceScoringIntegration:
    """Verify confidence scoring formula produces expected values."""

    def test_high_similarity_produces_high_confidence(self):
        conf = RecognitionAgent().run({
            "similarity": 0.92, "is_masked": False,
            "face_quality": 0.9, "track_duration": 5.0,
            "memory_context": {"confidence_boost": 10},
            "margin": 0.25, "name": "Alice",
            "track_id": "test_high_conf",
        })
        assert conf["status"] == "known"
        assert conf["confidence"] >= 85

    def test_low_similarity_produces_low_confidence(self):
        conf = RecognitionAgent().run({
            "similarity": 0.20, "is_masked": False,
            "face_quality": 0.5, "track_duration": 1.0,
            "memory_context": {},
            "track_id": "test_low_conf",
        })
        assert conf["status"] == "unknown"
        assert conf["confidence"] < 40

    def test_confidence_in_range(self):
        for sim, qual, dur, mem in [
            (0.0, 0.0, 0.0, 0),
            (0.30, 0.8, 0.5, 0),
            (0.50, 0.8, 1.0, 0),
            (0.65, 0.9, 1.5, 5),
            (0.80, 1.0, 2.0, 10),
            (0.92, 0.8, 1.0, 0),
        ]:
            conf = RecognitionAgent().run({
                "similarity": sim, "is_masked": False,
                "face_quality": qual, "track_duration": dur,
                "memory_context": {"confidence_boost": mem},
                "track_id": "test_range",
            })
            assert 1 <= conf["confidence"] <= 100, (
                f"sim={sim}: confidence {conf['confidence']} not in [1, 100]"
            )

    def test_masked_confidence_lower_than_unmasked(self):
        unmasked = RecognitionAgent().run({
            "similarity": 0.55, "is_masked": False,
            "face_quality": 0.8, "track_duration": 2.0,
            "memory_context": {}, "track_id": "test_mask_a",
        })
        masked = RecognitionAgent().run({
            "similarity": 0.55, "is_masked": True,
            "face_quality": 0.8, "track_duration": 2.0,
            "memory_context": {}, "track_id": "test_mask_b",
        })
        assert masked["confidence"] < unmasked["confidence"]
