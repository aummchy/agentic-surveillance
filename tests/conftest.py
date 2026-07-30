import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from pipeline.models import Track


@pytest.fixture
def sample_embedding():
    return [float(v) for v in np.random.RandomState(42).randn(512).tolist()]


@pytest.fixture
def sample_track():
    return Track(
        track_id="test_cam_1000_1",
        first_seen=1000.0,
        last_seen=1010.0,
        person_box=(50, 100, 200, 400),
        best_face_score=0.85,
        max_track_secs=300,
        total_frames_seen=30,
        face_detected_once=True,
        embedding=[float(v) for v in np.random.RandomState(1).randn(512).tolist()],
    )


@pytest.fixture
def mock_vector_search():
    with patch("agents.matching_agent.vector_search") as mock:
        mock.return_value = {
            "matches": [],
            "top2": None,
            "margin": None,
        }
        yield mock


@pytest.fixture
def mock_get_or_create_memory():
    with patch("agents.memory.get_or_create_memory") as mock:
        mock.return_value = {
            "visit_count": 0,
            "last_seen": None,
            "first_seen": None,
            "avg_similarity": 0.0,
            "typical_hours": [],
            "typical_cameras": [],
            "last_status": None,
        }
        yield mock


@pytest.fixture
def mock_update_visit_memory():
    with patch("agents.memory.update_visit_memory") as mock:
        yield mock


def make_match_result(
    person_id="p1", name="Alice", role="visitor", tags=None,
    similarity=0.66, matched=True, verified=False, alert_level="low",
    top2=0.51, margin=0.15, candidate_count=2,
):
    from agents.matching_agent import MatchResult
    if tags is None:
        tags = []
    return MatchResult(
        person_id=person_id,
        name=name,
        role=role,
        tags=tags,
        similarity_score=similarity,
        matched=matched,
        verified=verified,
        alert_level=alert_level,
        second_best_similarity=top2,
        margin=margin,
        candidate_count=candidate_count,
    )
