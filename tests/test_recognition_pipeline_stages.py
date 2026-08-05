"""
Tests for the extracted RecognitionPipeline (pipeline/recognition_pipeline.py).
Run: python -m pytest tests/test_recognition_pipeline_stages.py -v
"""

import time
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from pipeline.recognition_pipeline import RecognitionPipeline, PipelineResult, RecognitionMetrics
from pipeline.models import Track, QualityResult, MatchResult, DecisionResult
from config.status import Status

_EMBEDDING_512 = np.random.rand(512).astype(np.float32)


def _track(**kw):
    defaults = dict(
        track_id="cam_01_123456_42", first_seen=time.time() - 2.0,
        last_seen=time.time(), person_box=(100, 200, 300, 500),
    )
    defaults.update(kw)
    return Track(**defaults)


def _frame(h=480, w=640):
    return np.zeros((h, w, 3), dtype=np.uint8)


def _make_match(similarity=0.65, name="aum"):
    return MatchResult(
        matched=True, person_id="p_aum", name=name,
        similarity_score=similarity, verified=True,
        alert_level="low", second_best_similarity=0.41,
        margin=0.24, candidate_count=2,
    )


def _face(embedding=None):
    return [{
        "bbox": (10, 10, 100, 120), "det_score": 0.95,
        "embedding": embedding if embedding is not None else _EMBEDDING_512.copy(),
        "is_masked": False,
    }]


class TestPipelineHighConfidenceSkip:
    def test_skips_when_above_threshold(self):
        pipe = RecognitionPipeline(matching_fn=MagicMock())
        track = _track(confidence=95)
        track.pending_match_result = MatchResult(matched=True, similarity_score=0.92)
        result = pipe.run(_frame(), track)
        assert result.skip_reason == "high_confidence"
        assert result.decision is None


class TestPipelineFaceDetection:
    def test_empty_crop_skips(self):
        pipe = RecognitionPipeline(matching_fn=MagicMock())
        track = _track()
        with patch("pipeline.recognition_pipeline.crop_person",
                   return_value=np.array([], dtype=np.uint8)):
            result = pipe.run(_frame(), track)
            assert result.skip_reason == "no_face"

    def test_no_faces_skips(self):
        pipe = RecognitionPipeline(matching_fn=MagicMock())
        track = _track()
        mock_app = MagicMock()
        mock_app.detect_faces_raw.return_value = []
        with (
            patch("pipeline.recognition_pipeline.crop_person",
                  return_value=np.zeros((200, 150, 3), dtype=np.uint8)),
            patch("pipeline.recognition_pipeline.get_insightface",
                  return_value=mock_app),
        ):
            result = pipe.run(_frame(), track)
            assert result.skip_reason == "no_face"


class TestPipelineQualityGate:
    @pytest.fixture(autouse=True)
    def _patch_face(self):
        with (
            patch("pipeline.recognition_pipeline.crop_person",
                  return_value=np.zeros((200, 150, 3), dtype=np.uint8)),
            patch("pipeline.recognition_pipeline.get_insightface") as mock_if,
        ):
            mock_if.return_value.detect_faces_raw.return_value = _face()
            yield

    def test_invalid_quality_skips(self):
        pipe = RecognitionPipeline(matching_fn=MagicMock())
        track = _track()
        with patch("pipeline.recognition_pipeline.compute_quality",
                   return_value=QualityResult.invalid()):
            result = pipe.run(_frame(), track)
            assert result.skip_reason == "low_quality"
            assert result.quality is not None
            assert result.quality.is_valid is False


class TestPipelineFullFlow:
    @pytest.fixture(autouse=True)
    def _patch_detection(self):
        with (
            patch("pipeline.recognition_pipeline.crop_person",
                  return_value=np.zeros((200, 150, 3), dtype=np.uint8)),
            patch("pipeline.recognition_pipeline.get_insightface") as mock_if,
            patch("pipeline.recognition_pipeline.compute_quality",
                  return_value=QualityResult(
                      blur_score=150, brightness=140, face_area=10000,
                      is_valid=True, overall_score=0.85)),
        ):
            mock_if.return_value.detect_faces_raw.return_value = _face()
            yield

    def test_known_visitor(self):
        mock_match = MagicMock(return_value=_make_match())
        mock_mem = MagicMock(return_value={"visit_count": 12, "is_known": True, "confidence_boost": 5})
        mock_recog = MagicMock(return_value={"status": Status.KNOWN, "confidence": 87, "similarity": 0.65})
        mock_decide = MagicMock(return_value=DecisionResult(
            status=Status.VERIFIED, alert_level="low", should_alert=False))

        pipe = RecognitionPipeline(
            matching_fn=mock_match,
            memory_agent=MagicMock(run=mock_mem),
            recognition_agent=MagicMock(run=mock_recog),
            decide_fn=mock_decide,
        )
        result = pipe.run(_frame(), _track())
        assert result.skip_reason == "success"
        assert result.match and result.match.name == "aum"
        assert result.recognition and result.recognition["status"] == Status.KNOWN
        assert result.decision and result.decision.status == Status.VERIFIED
        assert result.memory and result.memory["visit_count"] == 12
        assert result.embedding is not None
        assert result.quality and result.quality.is_valid

    def test_no_match(self):
        mock_match = MagicMock(return_value=MatchResult(matched=False))
        mock_recog = MagicMock(return_value={"status": Status.UNKNOWN, "confidence": 15, "similarity": 0.0})
        mock_decide = MagicMock(return_value=DecisionResult(status=Status.UNKNOWN, alert_level="none"))

        pipe = RecognitionPipeline(
            matching_fn=mock_match,
            recognition_agent=MagicMock(run=mock_recog),
            decide_fn=mock_decide,
        )
        result = pipe.run(_frame(), _track())
        assert result.skip_reason == "success"
        assert result.match and result.match.matched is False
        assert result.recognition["status"] == Status.UNKNOWN
        assert result.decision.status == Status.UNKNOWN

    def test_injected_agents_are_called(self):
        mock_match = MagicMock(return_value=_make_match())
        mock_mem = MagicMock(return_value={"visit_count": 5, "is_known": True})
        mock_recog = MagicMock(return_value={"status": Status.KNOWN, "confidence": 80})
        mock_decide = MagicMock(return_value=DecisionResult(status=Status.VERIFIED, alert_level="low"))

        pipe = RecognitionPipeline(
            matching_fn=mock_match,
            recognition_agent=MagicMock(run=mock_recog),
            memory_agent=MagicMock(run=mock_mem),
            decide_fn=mock_decide,
        )
        result = pipe.run(_frame(), _track())
        assert result.skip_reason == "success"
        mock_match.assert_called_once()
        mock_recog.assert_called_once()
        mock_mem.assert_called_once()
        mock_decide.assert_called_once()

    def test_metrics_populated(self):
        mock_match = MagicMock(return_value=_make_match())
        mock_mem = MagicMock(return_value={"visit_count": 1, "is_known": True, "confidence_boost": 0})
        mock_recog = MagicMock(return_value={"status": Status.KNOWN, "confidence": 70, "similarity": 0.65})
        mock_decide = MagicMock(return_value=DecisionResult(status=Status.VERIFIED, alert_level="low"))

        pipe = RecognitionPipeline(
            matching_fn=mock_match,
            memory_agent=MagicMock(run=mock_mem),
            recognition_agent=MagicMock(run=mock_recog),
            decide_fn=mock_decide,
        )
        result = pipe.run(_frame(), _track())
        assert result.metrics.total_ms > 0
        for attr in ("crop_detect_ms", "fallback_detect_ms", "quality_ms",
                     "embed_ms", "db_ms", "memory_ms", "recog_policy_ms"):
            assert getattr(result.metrics, attr) >= 0, f"metrics.{attr} < 0"


class TestPipelineCache:
    @pytest.fixture(autouse=True)
    def _patch_detection(self):
        with (
            patch("pipeline.recognition_pipeline.crop_person",
                  return_value=np.zeros((200, 150, 3), dtype=np.uint8)),
            patch("pipeline.recognition_pipeline.get_insightface") as mock_if,
            patch("pipeline.recognition_pipeline.compute_quality",
                  return_value=QualityResult(
                      blur_score=150, brightness=140, face_area=10000,
                      is_valid=True, overall_score=0.85)),
        ):
            yield mock_if

    def test_always_calls_atlas(self, _patch_detection):
        mock_match = MagicMock(return_value=_make_match())
        pipe = RecognitionPipeline(matching_fn=mock_match)
        track = _track()
        _patch_detection.return_value.detect_faces_raw.return_value = _face()

        result = pipe.run(_frame(), track)
        assert result.skip_reason == "success"
        mock_match.assert_called_once()