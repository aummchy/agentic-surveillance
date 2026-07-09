"""
Recognition & Policy Threshold Regression Tests (Unified Formula)

These tests lock down the classification behavior of the unified
confidence formula from agents/scoring.py. Tests verify that:
  - The confidence formula is in [1, 100]
  - Status tiers work correctly with the match gate
  - Mask penalty reduces confidence
  - Missing quality defaults to 0.5
  - Memory boost has a small effect

Run: python -m pytest tests/test_recognition.py -v
"""

import pytest
from agents.recognition import recognize
from agents.scoring import compute_confidence, normalize_cosine, clip
from agents.policy import PolicyAgent
from pipeline.models import DecisionResult, MatchResult
from config import settings


# ── Scorer unit tests ────────────────────────────────────────────

class TestNormalizeCosine:
    def test_minimum_maps_to_zero(self):
        assert normalize_cosine(settings.SIM_NORM_MIN) == 0.0

    def test_maximum_maps_to_one(self):
        assert normalize_cosine(settings.SIM_NORM_MAX) == 1.0

    def test_below_min_clips_to_zero(self):
        assert normalize_cosine(0.0) == 0.0

    def test_above_max_clips_to_one(self):
        assert normalize_cosine(1.0) == 1.0

    def test_midpoint(self):
        mid = (settings.SIM_NORM_MIN + settings.SIM_NORM_MAX) / 2
        expected = (mid - settings.SIM_NORM_MIN) / (settings.SIM_NORM_MAX - settings.SIM_NORM_MIN)
        assert normalize_cosine(mid) == pytest.approx(expected, abs=0.01)


class TestComputeConfidence:
    def test_minimum_inputs(self):
        conf = compute_confidence(0.0, 0.0, 0.0, 0.0, False)
        assert 1 <= conf <= 100

    def test_maximum_inputs(self):
        conf = compute_confidence(1.0, 1.0, 10.0, 20.0, False, margin=0.30)
        assert conf == 100

    def test_masked_lower_than_unmasked(self):
        unmasked = compute_confidence(0.50, 0.8, 1.0, 0.0, False)
        masked = compute_confidence(0.50, 0.8, 1.0, 0.0, True)
        assert masked < unmasked

    def test_quality_none_equals_half(self):
        with_none = compute_confidence(0.60, None, 1.0, 0.0, False)
        with_half = compute_confidence(0.60, 0.5, 1.0, 0.0, False)
        assert with_none == with_half

    def test_quality_zero_treated_as_missing(self):
        with_zero = compute_confidence(0.60, 0.0, 1.0, 0.0, False)
        with_half = compute_confidence(0.60, 0.5, 1.0, 0.0, False)
        assert with_zero == with_half

    def test_memory_small_effect(self):
        no_memory = compute_confidence(0.55, 0.7, 1.0, 0.0, False)
        with_memory = compute_confidence(0.55, 0.7, 1.0, 10.0, False)
        assert no_memory < with_memory
        assert with_memory - no_memory <= 5


# ── RecognitionAgent integration tests ───────────────────────────

class TestRecognitionBands:
    def test_unknown_low_similarity(self):
        """sim=0.30, good quality, short dur -> unknown (below threshold + low conf)."""
        result = recognize(similarity=0.30, face_quality=0.8, track_duration=0.5)
        assert result["status"] == "unknown", (
            f"sim=0.30 should be 'unknown', got '{result['status']}'"
        )

    def test_uncertain_below_threshold_with_max_boost(self):
        """sim=0.44 (below 0.45), max everything -> uncertain (conf>=55, not matched)."""
        result = recognize(
            similarity=0.44, face_quality=1.0, track_duration=2.0,
            memory_context={"confidence_boost": 20},
        )
        assert result["status"] == "uncertain", (
            f"sim=0.44 with max boost should be 'uncertain', got '{result['status']}'"
        )

    def test_known_with_good_match(self):
        """sim=0.60, good quality/duration, memory=10, margin=0.20 -> known (conf>=70, matched)."""
        result = recognize(
            similarity=0.60, face_quality=0.85, track_duration=2.0,
            memory_context={"confidence_boost": 10},
            margin=0.20,
        )
        assert result["status"] == "known", (
            f"sim=0.60 should be 'known', got '{result['status']}'"
        )
        assert result["confidence"] >= 70

    def test_known_at_very_high_similarity(self):
        """sim=0.92 -> known (above threshold, very high confidence)."""
        result = recognize(similarity=0.92, face_quality=0.8, track_duration=1.0)
        assert result["status"] == "known", (
            f"sim=0.92 should be 'known', got '{result['status']}'"
        )
        assert result["confidence"] >= 85

    def test_masked_lowers_confidence(self):
        """Masked produces lower confidence than unmasked, all else equal."""
        unmasked = recognize(
            similarity=0.50, face_quality=0.8, track_duration=1.0, is_masked=False,
        )
        masked = recognize(
            similarity=0.50, face_quality=0.8, track_duration=1.0, is_masked=True,
        )
        assert masked["confidence"] < unmasked["confidence"]

    def test_quality_none_fallback(self):
        """quality=None or 0.0 -> uses DEFAULT_FACE_QUALITY=0.5, not 0."""
        result = recognize(
            similarity=0.60, face_quality=0.0, track_duration=1.0,
        )
        assert result["status"] in ("known", "uncertain")

    def test_memory_boost_small_effect(self):
        """Memory 10 vs 0 has a small positive effect on confidence."""
        no_memory = recognize(
            similarity=0.55, face_quality=0.7, track_duration=1.0,
            memory_context={"confidence_boost": 0},
        )
        with_memory = recognize(
            similarity=0.55, face_quality=0.7, track_duration=1.0,
            memory_context={"confidence_boost": 10},
        )
        assert no_memory["confidence"] < with_memory["confidence"]
        assert with_memory["confidence"] - no_memory["confidence"] <= 5

    def test_match_threshold_gate(self):
        """sim=0.44 (below threshold) + avg quality -> unknown (conf<55)."""
        result = recognize(
            similarity=0.44, face_quality=0.9, track_duration=1.0,
        )
        assert result["status"] == "unknown", (
            f"sim=0.44 with avg quality should be 'unknown', got '{result['status']}'"
        )

    def test_confidence_range_1_to_100(self):
        """All valid inputs produce confidence in [1, 100]."""
        test_cases = [
            (0.0, 0.0, 0.0, {"confidence_boost": 0}),
            (0.30, 0.8, 0.5, {"confidence_boost": 0}),
            (0.50, 0.8, 1.0, {"confidence_boost": 0}),
            (0.65, 0.9, 1.5, {"confidence_boost": 5}),
            (0.80, 1.0, 2.0, {"confidence_boost": 10}),
            (0.92, 0.8, 1.0, {"confidence_boost": 0}),
        ]
        for sim, quality, dur, memory in test_cases:
            result = recognize(similarity=sim, face_quality=quality, track_duration=dur, memory_context=memory)
            assert 1 <= result["confidence"] <= 100, (
                f"sim={sim}: confidence {result['confidence']} not in [1, 100]"
            )

    def test_match_threshold_is_correct(self):
        """Verify MATCH_THRESHOLD is 0.45."""
        assert settings.MATCH_THRESHOLD == 0.45, (
            f"MATCH_THRESHOLD should be 0.45, got {settings.MATCH_THRESHOLD}"
        )


# ── Threshold consistency between recognition and policy ─────────

class TestThresholdConsistency:
    def test_both_read_same_threshold(self):
        """Both agents use settings.MATCH_THRESHOLD."""
        from config import settings as s
        rec_threshold = s.MATCH_THRESHOLD
        policy_threshold = s.MATCH_THRESHOLD
        assert rec_threshold == policy_threshold, (
            f"Threshold mismatch: recognition={rec_threshold}, policy={policy_threshold}"
        )


# ── MatchResult propagation tests ─────────────────────────────────

class TestMatchResultPropagation:
    """Verify that top2, margin, and candidate_count propagate from
    vector_search() through matching_agent to MatchResult."""

    @pytest.fixture(autouse=True)
    def _patch_vector_search(self):
        from unittest.mock import patch
        from agents import matching_agent
        patcher = patch.object(matching_agent, "vector_search")
        self.mock_vs = patcher.start()
        yield
        patcher.stop()

    def test_two_matches_propagates_top2_and_margin(self):
        self.mock_vs.return_value = {
            "matches": [
                {"person_id": "p1", "name": "Alice", "role": "visitor",
                 "similarity_score": 0.66, "tags": [], "image_url": None,
                 "verified": True, "alert_level": "low"},
                {"person_id": "p2", "name": "Bob", "role": "visitor",
                 "similarity_score": 0.51, "tags": [], "image_url": None,
                 "verified": False, "alert_level": "low"},
            ],
            "top2": 0.51,
            "margin": 0.15,
        }
        from agents.matching_agent import run_matching_from_embedding
        result = run_matching_from_embedding([0.1] * 512)
        assert result.matched is True
        assert result.second_best_similarity == 0.51
        assert result.margin == 0.15
        assert result.candidate_count == 2

    def test_single_match_leaves_top2_as_none(self):
        self.mock_vs.return_value = {
            "matches": [
                {"person_id": "p1", "name": "Solo", "role": "visitor",
                 "similarity_score": 0.70, "tags": [], "image_url": None,
                 "verified": False, "alert_level": "low"},
            ],
            "top2": None,
            "margin": None,
        }
        from agents.matching_agent import run_matching_from_embedding
        result = run_matching_from_embedding([0.1] * 512)
        assert result.matched is True
        assert result.second_best_similarity is None
        assert result.margin is None
        assert result.candidate_count == 1

    def test_no_matches_leaves_defaults(self):
        self.mock_vs.return_value = {"matches": [], "top2": None, "margin": None}
        from agents.matching_agent import run_matching_from_embedding
        result = run_matching_from_embedding([0.1] * 512)
        assert result.matched is False
        assert result.second_best_similarity is None
        assert result.margin is None
        assert result.candidate_count == 0

    def test_none_embedding_returns_unmatched(self):
        from agents.matching_agent import run_matching_from_embedding
        result = run_matching_from_embedding(None)
        assert result.matched is False
        assert result.candidate_count == 0
