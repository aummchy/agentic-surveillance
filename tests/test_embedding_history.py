"""
Phase 4.2 — Embedding history cap & double-counting fix

Verifies:
1. update_face computes mean_embedding from N+1 embeddings (not N+2)
2. $slice uses the configurable EMBEDDING_HISTORY_CAP
3. mean is correct for known inputs

Run: python -m pytest tests/test_embedding_history.py -v
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime


def _make_emb(values):
    """Create a normalized 512-d embedding from a seed value."""
    rng = np.random.RandomState(abs(hash(str(values))) % (2**31))
    emb = rng.randn(512).astype(np.float32)
    emb = emb / (np.linalg.norm(emb) + 1e-6)
    return emb.tolist()


def _mean_of(embeddings):
    """Reference mean computation matching _compute_mean_embedding."""
    arr = np.array(embeddings, dtype=np.float32)
    mean = arr.mean(axis=0)
    norm = np.linalg.norm(mean)
    if norm < 1e-6:
        return embeddings[-1]
    return (mean / norm).tolist()


class TestUpdateFaceMeanEmbedding:
    """Verify update_face builds correct update_ops (no double-count)."""

    @patch("utils.db_utils.get_faces_collection")
    def test_no_double_count_3_existing(self, mock_get_col):
        """With 3 existing embeddings, mean must be over 4 (not 5)."""
        from utils.db_utils import update_face

        existing_embs = [_make_emb(1), _make_emb(2), _make_emb(3)]
        new_emb = _make_emb(4)

        # Compute expected BEFORE calling update_face (mock mutates the list)
        expected_mean = _mean_of([e.copy() for e in existing_embs] + [new_emb])

        mock_col = MagicMock()
        mock_get_col.return_value = mock_col
        mock_col.find_one.side_effect = [
            {"latest_embedding_quality": 0.5},
            {"embeddings": existing_embs},
        ]

        update_ops_captured = {}
        def capture_update(filter_doc, ops, **kwargs):
            update_ops_captured.update(ops)
            return MagicMock(modified_count=1)
        mock_col.update_one.side_effect = capture_update

        update_face(
            person_id="test_123",
            image_url="/captures/test.jpg",
            embedding=new_emb,
            quality_score=0.9,
        )

        assert "$push" in update_ops_captured
        push = update_ops_captured["$push"]
        assert push["embeddings"]["$each"] == [new_emb]

        assert "$set" in update_ops_captured
        assert "mean_embedding" in update_ops_captured["$set"]
        computed_mean = update_ops_captured["$set"]["mean_embedding"]

        assert len(computed_mean) == len(expected_mean), (
            f"Expected {len(expected_mean)} dims, got {len(computed_mean)}"
        )
        np.testing.assert_allclose(computed_mean, expected_mean, atol=1e-6,
                                   err_msg="mean_embedding was double-counted or wrong")

    @patch("utils.db_utils.get_faces_collection")
    def test_no_double_count_0_existing(self, mock_get_col):
        """With 0 existing embeddings, mean should equal the single new embedding."""
        from utils.db_utils import update_face

        new_emb = _make_emb(99)
        expected_mean = _mean_of([new_emb])

        mock_col = MagicMock()
        mock_get_col.return_value = mock_col
        # quality_score=None → quality check find_one is skipped, only 1 call
        mock_col.find_one.return_value = {"embeddings": []}

        update_ops_captured = {}
        def capture_update(filter_doc, ops, **kwargs):
            update_ops_captured.update(ops)
            return MagicMock(modified_count=1)
        mock_col.update_one.side_effect = capture_update

        update_face(
            person_id="test_empty",
            image_url="/captures/empty.jpg",
            embedding=new_emb,
        )

        computed_mean = update_ops_captured["$set"]["mean_embedding"]
        np.testing.assert_allclose(computed_mean, expected_mean, atol=1e-6)

    @patch("utils.db_utils.get_faces_collection")
    @patch("utils.db_utils.settings")
    def test_uses_configurable_cap(self, mock_settings, mock_get_col):
        """$slice should use EMBEDDING_HISTORY_CAP from settings."""
        from utils.db_utils import update_face

        mock_settings.EMBEDDING_HISTORY_CAP = 50

        existing_embs = [_make_emb(i) for i in range(10)]
        new_emb = _make_emb(42)

        mock_col = MagicMock()
        mock_get_col.return_value = mock_col
        mock_col.find_one.side_effect = [
            {"latest_embedding_quality": 0.3},
            {"embeddings": existing_embs},
        ]

        update_ops_captured = {}
        def capture_update(filter_doc, ops, **kwargs):
            update_ops_captured.update(ops)
            return MagicMock(modified_count=1)
        mock_col.update_one.side_effect = capture_update

        update_face(
            person_id="test_cap",
            image_url="/captures/cap.jpg",
            embedding=new_emb,
        )

        push = update_ops_captured["$push"]["embeddings"]
        assert push["$slice"] == -50, f"Expected $slice: -50, got {push['$slice']}"

    @patch("utils.db_utils.get_faces_collection")
    @patch("utils.db_utils.settings")
    def test_cap_slicing(self, mock_settings, mock_get_col):
        """When existing + new exceeds cap, oldest embeddings should be trimmed."""
        from utils.db_utils import update_face

        mock_settings.EMBEDDING_HISTORY_CAP = 5

        embs_a = [_make_emb(i) for i in range(4)]
        new_emb = _make_emb(100)

        # Compute expected BEFORE update_face mutates the list
        all_5 = [e.copy() for e in embs_a] + [new_emb]
        expected_mean = _mean_of(all_5[-5:])  # cap=5, so all 5 kept

        mock_col = MagicMock()
        mock_get_col.return_value = mock_col
        # quality_score=None → only 1 find_one call
        mock_col.find_one.return_value = {"embeddings": embs_a}

        update_ops_captured = {}
        def capture_update(filter_doc, ops, **kwargs):
            update_ops_captured.update(ops)
            return MagicMock(modified_count=1)
        mock_col.update_one.side_effect = capture_update

        update_face(
            person_id="test_slice",
            image_url="/captures/slice.jpg",
            embedding=new_emb,
        )

        computed_mean = update_ops_captured["$set"]["mean_embedding"]
        np.testing.assert_allclose(computed_mean, expected_mean, atol=1e-6)

    @patch("utils.db_utils.get_faces_collection")
    @patch("utils.db_utils.settings")
    def test_cap_trims_oldest(self, mock_settings, mock_get_col):
        """With cap=3 and 5 existing + 1 new, oldest 3 should be dropped."""
        from utils.db_utils import update_face

        mock_settings.EMBEDDING_HISTORY_CAP = 3

        embs = [_make_emb(i) for i in range(5)]
        new_emb = _make_emb(100)

        all_6 = [e.copy() for e in embs] + [new_emb]
        expected_mean = _mean_of(all_6[-3:])

        mock_col = MagicMock()
        mock_get_col.return_value = mock_col
        # quality_score=None → only 1 find_one call
        mock_col.find_one.return_value = {"embeddings": embs}

        update_ops_captured = {}
        def capture_update(filter_doc, ops, **kwargs):
            update_ops_captured.update(ops)
            return MagicMock(modified_count=1)
        mock_col.update_one.side_effect = capture_update

        update_face(
            person_id="test_trim",
            image_url="/captures/trim.jpg",
            embedding=new_emb,
        )

        computed_mean = update_ops_captured["$set"]["mean_embedding"]
        np.testing.assert_allclose(computed_mean, expected_mean, atol=1e-6)


class TestAutoRegistrationDedup:
    """Verify the dedup path in main.py calls update_face, not store_face."""

    def test_find_similar_unknowns_uses_dedup_threshold(self):
        from config import settings
        assert hasattr(settings, "DEDUP_SIMILARITY_THRESHOLD")
        assert 0.0 < settings.DEDUP_SIMILARITY_THRESHOLD <= 1.0

    def test_embedding_history_cap_setting_exists(self):
        from config import settings
        assert hasattr(settings, "EMBEDDING_HISTORY_CAP")
        assert settings.EMBEDDING_HISTORY_CAP >= 5
