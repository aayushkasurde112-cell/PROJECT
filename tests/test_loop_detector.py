"""
Tests for utils/loop_detector.py.

These tests are split into two groups:

1. Pure-math / empty-input tests (test_cosine_similarity_* and
   test_calculate_loop_score_*_empty*) — these never load the
   sentence-transformers model, so they run fully offline and fast.

2. Mocked-embedding tests (test_calculate_loop_score_*_mocked) — these
   patch out the embedding model with deterministic fake vectors, so
   the "nearly identical -> high similarity" and "different -> low
   similarity" behavior specified in the project requirements is
   verified without depending on network access or a downloaded model.

3. One real-model integration test at the bottom, which exercises the
   actual sentence-transformers model end to end. It is automatically
   skipped (not failed) if the model can't be loaded — e.g. no network
   access to download it — since that's an environment constraint, not
   a code bug. Run it explicitly with network access to confirm real
   embedding behavior.
"""

import numpy as np
import pytest

from utils.loop_detector import (
    LOOP_SIMILARITY_THRESHOLD,
    _cosine_similarity,
    calculate_loop_score,
)


# ---------------------------------------------------------------------------
# 1. Pure math — no model involved
# ---------------------------------------------------------------------------


def test_cosine_similarity_identical_vectors():
    vec = np.array([1.0, 2.0, 3.0])
    assert _cosine_similarity(vec, vec) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors():
    vec_a = np.array([1.0, 0.0])
    vec_b = np.array([0.0, 1.0])
    assert _cosine_similarity(vec_a, vec_b) == pytest.approx(0.0)


def test_cosine_similarity_opposite_vectors():
    vec_a = np.array([1.0, 0.0])
    vec_b = np.array([-1.0, 0.0])
    assert _cosine_similarity(vec_a, vec_b) == pytest.approx(-1.0)


def test_cosine_similarity_zero_vector_handled_safely():
    vec_a = np.array([0.0, 0.0, 0.0])
    vec_b = np.array([1.0, 2.0, 3.0])
    # Should not raise a division-by-zero error.
    assert _cosine_similarity(vec_a, vec_b) == 0.0


# ---------------------------------------------------------------------------
# Empty / invalid input handling — required by project spec, no model needed
# ---------------------------------------------------------------------------


def test_calculate_loop_score_empty_previous_output():
    assert calculate_loop_score("", "some current output") == 0.0


def test_calculate_loop_score_empty_current_output():
    assert calculate_loop_score("some previous output", "") == 0.0


def test_calculate_loop_score_both_empty():
    assert calculate_loop_score("", "") == 0.0


def test_calculate_loop_score_whitespace_only():
    assert calculate_loop_score("   \n\t  ", "real text here") == 0.0


def test_calculate_loop_score_non_string_input():
    assert calculate_loop_score(None, "real text here") == 0.0  # type: ignore[arg-type]
    assert calculate_loop_score(123, "real text here") == 0.0  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 2. Mocked embedding model — validates the full calculate_loop_score
#    pipeline (including LOOP_SIMILARITY_THRESHOLD) without network access
# ---------------------------------------------------------------------------


class _FakeModel:
    """Stand-in for a SentenceTransformer with a scripted .encode()."""

    def __init__(self, vectors):
        self._vectors = vectors

    def encode(self, texts):
        return self._vectors


def test_calculate_loop_score_nearly_identical_texts_scores_high(monkeypatch):
    # Two near-identical vectors -> cosine similarity close to 1.0,
    # above the loop threshold.
    fake_vectors = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.99, 0.01, 0.0]),
    ]
    monkeypatch.setattr(
        "utils.loop_detector._get_embedding_model",
        lambda: _FakeModel(fake_vectors),
    )

    score = calculate_loop_score(
        "The system processed the request successfully.",
        "The system processed the request successfully!",
    )

    assert score > LOOP_SIMILARITY_THRESHOLD


def test_calculate_loop_score_different_texts_scores_low(monkeypatch):
    # Two orthogonal vectors -> cosine similarity of 0.0, well below
    # the loop threshold.
    fake_vectors = [
        np.array([1.0, 0.0]),
        np.array([0.0, 1.0]),
    ]
    monkeypatch.setattr(
        "utils.loop_detector._get_embedding_model",
        lambda: _FakeModel(fake_vectors),
    )

    score = calculate_loop_score(
        "The weather in Tokyo is sunny today.",
        "Quarterly revenue grew by twelve percent.",
    )

    assert score < 0.5
    assert score < LOOP_SIMILARITY_THRESHOLD


# ---------------------------------------------------------------------------
# 3. Real-model integration test (skipped automatically if unavailable)
# ---------------------------------------------------------------------------


def test_calculate_loop_score_real_model_end_to_end():
    """
    Exercises the actual sentence-transformers model. Skipped (not
    failed) if the model can't be loaded, e.g. because this environment
    has no network access to download it.
    """
    try:
        high_score = calculate_loop_score(
            "The weather today is sunny and warm.",
            "Today the weather is warm and sunny.",
        )
        low_score = calculate_loop_score(
            "The weather today is sunny and warm.",
            "The quarterly financial report shows a twelve percent increase in revenue.",
        )
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"Could not load embedding model (likely no network access): {exc}")

    assert high_score > LOOP_SIMILARITY_THRESHOLD
    assert low_score < high_score
