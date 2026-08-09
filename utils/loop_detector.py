"""
Cosine-similarity based loop detection.

RESPONSIBILITY: Aayush + Rehan

Compares two agent outputs (e.g. successive `synthesized_answer` or
`formatted_answer` values across pipeline iterations) and returns a
`loop_score` between 0.0 and 1.0. A score above LOOP_SIMILARITY_THRESHOLD
(0.95) indicates the workflow is likely stuck producing near-identical
output.

This module is intentionally self-contained:
- It does not read or write LangGraph state directly.
- It does not know about a Debugger node or any conditional edge.
- It's designed to be called from a future conditional-edge function
  (owned by Pari/Gangotri) roughly like:

      from utils.loop_detector import calculate_loop_score, LOOP_SIMILARITY_THRESHOLD

      score = calculate_loop_score(previous_output, current_output)
      if score > LOOP_SIMILARITY_THRESHOLD:
          route to "debugger"
      else:
          route to "continue"

No Debugger node or conditional edge is added here — only the scoring
function and the threshold constant it should be compared against.
"""

import os
from functools import lru_cache

import numpy as np

# Threshold above which two outputs are considered a "loop". Exposed as
# a module constant so the future conditional edge can import it
# directly instead of re-hardcoding 0.95 elsewhere.
LOOP_SIMILARITY_THRESHOLD = 0.95


@lru_cache(maxsize=1)
def _get_embedding_model():
    """
    Lazily loads and caches the sentence-transformers model.

    Loading is deferred to first use (rather than at import time) so
    importing this module never triggers a model download, and so test
    code that only exercises the empty/invalid-input paths never needs
    the model at all.

    Configurable via the LOOP_DETECTOR_MODEL env var so the embedding
    model can be swapped without a code change. Defaults to a small,
    fast general-purpose sentence-embedding model, which is a
    reasonable fit for comparing short-to-medium agent text outputs.
    """
    from sentence_transformers import SentenceTransformer

    model_name = os.getenv("LOOP_DETECTOR_MODEL", "all-MiniLM-L6-v2")
    return SentenceTransformer(model_name)


def _cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Pure cosine-similarity computation over two vectors.

    Kept separate from embedding generation so the math itself can be
    unit tested with synthetic vectors, without needing to load a real
    embedding model (and therefore without needing network access).

    Returns 0.0 for a zero-magnitude vector instead of raising a
    division-by-zero error, since a zero vector has no defined
    direction to compare.
    """
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


def calculate_loop_score(previous_output: str, current_output: str) -> float:
    """
    Computes a cosine-similarity based loop_score between two
    successive agent outputs.

    Args:
        previous_output: The prior agent output to compare against.
        current_output: The most recent agent output.

    Returns:
        A float loop_score, generally in [0.0, 1.0] (cosine similarity
        of sentence embeddings for non-degenerate text stays in this
        range in practice; callers comparing against
        LOOP_SIMILARITY_THRESHOLD don't need to special-case this).

        Returns 0.0 — meaning "not a loop" — if either input is empty,
        whitespace-only, or not a string. Two blank/invalid outputs
        carry no comparable content, so they must never be reported as
        a loop just because they're both "nothing".

    Raises:
        Nothing under normal use. Model-loading failures (e.g. no
        network access to download the embedding model on first use)
        propagate as whatever exception sentence-transformers raises,
        since silently returning 0.0 in that case would hide a real
        infrastructure problem from the caller.
    """
    if not isinstance(previous_output, str) or not isinstance(current_output, str):
        return 0.0
    if not previous_output.strip() or not current_output.strip():
        return 0.0

    model = _get_embedding_model()
    embeddings = model.encode([previous_output, current_output])
    return _cosine_similarity(embeddings[0], embeddings[1])
