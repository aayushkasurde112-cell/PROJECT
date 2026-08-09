

import os
from functools import lru_cache

import numpy as np

# Threshold above which two outputs are considered a "loop". Exposed as
# a module constant so the future conditional edge can import it
# directly instead of re-hardcoding 0.95 elsewhere.
LOOP_SIMILARITY_THRESHOLD = 0.95


@lru_cache(maxsize=1)
def _get_embedding_model():
    # pyrefly: ignore [missing-import]
    from sentence_transformers import SentenceTransformer

    model_name = os.getenv("LOOP_DETECTOR_MODEL", "all-MiniLM-L6-v2")
    return SentenceTransformer(model_name)


def _cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


def calculate_loop_score(previous_output: str, current_output: str) -> float:
    if not isinstance(previous_output, str) or not isinstance(current_output, str):
        return 0.0
    if not previous_output.strip() or not current_output.strip():
        return 0.0

    model = _get_embedding_model()
    embeddings = model.encode([previous_output, current_output])
    return _cosine_similarity(embeddings[0], embeddings[1])
