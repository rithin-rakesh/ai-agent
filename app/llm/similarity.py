"""Cosine similarity and vector normalization utilities.

Provides safe mathematical calculations for dense embedding vectors
with zero-vector protection, dimension validation, and 0-100 normalization.
"""

import math
from typing import List, Optional


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Compute raw cosine similarity between two float vectors.

    Returns:
        float: Cosine similarity in range [-1.0, 1.0]. Returns 0.0 for zero vectors
               or mismatched/empty dimensions.
    """
    if not vec1 or not vec2:
        return 0.0

    if len(vec1) != len(vec2):
        return 0.0

    dot_product = 0.0
    norm_a_sq = 0.0
    norm_b_sq = 0.0

    for a, b in zip(vec1, vec2):
        dot_product += a * b
        norm_a_sq += a * a
        norm_b_sq += b * b

    if norm_a_sq <= 0.0 or norm_b_sq <= 0.0:
        return 0.0

    similarity = dot_product / (math.sqrt(norm_a_sq) * math.sqrt(norm_b_sq))
    # Clamp to [-1.0, 1.0] to handle floating point precision artifacts
    return max(-1.0, min(1.0, similarity))


def calculate_semantic_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate normalized semantic similarity score on a 0.0 - 100.0 scale.

    Maps cosine similarity [-1.0, 1.0] into non-negative [0.0, 100.0] score.
    For positive alignment (sim >= 0), score = sim * 100.
    For negative alignment (sim < 0), score = 0.0.

    Returns:
        float: Normalized score between 0.0 and 100.0 rounded to 2 decimal places.
    """
    sim = cosine_similarity(vec1, vec2)
    if sim <= 0.0:
        return 0.0
    return round(min(100.0, sim * 100.0), 2)
