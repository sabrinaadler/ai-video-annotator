"""Cosine retrieval over generic vectors, independent of their encoder."""

from math import isfinite

import numpy as np


def _matrix(value: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] == 0 or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite two-dimensional embedding matrix")
    return result


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0) or not np.isfinite(norms).all():
        raise ValueError("Cosine similarity requires nonzero vectors with finite norms")
    return matrix / norms


def similarity_scores(
    candidates: np.ndarray,
    examples: np.ndarray,
    method: str = "nearest-example",
    *,
    negative_examples: np.ndarray | None = None,
    negative_weight: float = 0.5,
    negative_method: str = "penalty",
) -> np.ndarray:
    """Nearest-example max cosine or cosine to the mean raw example vector."""
    if not isfinite(negative_weight) or not 0 <= negative_weight <= 1:
        raise ValueError("negative_weight must be between 0 and 1")
    if negative_method not in {"penalty", "prototype-difference"}:
        raise ValueError("Unknown counterexample scoring method")
    candidates = _matrix(candidates, "candidates")
    examples = _matrix(examples, "examples")
    if len(examples) == 0 or candidates.shape[1] != examples.shape[1]:
        raise ValueError("Provide at least one example with the same embedding dimension")
    if method == "nearest-example":
        scores = (_normalize(candidates) @ _normalize(examples).T).max(axis=1)
    elif method == "prototype":
        # Validate individual examples as well as a potentially cancelling prototype.
        _normalize(examples)
        prototype = _normalize(examples.mean(axis=0, keepdims=True))
        scores = (_normalize(candidates) @ prototype.T)[:, 0]
    else:
        raise ValueError("method must be nearest-example or prototype")
    scores = np.clip(scores, -1.0, 1.0)
    if negative_examples is not None:
        negatives = _matrix(negative_examples, "negative_examples")
        if negatives.shape[1] != candidates.shape[1] or not len(negatives):
            raise ValueError("Provide nonempty negatives with the same embedding dimension")
        if negative_method == "prototype-difference" and negative_weight:
            positive = _normalize(_normalize(examples).mean(axis=0, keepdims=True))
            negative = _normalize(_normalize(negatives).mean(axis=0, keepdims=True))
            direction = _normalize(positive - negative_weight * negative)
            return np.clip((_normalize(candidates) @ direction.T)[:, 0], -1, 1)
        penalty = np.clip((_normalize(candidates) @ _normalize(negatives).T).max(axis=1), -1, 1)
        scores = scores - negative_weight * penalty
    return scores
