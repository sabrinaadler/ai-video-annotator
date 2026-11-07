"""Encoder-independent few-shot retrieval and temporal deduplication."""

from .postprocess import temporal_nms
from .search import example_window_indices, retrieve
from .similarity import similarity_scores

__all__ = ["example_window_indices", "retrieve", "similarity_scores", "temporal_nms"]
