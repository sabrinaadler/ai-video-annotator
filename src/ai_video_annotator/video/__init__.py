"""Temporal windowing and generic video frame sampling."""

from .reader import iter_sampled_windows
from .sampling import sample_frame_indices
from .windows import generate_windows

__all__ = ["generate_windows", "iter_sampled_windows", "sample_frame_indices"]
