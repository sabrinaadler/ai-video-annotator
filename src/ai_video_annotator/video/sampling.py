"""Uniform frame sampling for constant-frame-rate video."""

from math import ceil, isfinite

import numpy as np

from ai_video_annotator.types import TimeInterval


def sample_frame_indices(
    interval: TimeInterval,
    fps: float,
    frame_count: int,
    num_frames: int = 16,
) -> np.ndarray:
    """Sample first through last eligible frame in a half-open interval.

    Repeat indices when fewer source frames than requested are available.
    Frame timestamps are index / fps; the end timestamp is excluded.
    """
    if not isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    for name, value in (("frame_count", frame_count), ("num_frames", num_frames)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if interval.end_seconds > frame_count / fps + 1e-9:
        raise ValueError("interval exceeds video duration")
    first = ceil(interval.start_seconds * fps - 1e-9)
    last = min(frame_count - 1, ceil(interval.end_seconds * fps - 1e-9) - 1)
    if first > last:
        raise ValueError("interval contains no frame timestamps")
    return np.rint(np.linspace(first, last, num_frames)).astype(np.int64)
