"""Batched encoding independent of datasets and model implementations."""

from collections.abc import Callable, Iterable

import numpy as np

from ai_video_annotator.encoders.base import VideoEncoder
from ai_video_annotator.types import VideoWindow


def encode_sampled_windows(
    encoder: VideoEncoder,
    sampled: Iterable[tuple[VideoWindow, np.ndarray]],
    batch_size: int = 8,
    progress: Callable[[int], None] | None = None,
) -> tuple[np.ndarray, tuple[VideoWindow, ...]]:
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    windows, clips, batches = [], [], []

    def flush() -> None:
        result = encoder.encode(clips)
        if result.shape != (len(clips), encoder.embedding_dim):
            raise ValueError("Encoder returned an unexpected embedding shape")
        if result.dtype != np.float32 or not np.isfinite(result).all():
            raise ValueError("Encoder must return finite float32 embeddings")
        batches.append(result)
        clips.clear()
        if progress is not None:
            progress(len(windows))

    for window, clip in sampled:
        windows.append(window)
        clips.append(clip)
        if len(clips) == batch_size:
            flush()
    if clips:
        flush()
    if not batches:
        raise ValueError("No windows to encode")
    return np.concatenate(batches), tuple(windows)
