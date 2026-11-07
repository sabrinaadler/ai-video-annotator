"""Aspect-ratio-preserving full-frame preprocessing for video encoders."""

import numpy as np
from PIL import Image


def letterbox_frame(
    frame: np.ndarray,
    size: int,
    fill: tuple[int, int, int],
) -> np.ndarray:
    """Fit the entire RGB frame inside a square, with symmetric neutral padding.

    Bilinear resizing preserves aspect ratio to pixel-rounding precision. The
    padding is filled with the checkpoint's RGB mean (in uint8 units).
    """
    if (
        frame.ndim != 3
        or frame.shape[-1] != 3
        or frame.dtype != np.uint8
        or min(frame.shape[:2]) <= 0
    ):
        raise ValueError("Expected a nonempty uint8 RGB frame")
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise ValueError("size must be a positive integer")
    height, width = frame.shape[:2]
    scale = min(size / width, size / height)
    resized_width = max(1, min(size, round(width * scale)))
    resized_height = max(1, min(size, round(height * scale)))
    resized = Image.fromarray(frame).resize(
        (resized_width, resized_height), Image.Resampling.BILINEAR
    )
    output = Image.new("RGB", (size, size), fill)
    output.paste(resized, ((size - resized_width) // 2, (size - resized_height) // 2))
    return np.asarray(output)
