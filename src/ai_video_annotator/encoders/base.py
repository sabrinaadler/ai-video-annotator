"""Generic clip encoder contract."""

from collections.abc import Sequence
from typing import Protocol

import numpy as np


class VideoEncoder(Protocol):
    @property
    def embedding_dim(self) -> int: ...

    def encode(self, clips: Sequence[np.ndarray]) -> np.ndarray:
        """Return (clips, embedding_dim) float32 vectors for uint8 RGB clips."""
        ...
