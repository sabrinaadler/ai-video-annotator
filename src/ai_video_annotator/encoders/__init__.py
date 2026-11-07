"""Frozen video representation encoders."""

from .base import VideoEncoder
from .videomae import VideoMAEEncoder

__all__ = ["VideoEncoder", "VideoMAEEncoder"]
