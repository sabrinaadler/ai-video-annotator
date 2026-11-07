"""Dataset adapters exposing generic videos and behavior intervals."""

from .base import BehaviorDataset, VideoMetadata
from .calms21 import CalMS21Dataset

__all__ = ["BehaviorDataset", "CalMS21Dataset", "VideoMetadata"]
