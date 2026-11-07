"""Dataset interfaces independent of encoders and retrieval."""

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Protocol

from ai_video_annotator.types import BehaviorEvent


@dataclass(frozen=True, slots=True)
class VideoMetadata:
    video_id: str
    path: Path
    fps: float
    frame_count: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if not self.video_id.strip():
            raise ValueError("video_id must not be empty")
        if not isfinite(self.fps) or self.fps <= 0:
            raise ValueError("fps must be finite and positive")
        if self.frame_count <= 0 or self.width <= 0 or self.height <= 0:
            raise ValueError("frame count and dimensions must be positive")

    @property
    def duration_seconds(self) -> float:
        return self.frame_count / self.fps


class BehaviorDataset(Protocol):
    def video_ids(self) -> tuple[str, ...]: ...

    def video_metadata(self, video_id: str) -> VideoMetadata: ...

    def behavior_events(
        self, video_id: str, behavior: str | None = None
    ) -> tuple[BehaviorEvent, ...]: ...
