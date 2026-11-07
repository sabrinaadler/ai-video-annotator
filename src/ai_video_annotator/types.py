"""Shared domain types for temporal video retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True, slots=True)
class TimeInterval:
    """Half-open interval [start_seconds, end_seconds) on a video timeline."""

    start_seconds: float
    end_seconds: float

    def __post_init__(self) -> None:
        if not isfinite(self.start_seconds) or not isfinite(self.end_seconds):
            raise ValueError("interval bounds must be finite")
        if self.start_seconds < 0:
            raise ValueError("start_seconds must be non-negative")
        if self.end_seconds <= self.start_seconds:
            raise ValueError("end_seconds must be greater than start_seconds")

    @property
    def duration_seconds(self) -> float:
        return self.end_seconds - self.start_seconds

    def intersection_seconds(self, other: TimeInterval) -> float:
        start = max(self.start_seconds, other.start_seconds)
        end = min(self.end_seconds, other.end_seconds)
        return max(0.0, end - start)

    def union_seconds(self, other: TimeInterval) -> float:
        return self.duration_seconds + other.duration_seconds - self.intersection_seconds(other)

    def temporal_iou(self, other: TimeInterval) -> float:
        union = self.union_seconds(other)
        if union == 0:
            return 0.0
        return self.intersection_seconds(other) / union


@dataclass(frozen=True, slots=True)
class BehaviorEvent:
    """Ground-truth or researcher-provided behavior interval."""

    video_id: str
    behavior: str
    interval: TimeInterval

    def __post_init__(self) -> None:
        if not self.video_id.strip():
            raise ValueError("video_id must not be empty")
        if not self.behavior.strip():
            raise ValueError("behavior must not be empty")


@dataclass(frozen=True, slots=True)
class VideoWindow:
    """A model candidate window generated from a continuous source video."""

    video_id: str
    index: int
    interval: TimeInterval

    def __post_init__(self) -> None:
        if not self.video_id.strip():
            raise ValueError("video_id must not be empty")
        if self.index < 0:
            raise ValueError("index must be non-negative")


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """A ranked retrieval candidate and its similarity score."""

    window: VideoWindow
    score: float
