"""Fly-vs-Fly aggression annotations, adapted to scene-level temporal events."""

from pathlib import Path

import numpy as np

from ai_video_annotator.datasets.calms21 import probe_video
from ai_video_annotator.types import BehaviorEvent, TimeInterval


def merge_intervals(intervals):
    """Union overlapping actors' bouts; do not count simultaneous actors twice."""
    merged = []
    for interval in sorted(intervals, key=lambda i: (i.start_seconds, i.end_seconds)):
        if merged and interval.start_seconds < merged[-1].end_seconds:
            previous = merged.pop()
            merged.append(
                TimeInterval(
                    previous.start_seconds, max(previous.end_seconds, interval.end_seconds)
                )
            )
        else:
            merged.append(interval)
    return tuple(merged)


def load_bouts(path: Path, video_id: str, fps: float, frame_count: int):
    from scipy.io import loadmat

    data = loadmat(path, simplify_cells=True)
    behaviors = list(np.atleast_1d(data["behs"]))
    cells = np.asarray(data["bouts"], dtype=object)
    if cells.ndim != 2 or cells.shape[1] != len(behaviors):
        raise ValueError("Expected actor × behavior MATLAB bout cells")
    records = []
    grouped = {}
    for actor in range(cells.shape[0]):
        for index, behavior in enumerate(behaviors):
            values = np.asarray(cells[actor, index])
            if not values.size:
                continue
            values = values.reshape(-1, 3)
            if (
                values.dtype.kind not in "iu"
                or np.any(values[:, 0] < 1)
                or np.any(values[:, 1] < values[:, 0])
                or np.any(values[:, 1] > frame_count)
            ):
                raise ValueError("Invalid one-based inclusive bout frame range")
            for first, last, flag in values:
                interval = TimeInterval((int(first) - 1) / fps, int(last) / fps)
                grouped.setdefault(str(behavior), []).append(interval)
                records.append(
                    {
                        "actor": actor + 1,
                        "behavior": str(behavior),
                        "first_frame": int(first),
                        "last_frame": int(last),
                        "raw_flag": int(flag),
                    }
                )
    events = tuple(
        BehaviorEvent(video_id, behavior, interval)
        for behavior, intervals in grouped.items()
        for interval in merge_intervals(intervals)
    )
    return events, records


def official_split(directory: Path):
    from scipy.io import loadmat

    data = loadmat(directory / "traintest.mat", simplify_cells=True)
    names = list(np.atleast_1d(data["movienames"]))
    return {
        str(names[int(index) - 1]): split
        for split, key in (("train", "trainids"), ("test", "testids"))
        for index in np.atleast_1d(data[key])
    }


def recording(directory: Path, name: str):
    video_id = f"flyvsfly/aggression/{name}"
    video = directory / name / f"{name}.mp4"
    metadata = probe_video(video, video_id)
    events, provenance = load_bouts(
        directory / name / f"{name}_actions.mat", video_id, metadata.fps, metadata.frame_count
    )
    return metadata, events, provenance
