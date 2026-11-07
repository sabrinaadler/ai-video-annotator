"""CalMS21 Task 1 JSON ingestion; no model or pose processing required."""

from collections.abc import Callable, Mapping, Sequence
from math import isfinite
from pathlib import Path

import ijson
import numpy as np

from ai_video_annotator.datasets.base import VideoMetadata
from ai_video_annotator.types import BehaviorEvent, TimeInterval

TASK1_VOCAB = {"attack": 0, "investigation": 1, "mount": 2, "other": 3}


def labels_to_events(
    video_id: str,
    labels: Sequence[int],
    fps: float,
    vocab: Mapping[str, int],
) -> tuple[BehaviorEvent, ...]:
    """Convert label runs to half-open intervals, including the final frame."""
    if not isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    if (
        not vocab
        or any(
            not isinstance(name, str)
            or not name.strip()
            or isinstance(value, bool)
            or not isinstance(value, int)
            for name, value in vocab.items()
        )
        or len(set(vocab.values())) != len(vocab)
    ):
        raise ValueError("vocab must map nonempty behavior names to unique integer IDs")
    values = np.asarray(labels)
    if values.ndim != 1:
        raise ValueError("annotations must be a one-dimensional label sequence")
    if not len(values):
        return ()
    if values.dtype.kind not in "iu":
        raise ValueError("annotations must contain integer labels")
    names = {value: name for name, value in vocab.items()}
    unknown = set(values.tolist()) - names.keys()
    if unknown:
        raise ValueError(f"Unknown annotation IDs: {sorted(unknown)}")
    boundaries = np.concatenate(([0], np.flatnonzero(values[1:] != values[:-1]) + 1, [len(values)]))
    return tuple(
        BehaviorEvent(video_id, names[int(values[start])], TimeInterval(start / fps, end / fps))
        for start, end in zip(boundaries[:-1], boundaries[1:], strict=True)
    )


def probe_video(path: Path, video_id: str) -> VideoMetadata:
    """Decode once to count frames exactly rather than trusting container estimates."""
    try:
        import av
    except ImportError as exc:
        raise RuntimeError('Video inspection requires PyAV: pip install -e ".[ml]"') from exc
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError(f"No video stream in {path}")
        stream = container.streams.video[0]
        if stream.average_rate is None:
            raise ValueError(f"Missing frame rate in {path}")
        fps = float(stream.average_rate)
        count = sum(1 for _ in container.decode(stream))
        return VideoMetadata(video_id, path, fps, count, stream.width, stream.height)


class CalMS21Dataset:
    """Read official nested Task 1 JSON and pair sequence IDs with local MP4s.

    Metadata vocabulary takes precedence over the documented Task 1 fallback.
    FPS comes from each video; annotations must align exactly with decoded frames.
    """

    def __init__(
        self,
        annotations_path: str | Path,
        videos_root: str | Path,
        *,
        metadata_reader: Callable[[Path, str], VideoMetadata] = probe_video,
    ) -> None:
        self.annotations_path = Path(annotations_path)
        self.videos_root = Path(videos_root)
        self._metadata_reader = metadata_reader
        self._metadata: dict[str, VideoMetadata] = {}
        self._records: dict[str, dict] = {}
        # Stream labels and vocabulary; never materialize unused keypoint arrays.
        with self.annotations_path.open("rb") as handle:
            record = None
            video_id = None
            for prefix, event, value in ijson.parse(handle):
                depth = prefix.count(".")
                if prefix == "" and event != "map_key" and event not in {"start_map", "end_map"}:
                    raise ValueError("CalMS21 JSON must contain annotator mappings")
                if depth == 0 and prefix and event not in {"map_key", "start_map", "end_map"}:
                    raise ValueError("CalMS21 annotator groups must be mappings")
                if depth == 1 and event == "start_map":
                    video_id = prefix.split(".", 1)[1]
                    if not video_id.startswith("task1/"):
                        raise ValueError("This adapter supports CalMS21 Task 1 only")
                    if video_id in self._records:
                        raise ValueError(f"Duplicate video ID: {video_id}")
                    record = {}
                elif depth == 1 and event == "end_map":
                    if record is None or "annotations" not in record:
                        raise ValueError(f"Missing annotations for {video_id!r}")
                    self._records[video_id] = record
                    record = None
                elif record is not None:
                    if prefix.endswith(".annotations") and event == "start_array":
                        record["annotations"] = []
                    elif prefix.endswith(".annotations.item"):
                        if event != "number" or not isinstance(value, int):
                            raise ValueError("Annotations must contain integer labels")
                        record["annotations"].append(value)
                    elif prefix.endswith(".metadata.vocab") and event == "start_map":
                        record["metadata"] = {"vocab": {}}
                    elif ".metadata.vocab." in prefix and event not in {"map_key", "end_map"}:
                        name = prefix.rsplit(".", 1)[1]
                        record["metadata"]["vocab"][name] = value
        if not self._records:
            raise ValueError("No annotated sequences found")
        if not self.videos_root.is_dir():
            raise FileNotFoundError(f"Video directory does not exist: {self.videos_root}")
        self._paths: dict[str, list[Path]] = {}
        for path in sorted(self.videos_root.rglob("*")):
            if path.is_file() and path.suffix.lower() == ".mp4":
                self._paths.setdefault(path.stem, []).append(path)

    def video_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._records))

    def annotation_event_counts(self) -> dict[str, dict[str, int]]:
        """Count annotated bouts without requiring extracted video or assuming its FPS."""
        counts = {}
        for video_id, record in self._records.items():
            vocab = record.get("metadata", {}).get("vocab", TASK1_VOCAB)
            # Run counts do not depend on FPS; these temporary intervals are not exposed.
            events = labels_to_events(video_id, record["annotations"], 1.0, vocab)
            counts[video_id] = {name: sum(e.behavior == name for e in events) for name in vocab}
        return counts

    def video_metadata(self, video_id: str) -> VideoMetadata:
        if video_id not in self._records:
            raise KeyError(video_id)
        if video_id not in self._metadata:
            matches = self._paths.get(Path(video_id).name, [])
            if not matches:
                raise FileNotFoundError(f"No MP4 for {video_id} under {self.videos_root}")
            if len(matches) != 1:
                raise ValueError(f"Ambiguous MP4 matches for {video_id}: {matches}")
            metadata = self._metadata_reader(matches[0], video_id)
            if metadata.video_id != video_id:
                raise ValueError("Video metadata ID does not match requested sequence")
            labels = self._records[video_id]["annotations"]
            if metadata.frame_count != len(labels):
                raise ValueError(
                    f"Frame mismatch for {video_id}: video={metadata.frame_count}, "
                    f"annotations={len(labels)}; verify the source recording"
                )
            self._metadata[video_id] = metadata
        return self._metadata[video_id]

    def behavior_events(
        self, video_id: str, behavior: str | None = None
    ) -> tuple[BehaviorEvent, ...]:
        metadata = self.video_metadata(video_id)
        record = self._records[video_id]
        vocab = record.get("metadata", {}).get("vocab", TASK1_VOCAB)
        if behavior is not None and behavior not in vocab:
            raise ValueError(f"Unknown behavior: {behavior}")
        events = labels_to_events(video_id, record["annotations"], metadata.fps, vocab)
        return tuple(event for event in events if behavior is None or event.behavior == behavior)
