"""Bounded-memory sequential RGB decoding for overlapping windows."""

from collections.abc import Iterable, Iterator

import numpy as np

from ai_video_annotator.datasets.base import VideoMetadata
from ai_video_annotator.types import VideoWindow
from ai_video_annotator.video.sampling import sample_frame_indices


def iter_sampled_windows(
    metadata: VideoMetadata,
    windows: Iterable[VideoWindow],
    num_frames: int = 16,
) -> Iterator[tuple[VideoWindow, np.ndarray]]:
    """Yield uint8 RGB arrays shaped (frames, height, width, channels).

    Decode forward once, retaining only frames needed for the current overlap.
    Windows must be ordered by start and end. Constant frame rate and frame-zero
    alignment are assumed, matching the initial dataset adapter.
    """
    import av

    with av.open(str(metadata.path)) as container:
        if not container.streams.video:
            raise ValueError(f"No video stream in {metadata.path}")
        decoded = iter(container.decode(container.streams.video[0]))
        buffer: dict[int, np.ndarray] = {}
        current_index = -1
        previous_start = previous_end = -1.0
        for window in windows:
            if window.video_id != metadata.video_id:
                raise ValueError("Window video ID does not match metadata")
            interval = window.interval
            if interval.start_seconds < previous_start or interval.end_seconds < previous_end:
                raise ValueError("Windows must be ordered by start and end")
            indices = sample_frame_indices(interval, metadata.fps, metadata.frame_count, num_frames)
            first, last = int(indices[0]), int(indices[-1])
            buffer = {index: frame for index, frame in buffer.items() if index >= first}
            while current_index < last:
                try:
                    frame = next(decoded)
                except StopIteration as exc:
                    raise ValueError(
                        "Video ended before the requested frame; check metadata"
                    ) from exc
                current_index += 1
                if current_index >= first:
                    buffer[current_index] = frame.to_ndarray(format="rgb24")
            if any(int(index) not in buffer for index in indices):
                raise ValueError("Windows require frames already discarded; use ordered windows")
            yield window, np.stack([buffer[int(index)] for index in indices])
            previous_start, previous_end = interval.start_seconds, interval.end_seconds
