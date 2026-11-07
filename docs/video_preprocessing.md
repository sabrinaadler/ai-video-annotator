# Temporal windows and frame sampling

The preprocessing library works independently of CalMS21 and the encoder.

```python
from ai_video_annotator.video import generate_windows, iter_sampled_windows

# metadata is a VideoMetadata instance, e.g. from dataset.video_metadata(video_id).
windows = generate_windows(metadata.video_id, metadata.duration_seconds,
                           window_seconds=2.0, stride_seconds=0.5)
for window, frames in iter_sampled_windows(metadata, windows, num_frames=16):
    # frames: uint8 RGB, shape (16, height, width, 3)
    pass  # The encoder will resize and normalize these in the next milestone.
```

Windows start at zero and advance by the stride. When the stride does not land
exactly on the end, one final full-length window is aligned to the video end.
Short videos produce one shorter window. Strides larger than the window are
rejected to avoid gaps. Intervals use seconds and exclude the end timestamp.

Sampling includes the first and last frame timestamps inside each window and
uniformly distributes the requested indices between them. Short intervals repeat
frames to preserve the requested shape. Intervals with no frame timestamps or
outside the video fail explicitly. This version assumes constant frame rate and
frame-zero alignment; variable frame rate timestamp sampling is future work.

The reader requires windows ordered by both start and end. It decodes forward once,
reuses overlapping decoded frames, and discards frames before the current window.
Its working memory scales with the window length rather than video duration.
Process each yielded array incrementally; collecting the iterator into a list
would retain the entire result in memory.

Tests compare sampled pixels against an independently decoded synthetic video,
including overlapping windows, short intervals, wrong IDs, and truncated media.
