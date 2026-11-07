"""Reuse image features for repeated source frames across overlapping video windows."""

import numpy as np

from ai_video_annotator.encoders.image import pool_frame_features
from ai_video_annotator.video.sampling import sample_frame_indices


def encode_image_windows(
    encoder, metadata, sampled, *, num_frames=16, window_batch_size=4, progress=None
):
    if window_batch_size < 1 or num_frames < 4:
        raise ValueError("Positive batches and at least four sampled frames required")
    buffer, new_frames, pending, windows = {}, {}, [], []
    vectors = {"mean": [], "ordered4": []}
    frame_count = 0

    def flush():
        nonlocal frame_count, buffer
        indices = sorted(new_frames)
        if indices:
            encoded = encoder.encode_frames([new_frames[i] for i in indices])
            if encoded.shape != (len(indices), encoder.embedding_dim):
                raise ValueError("Unexpected image feature shape")
            buffer.update(zip(indices, encoded, strict=True))
            frame_count += len(indices)
        for window, sampled_indices in pending:
            features = np.stack([buffer[int(i)] for i in sampled_indices])
            for name, bins in (("mean", 1), ("ordered4", 4)):
                vectors[name].append(pool_frame_features(features, bins))
            windows.append(window)
        first = int(pending[-1][1][0])
        buffer = {i: feature for i, feature in buffer.items() if i >= first}
        new_frames.clear()
        pending.clear()
        if progress:
            progress(len(windows), frame_count)

    previous = -1.0
    for window, clip in sampled:
        if window.video_id != metadata.video_id or window.interval.start_seconds < previous:
            raise ValueError("Windows must belong to the recording and be ordered")
        indices = sample_frame_indices(
            window.interval, metadata.fps, metadata.frame_count, num_frames
        )
        if len(clip) != len(indices):
            raise ValueError("Sampled clip length mismatch")
        for index, frame in zip(indices, clip, strict=True):
            if int(index) not in buffer and int(index) not in new_frames:
                new_frames[int(index)] = frame
        pending.append((window, indices))
        previous = window.interval.start_seconds
        if len(pending) >= window_batch_size:
            flush()
    if pending:
        flush()
    if not windows:
        raise ValueError("No windows to encode")
    return (
        {key: np.stack(value).astype(np.float32) for key, value in vectors.items()},
        tuple(windows),
        frame_count,
    )
