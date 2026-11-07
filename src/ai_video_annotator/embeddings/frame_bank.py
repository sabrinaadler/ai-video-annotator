"""Resumable, bounded-batch frame features shared by genuine temporal proposals."""

import json
import os
from pathlib import Path

import numpy as np

from ai_video_annotator.encoders.image import pool_frame_features
from ai_video_annotator.video.sampling import sample_frame_indices


def encode_frame_bank(encoder, metadata, directory, identity, chunk_size=1024, progress=None):
    import av

    if chunk_size < encoder.frame_batch_size or chunk_size < 1:
        raise ValueError("Chunk size must be at least the inference batch size")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    header = directory / "identity.json"
    identity = {
        **identity,
        "frame_count": metadata.frame_count,
        "fps": metadata.fps,
        "embedding_dim": encoder.embedding_dim,
        "chunk_size": chunk_size,
    }
    if header.exists():
        previous = json.loads(header.read_text())
        if any(previous.get(key) != value for key, value in identity.items()):
            raise ValueError("Frame bank identity mismatch")
    else:
        header.write_text(json.dumps(identity, indent=2))
    paths = sorted(directory.glob("chunk-*.npy"))
    complete = 0
    for index, path in enumerate(paths):
        values = np.load(path, allow_pickle=False)
        expected = min(chunk_size, metadata.frame_count - complete)
        if (
            path.name != f"chunk-{index:06d}.npy"
            or values.shape != (expected, encoder.embedding_dim)
            or values.dtype != np.float32
            or not np.isfinite(values).all()
        ):
            raise ValueError("Invalid/incomplete frame bank chunk")
        complete += len(values)
    if complete > metadata.frame_count:
        raise ValueError("Frame bank exceeds recording")
    if complete == metadata.frame_count:
        return np.concatenate([np.load(p, allow_pickle=False) for p in paths])
    frames, features = [], []
    chunk_rows = 0

    def flush_chunk():
        nonlocal chunk_rows, complete
        values = np.concatenate(features).astype(np.float32)
        if not np.isfinite(values).all():
            raise ValueError("Frame bank contains nonfinite values")
        path = directory / f"chunk-{len(paths):06d}.npy"
        temporary = path.with_suffix(".tmp")
        with temporary.open("wb") as handle:
            np.save(handle, values, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        paths.append(path)
        complete += len(values)
        features.clear()
        chunk_rows = 0
        if progress:
            progress(complete, metadata.frame_count)

    with av.open(str(metadata.path)) as container:
        for index, frame in enumerate(container.decode(video=0)):
            if index < complete:
                continue
            if index >= metadata.frame_count:
                raise ValueError("Decoded recording exceeds metadata")
            frames.append(frame.to_ndarray(format="rgb24"))
            if len(frames) == encoder.frame_batch_size or index + 1 == metadata.frame_count:
                values = encoder.encode_frames(frames)
                if values.shape != (len(frames), encoder.embedding_dim):
                    raise ValueError("Unexpected frame feature shape")
                features.append(values)
                chunk_rows += len(frames)
                frames.clear()
                # Chunk boundaries must be independent of inference batch size.
                if chunk_rows >= chunk_size:
                    merged = np.concatenate(features)
                    features[:] = [merged[:chunk_size]]
                    remainder = merged[chunk_size:]
                    flush_chunk()
                    if len(remainder):
                        features.append(remainder)
                        chunk_rows = len(remainder)
    if features:
        flush_chunk()
    if complete != metadata.frame_count:
        raise ValueError("Recording ended before all frame features were encoded")
    return np.concatenate([np.load(p, allow_pickle=False) for p in paths])


def pool_frame_bank(features, metadata, windows, num_frames=16):
    """Re-sample each actual proposal; never attach shorter boundaries to old vectors."""
    if len(features) != metadata.frame_count:
        raise ValueError("Frame bank does not match recording")
    return np.stack(
        [
            pool_frame_features(
                features[
                    sample_frame_indices(
                        window.interval, metadata.fps, metadata.frame_count, num_frames
                    )
                ]
            )
            for window in windows
        ]
    ).astype(np.float32)
