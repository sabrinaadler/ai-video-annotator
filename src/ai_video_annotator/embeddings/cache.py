"""Validated local embedding caches with atomic directory publication."""

import hashlib
import json
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ai_video_annotator.types import TimeInterval, VideoWindow

CACHE_VERSION = 1


@dataclass(frozen=True)
class EmbeddingCache:
    embeddings: np.ndarray
    windows: tuple[VideoWindow, ...]
    metadata: dict[str, Any]


def source_fingerprint(path: str | Path) -> dict[str, Any]:
    """Identify the complete video contents, independent of file location."""
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return {"sha256": digest.hexdigest(), "size_bytes": path.stat().st_size}


def _validate(embeddings: np.ndarray, windows: tuple[VideoWindow, ...]) -> None:
    if embeddings.ndim != 2 or embeddings.shape[0] != len(windows):
        raise ValueError("Embedding rows must correspond exactly to windows")
    if not windows or embeddings.shape[1] == 0:
        raise ValueError("Cache must contain windows and nonempty vectors")
    if embeddings.dtype != np.float32 or not np.isfinite(embeddings).all():
        raise ValueError("Embeddings must be finite float32 vectors")
    video_id = windows[0].video_id
    for index, window in enumerate(windows):
        if window.index != index or window.video_id != video_id:
            raise ValueError("Windows must have consecutive indices and one video ID")
        if index and (window.interval.start_seconds, window.interval.end_seconds) < (
            windows[index - 1].interval.start_seconds,
            windows[index - 1].interval.end_seconds,
        ):
            raise ValueError("Windows must be ordered")


def save_cache(
    directory: str | Path,
    embeddings: np.ndarray,
    windows: tuple[VideoWindow, ...],
    metadata: dict[str, Any],
) -> None:
    """Publish a complete cache; never overwrite an existing directory."""
    _validate(embeddings, windows)
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError(f"Cache already exists: {directory}")
    directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{directory.name}-", dir=directory.parent))
    try:
        np.save(temporary / "embeddings.npy", embeddings, allow_pickle=False)
        (temporary / "windows.json").write_text(
            json.dumps([asdict(window) for window in windows], indent=2), encoding="utf-8"
        )
        complete_metadata = {
            **metadata,
            "cache_version": CACHE_VERSION,
            "shape": list(embeddings.shape),
            "dtype": "float32",
        }
        (temporary / "metadata.json").write_text(
            json.dumps(complete_metadata, indent=2), encoding="utf-8"
        )
        temporary.rename(directory)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def load_cache(
    directory: str | Path,
    expected_metadata: dict[str, Any] | None = None,
) -> EmbeddingCache:
    directory = Path(directory)
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("cache_version") != CACHE_VERSION:
        raise ValueError("Unsupported cache version")
    if expected_metadata is not None:
        for key, value in expected_metadata.items():
            if metadata.get(key) != value:
                raise ValueError(f"Cache metadata mismatch for {key}; use a new cache directory")
    records = json.loads((directory / "windows.json").read_text(encoding="utf-8"))
    windows = tuple(
        VideoWindow(record["video_id"], record["index"], TimeInterval(**record["interval"]))
        for record in records
    )
    embeddings = np.load(directory / "embeddings.npy", allow_pickle=False)
    _validate(embeddings, windows)
    if metadata.get("shape") != list(embeddings.shape) or metadata.get("dtype") != "float32":
        raise ValueError("Cache shape/dtype does not match metadata")
    return EmbeddingCache(embeddings, windows, metadata)
