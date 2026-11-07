import json

import numpy as np
import pytest

from ai_video_annotator.embeddings import load_cache, save_cache, source_fingerprint
from ai_video_annotator.embeddings.encoding import encode_sampled_windows
from ai_video_annotator.encoders import VideoMAEEncoder
from ai_video_annotator.video import generate_windows


def test_cache_roundtrip_and_identity(tmp_path):
    windows = generate_windows("mouse", 3)
    vectors = np.arange(len(windows) * 4, dtype=np.float32).reshape(-1, 4)
    path = tmp_path / "cache"
    save_cache(path, vectors, windows, {"model": "test"})
    result = load_cache(path, {"model": "test"})
    np.testing.assert_array_equal(result.embeddings, vectors)
    assert result.windows == windows
    with pytest.raises(ValueError, match="mismatch"):
        load_cache(path, {"model": "different"})
    with pytest.raises(FileExistsError):
        save_cache(path, vectors, windows, {})


@pytest.mark.parametrize(
    "vectors",
    [
        np.ones((1, 4), dtype=np.float32),
        np.ones((3, 4), dtype=np.float64),
        np.full((3, 4), np.nan, dtype=np.float32),
    ],
)
def test_cache_rejects_invalid_vectors(tmp_path, vectors):
    with pytest.raises(ValueError):
        save_cache(tmp_path / "cache", vectors, generate_windows("mouse", 3), {})
    assert not (tmp_path / "cache").exists()


def test_corrupt_metadata(tmp_path):
    path = tmp_path / "cache"
    save_cache(path, np.ones((3, 4), dtype=np.float32), generate_windows("mouse", 3), {})
    metadata_path = path / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["shape"] = [3, 5]
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="shape"):
        load_cache(path)


def test_fingerprint_changes_with_contents(tmp_path):
    path = tmp_path / "video"
    path.write_bytes(b"one")
    original = source_fingerprint(path)
    path.write_bytes(b"two")
    assert original != source_fingerprint(path)


def test_partial_batch_and_order():
    class Encoder:
        embedding_dim = 1

        def __init__(self):
            self.batches = []

        def encode(self, clips):
            self.batches.append(len(clips))
            return np.asarray([[clip.item()] for clip in clips], dtype=np.float32)

    encoder = Encoder()
    windows = generate_windows("mouse", 4)
    vectors, result = encode_sampled_windows(
        encoder, ((window, np.array(index)) for index, window in enumerate(windows)), 2
    )
    assert encoder.batches == [2, 2, 1]
    assert result == windows
    np.testing.assert_array_equal(vectors[:, 0], np.arange(5))


def test_real_videomae_local_inference(tmp_path):
    import torch
    from transformers import VideoMAEConfig, VideoMAEImageProcessor, VideoMAEModel

    torch.manual_seed(0)
    config = VideoMAEConfig(
        image_size=16,
        patch_size=8,
        num_frames=4,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
    )
    model = VideoMAEModel(config)
    model.save_pretrained(tmp_path)
    processor = VideoMAEImageProcessor(
        size={"shortest_edge": 16}, crop_size={"height": 16, "width": 16}
    )
    processor.save_pretrained(tmp_path)
    encoder = VideoMAEEncoder(
        str(tmp_path), device="cpu", local_files_only=True, preprocessing="center-crop"
    )
    assert not encoder.model.training
    assert all(not parameter.requires_grad for parameter in encoder.model.parameters())
    clip = np.random.default_rng(0).integers(0, 256, (4, 20, 24, 3), dtype=np.uint8)
    vectors = encoder.encode([clip, clip])
    assert vectors.shape == (2, 16)
    assert vectors.dtype == np.float32
    np.testing.assert_allclose(vectors[0], vectors[1], atol=1e-6)
    pixels = encoder.processor([list(clip)], return_tensors="pt")["pixel_values"]
    with torch.inference_mode():
        expected = encoder.model(pixel_values=pixels).last_hidden_state.mean(dim=1).numpy()
    np.testing.assert_allclose(vectors[:1], expected, atol=1e-6)
    assert encoder.encode([]).shape == (0, 16)
    with pytest.raises(ValueError, match="Expected"):
        encoder.encode([clip.astype(np.float32)])


def test_cli_reuses_cache_without_loading_model(tmp_path):
    import importlib.metadata
    import os
    import subprocess
    import sys

    video = tmp_path / "source.mp4"
    # Intentionally not decodable: reuse should verify content, not open video/model.
    video.write_bytes(b"cached video content")
    identity = {
        "source": source_fingerprint(video),
        "video_id": "mouse",
        "model": "MCG-NJU/videomae-base",
        "requested_revision": "main",
        "window_seconds": 2.0,
        "stride_seconds": 0.5,
        "num_frames": 16,
        "pooling": "mean_all_tokens",
        "preprocessing": "letterbox-v1",
        "sampling": "uniform_inclusive_frame_indices_v1",
        "transformers_version": importlib.metadata.version("transformers"),
    }
    cache = tmp_path / "cache"
    save_cache(cache, np.ones((3, 4), dtype=np.float32), generate_windows("mouse", 3), identity)
    command = [
        sys.executable,
        "scripts/encode_video.py",
        "--video",
        str(video),
        "--video-id",
        "mouse",
        "--cache",
        str(cache),
        "--local-files-only",
    ]
    env = {**os.environ, "PYTHONPATH": "src", "HF_HUB_OFFLINE": "1"}
    result = subprocess.run(command, env=env, capture_output=True, text=True, check=True)
    assert "Model was not loaded" in result.stdout
    incompatible = subprocess.run(
        [*command, "--preprocessing", "center-crop"], env=env, capture_output=True, text=True
    )
    assert incompatible.returncode == 2
    assert "preprocessing" in incompatible.stderr
    video.write_bytes(b"changed contents")
    result = subprocess.run(command, env=env, capture_output=True, text=True)
    assert result.returncode == 2
    assert "metadata mismatch" in result.stderr
