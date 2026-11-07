from pathlib import Path

import numpy as np
import pytest

from ai_video_annotator.datasets.base import VideoMetadata
from ai_video_annotator.embeddings.image_encoding import encode_image_windows
from ai_video_annotator.encoders.image import pool_frame_features
from ai_video_annotator.video import generate_windows
from ai_video_annotator.video.sampling import sample_frame_indices


def test_mean_is_order_invariant_and_temporal_bins_preserve_order():
    features = np.array([[1, 0]] * 4 + [[0, 1]] * 4, dtype=np.float32)
    assert np.allclose(pool_frame_features(features), pool_frame_features(features[::-1]))
    assert not np.allclose(pool_frame_features(features, 4), pool_frame_features(features[::-1], 4))
    assert np.linalg.norm(pool_frame_features(features, 4)) == pytest.approx(1)
    with pytest.raises(ValueError, match="nonzero"):
        pool_frame_features(np.zeros((8, 2)))


def test_frame_reuse_matches_naive_window_features():
    metadata = VideoMetadata("video", Path("unused.mp4"), 8, 40, 4, 4)
    windows = generate_windows("video", 5, 2, 0.5)
    indices = [sample_frame_indices(w.interval, 8, 40, 16) for w in windows]
    clips = [np.stack([np.full((4, 4, 3), int(i), np.uint8) for i in ids]) for ids in indices]

    class Encoder:
        embedding_dim = 2
        seen = []

        def encode_frames(self, frames):
            values = [int(frame[0, 0, 0]) for frame in frames]
            self.seen.extend(values)
            return np.array([[1, value / 40] for value in values], dtype=np.float32)

    encoder = Encoder()
    output, encoded, count = encode_image_windows(
        encoder, metadata, zip(windows, clips, strict=False), window_batch_size=2
    )
    assert encoded == windows
    assert count == len(set(np.concatenate(indices)))
    assert len(encoder.seen) == len(set(encoder.seen))
    for mode, bins in (("mean", 1), ("ordered4", 4)):
        expected = np.stack(
            [pool_frame_features(np.array([[1, i / 40] for i in ids]), bins) for ids in indices]
        )
        assert np.allclose(output[mode], expected)


def test_cache_geometry_check_rejects_changed_sampling():
    import runpy

    from ai_video_annotator.embeddings.cache import EmbeddingCache

    verify = runpy.run_path("scripts/compare_encoders.py")["verify_geometry"]
    windows = generate_windows("video", 5, 2, 0.5)
    vectors = np.ones((len(windows), 2), np.float32)
    first = EmbeddingCache(vectors, windows, {"num_frames": 16})
    changed = EmbeddingCache(vectors, windows, {"num_frames": 8})
    with pytest.raises(ValueError, match="num_frames"):
        verify({"videomae": first, "bioclip": changed})


def test_real_image_forward_uses_inference_mode_and_validates_rgb():
    import torch

    from ai_video_annotator.encoders.image import ImageClipEncoder

    class Model(torch.nn.Module):
        def forward(self, pixels):
            assert not torch.is_grad_enabled()
            return pixels.mean(dim=(2, 3))

    encoder = ImageClipEncoder.__new__(ImageClipEncoder)
    encoder.backend = "bioclip"
    encoder.device = torch.device("cpu")
    encoder.embedding_dim = 3
    encoder.image_size = 8
    encoder.frame_batch_size = 2
    encoder.mean = [0, 0, 0]
    encoder.std = [1, 1, 1]
    encoder.model = Model()
    clip = np.full((4, 8, 8, 3), 128, dtype=np.uint8)
    result = encoder.encode([clip])
    assert result.shape == (1, 3)
    assert result.dtype == np.float32
    assert np.linalg.norm(result) == pytest.approx(1)
    with pytest.raises(ValueError, match="RGB"):
        encoder.encode_frames([np.zeros((8, 8), dtype=np.uint8)])


def test_visual_only_loading_matches_full_clip_and_rejects_missing_weights(tmp_path):
    import torch
    from open_clip.model import CLIP

    from ai_video_annotator.encoders.image import load_openclip_visual

    config = {
        "embed_dim": 8,
        "vision_cfg": {"image_size": 8, "patch_size": 4, "width": 64, "layers": 1},
        "text_cfg": {"context_length": 4, "vocab_size": 16, "width": 64, "heads": 1, "layers": 1},
    }
    original = CLIP(**config).eval()
    checkpoint = tmp_path / "model.bin"
    torch.save(original.state_dict(), checkpoint)
    visual = load_openclip_visual(config, checkpoint).eval()
    pixels = torch.randn(2, 3, 8, 8)
    with torch.inference_mode():
        assert torch.equal(original.visual(pixels), visual(pixels))
    torch.save(original.state_dict(), checkpoint, _use_new_zipfile_serialization=False)
    legacy = load_openclip_visual(config, checkpoint).eval()
    with torch.inference_mode():
        assert torch.equal(original.visual(pixels), legacy(pixels))
    state = original.state_dict()
    del state["visual.conv1.weight"]
    torch.save(state, checkpoint)
    with pytest.raises(RuntimeError, match="Missing key"):
        load_openclip_visual(config, checkpoint)
