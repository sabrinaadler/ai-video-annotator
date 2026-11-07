import numpy as np
import pytest

from ai_video_annotator.encoders.preprocessing import letterbox_frame
from ai_video_annotator.encoders.videomae import VideoMAEEncoder


@pytest.mark.parametrize("shape", [(20, 40), (40, 20)])
def test_letterbox_retains_both_edges_and_neutral_padding(shape):
    h, w = shape
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    if w > h:
        frame[:, :8] = [255, 0, 0]
        frame[:, -8:] = [0, 255, 0]
    else:
        frame[:8, :] = [255, 0, 0]
        frame[-8:, :] = [0, 255, 0]
    result = letterbox_frame(frame, 20, (100, 110, 120))
    assert result.shape == (20, 20, 3)
    if w > h:
        np.testing.assert_array_equal(result[10, 0], [255, 0, 0])
        np.testing.assert_array_equal(result[10, -1], [0, 255, 0])
        np.testing.assert_array_equal(result[0, 10], [100, 110, 120])
    else:
        np.testing.assert_array_equal(result[0, 10], [255, 0, 0])
        np.testing.assert_array_equal(result[-1, 10], [0, 255, 0])
        np.testing.assert_array_equal(result[10, 0], [100, 110, 120])


def test_square_frame_preserved_without_padding():
    frame = np.random.default_rng(42).integers(0, 256, (16, 16, 3), dtype=np.uint8)
    np.testing.assert_array_equal(letterbox_frame(frame, 16, (128, 128, 128)), frame)


def test_full_frame_processor_and_encoder_integration(tmp_path):
    from transformers import VideoMAEConfig, VideoMAEImageProcessor, VideoMAEModel

    VideoMAEModel(
        VideoMAEConfig(
            image_size=16,
            patch_size=8,
            num_frames=4,
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=32,
        )
    ).save_pretrained(tmp_path)
    VideoMAEImageProcessor(
        size={"shortest_edge": 16},
        crop_size={"height": 16, "width": 16},
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
    ).save_pretrained(tmp_path)
    encoder = VideoMAEEncoder(str(tmp_path), device="cpu", local_files_only=True)
    clip = np.zeros((4, 16, 32, 3), dtype=np.uint8)
    clip[:, :, :8] = [255, 0, 0]
    clip[:, :, -8:] = [0, 255, 0]
    pixels = encoder.prepare_pixels([clip])
    assert tuple(pixels.shape) == (1, 4, 3, 16, 16)
    np.testing.assert_allclose(pixels[0, 0, :, 8, 0].numpy(), [1, -1, -1], atol=0.01)
    np.testing.assert_allclose(pixels[0, 0, :, 8, -1].numpy(), [-1, 1, -1], atol=0.01)
    np.testing.assert_allclose(pixels[0, 0, :, 0, 8].numpy(), [0, 0, 0], atol=0.01)
    vectors = encoder.encode([clip])
    assert vectors.shape == (1, 16)
    assert np.isfinite(vectors).all()
