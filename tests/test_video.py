import numpy as np
import pytest

from ai_video_annotator.datasets.base import VideoMetadata
from ai_video_annotator.types import TimeInterval, VideoWindow
from ai_video_annotator.video import (
    generate_windows,
    iter_sampled_windows,
    sample_frame_indices,
)


def test_overlapping_windows():
    windows = generate_windows("mouse", 3, 2, 0.5)
    assert [(w.interval.start_seconds, w.interval.end_seconds) for w in windows] == [
        (0, 2),
        (0.5, 2.5),
        (1, 3),
    ]
    assert [w.index for w in windows] == [0, 1, 2]


def test_tail_and_short_video():
    windows = generate_windows("mouse", 3.2)
    assert windows[-1].interval.start_seconds == pytest.approx(1.2)
    assert windows[-1].interval.end_seconds == pytest.approx(3.2)
    assert generate_windows("mouse", 0.3)[0].interval == TimeInterval(0, 0.3)
    assert len(generate_windows("mouse", 2)) == 1


@pytest.mark.parametrize(
    "duration,window,stride",
    [
        (0, 2, 0.5),
        (3, 0, 0.5),
        (3, 2, 0),
        (3, 2, 3),
        (float("nan"), 2, 0.5),
        (3, float("inf"), 0.5),
    ],
)
def test_invalid_window_parameters(duration, window, stride):
    with pytest.raises(ValueError):
        generate_windows("mouse", duration, window, stride)


def test_uniform_sampling_excludes_end():
    indices = sample_frame_indices(TimeInterval(0.5, 2.5), 30, 90)
    assert len(indices) == 16
    assert indices[0] == 15
    assert indices[-1] == 74
    assert np.all(np.diff(indices) > 0)


def test_repeats_for_short_interval():
    indices = sample_frame_indices(TimeInterval(0, 1 / 30), 30, 1)
    np.testing.assert_array_equal(indices, np.zeros(16, dtype=np.int64))


def test_fractional_boundaries_and_final_frame():
    np.testing.assert_array_equal(
        sample_frame_indices(TimeInterval(0.11, 0.41), 10, 10, 3), [2, 3, 4]
    )
    assert sample_frame_indices(TimeInterval(0, 1), 10, 10)[-1] == 9


@pytest.mark.parametrize(
    "fps,count,samples", [(0, 10, 16), (30, 0, 16), (30, 10, 0), (30, 10, 1.5)]
)
def test_invalid_sampling_parameters(fps, count, samples):
    with pytest.raises(ValueError):
        sample_frame_indices(TimeInterval(0, 0.1), fps, count, samples)


def test_empty_and_out_of_bounds_sampling():
    with pytest.raises(ValueError, match="no frame"):
        sample_frame_indices(TimeInterval(0.01, 0.02), 30, 30)
    with pytest.raises(ValueError, match="exceeds"):
        sample_frame_indices(TimeInterval(0, 2), 30, 30)


@pytest.fixture
def video(tmp_path):
    import av

    path = tmp_path / "test.mp4"
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=10)
        stream.width, stream.height = 32, 32
        stream.pix_fmt = "yuv420p"
        for index in range(30):
            pixels = np.full((32, 32, 3), index * 7, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return VideoMetadata("mouse", path, 10, 30, 32, 32)


def test_sequential_reader_matches_independent_decode(video):
    import av

    with av.open(str(video.path)) as container:
        reference = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
    windows = generate_windows(video.video_id, video.duration_seconds)
    sampled = list(iter_sampled_windows(video, windows))
    assert len(sampled) == len(windows)
    for window, frames in sampled:
        indices = sample_frame_indices(window.interval, video.fps, video.frame_count)
        np.testing.assert_array_equal(frames, np.stack([reference[i] for i in indices]))
        assert frames.shape == (16, 32, 32, 3)
        assert frames.dtype == np.uint8


def test_reader_rejects_unordered_or_wrong_video(video):
    windows = generate_windows("mouse", 3)
    with pytest.raises(ValueError, match="ordered"):
        list(iter_sampled_windows(video, reversed(windows)))
    with pytest.raises(ValueError, match="video ID"):
        list(iter_sampled_windows(video, [VideoWindow("other", 0, TimeInterval(0, 1))]))


def test_reader_detects_truncated_video(video):
    incorrect = VideoMetadata("mouse", video.path, 10, 40, 32, 32)
    with pytest.raises(ValueError, match="ended"):
        list(iter_sampled_windows(incorrect, generate_windows("mouse", 4)))
