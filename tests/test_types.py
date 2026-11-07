import pytest

from ai_video_annotator.types import BehaviorEvent, TimeInterval, VideoWindow


def test_interval_duration() -> None:
    interval = TimeInterval(2.5, 5.0)
    assert interval.duration_seconds == pytest.approx(2.5)


def test_interval_rejects_invalid_bounds() -> None:
    with pytest.raises(ValueError):
        TimeInterval(-1.0, 1.0)

    with pytest.raises(ValueError):
        TimeInterval(2.0, 2.0)


def test_temporal_iou() -> None:
    left = TimeInterval(10.0, 14.0)
    right = TimeInterval(12.0, 16.0)
    assert left.temporal_iou(right) == pytest.approx(2.0 / 6.0)


def test_non_overlapping_intervals_have_zero_iou() -> None:
    left = TimeInterval(0.0, 1.0)
    right = TimeInterval(2.0, 3.0)
    assert left.temporal_iou(right) == 0.0


def test_behavior_event_requires_identifiers() -> None:
    interval = TimeInterval(0.0, 1.0)

    with pytest.raises(ValueError):
        BehaviorEvent("", "mount", interval)

    with pytest.raises(ValueError):
        BehaviorEvent("video-1", "", interval)


def test_video_window_requires_non_negative_index() -> None:
    with pytest.raises(ValueError):
        VideoWindow("video-1", -1, TimeInterval(0.0, 1.0))


@pytest.mark.parametrize("start,end", [(float("nan"), 1), (0, float("inf")), (0, float("nan"))])
def test_interval_rejects_nonfinite_bounds(start, end):
    with pytest.raises(ValueError, match="finite"):
        TimeInterval(start, end)
