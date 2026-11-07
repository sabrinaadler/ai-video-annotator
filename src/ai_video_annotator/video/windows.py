"""Generate deterministic overlapping intervals without model dependencies."""

from math import ceil, isfinite

from ai_video_annotator.types import TimeInterval, VideoWindow


def generate_windows(
    video_id: str,
    duration_seconds: float,
    window_seconds: float = 2.0,
    stride_seconds: float = 0.5,
) -> tuple[VideoWindow, ...]:
    """Cover the timeline with full windows and one end-aligned tail window.

    A video shorter than one window produces one clipped window. The tail may
    overlap more than the configured stride so that the final frames are covered.
    """
    for name, value in (
        ("duration_seconds", duration_seconds),
        ("window_seconds", window_seconds),
        ("stride_seconds", stride_seconds),
    ):
        if not isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if stride_seconds > window_seconds:
        raise ValueError("stride_seconds must not exceed window_seconds")
    last_start = max(0.0, duration_seconds - window_seconds)
    count = ceil(last_start / stride_seconds)
    starts = [min(index * stride_seconds, last_start) for index in range(count + 1)]
    return tuple(
        VideoWindow(
            video_id, index, TimeInterval(start, min(start + window_seconds, duration_seconds))
        )
        for index, start in enumerate(starts)
    )


def generate_multiscale_windows(video_id, duration_seconds, durations, stride_fraction=0.5):
    """Distinct, start/end sorted proposals at several actual temporal scales."""
    sizes = tuple(sorted(set(durations)))
    if not sizes or not isfinite(stride_fraction) or not 0 < stride_fraction <= 1:
        raise ValueError("Provide durations and a stride fraction in (0, 1]")
    intervals = {
        w.interval
        for size in sizes
        for w in generate_windows(video_id, duration_seconds, size, size * stride_fraction)
    }
    return tuple(
        VideoWindow(video_id, i, interval)
        for i, interval in enumerate(
            sorted(intervals, key=lambda interval: (interval.start_seconds, interval.end_seconds))
        )
    )
