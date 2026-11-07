"""Greedy temporal suppression of duplicate ranked suggestions."""

from collections.abc import Iterable
from math import isfinite

from ai_video_annotator.types import RetrievalResult


def temporal_nms(
    results: Iterable[RetrievalResult],
    iou_threshold: float = 0.3,
    top_k: int | None = None,
) -> tuple[RetrievalResult, ...]:
    """Suppress same-video intervals whose IoU exceeds the threshold.

    Scores descend, with ties resolved by video ID, start time, and window index.
    Truncate only after suppression so duplicates do not consume the result budget.
    """
    if not isfinite(iou_threshold) or not 0 <= iou_threshold <= 1:
        raise ValueError("iou_threshold must be between 0 and 1")
    if top_k is not None and (isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0):
        raise ValueError("top_k must be a positive integer")
    ranked = list(results)
    if any(not isfinite(result.score) for result in ranked):
        raise ValueError("Scores must be finite")
    ranked.sort(
        key=lambda r: (-r.score, r.window.video_id, r.window.interval.start_seconds, r.window.index)
    )
    kept: list[RetrievalResult] = []
    for result in ranked:
        if any(
            result.window.video_id == previous.window.video_id
            and result.window.interval.temporal_iou(previous.window.interval) > iou_threshold
            for previous in kept
        ):
            continue
        kept.append(result)
        if top_k is not None and len(kept) == top_k:
            break
    return tuple(kept)
