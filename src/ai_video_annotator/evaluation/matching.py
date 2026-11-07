"""Score-ordered, one-to-one temporal event matching."""

from collections.abc import Sequence
from math import isfinite

import numpy as np

from ai_video_annotator.types import BehaviorEvent, RetrievalResult


def match_predictions(
    predictions: Sequence[RetrievalResult],
    truths: Sequence[BehaviorEvent],
    iou_threshold: float = 0.3,
) -> tuple[np.ndarray, tuple[int | None, ...]]:
    """Match each ranked prediction to its highest-IoU unmatched same-video truth.

    Input predictions must already be in descending score order. Truths must
    belong to one behavior. Duplicates count as false positives.
    """
    if not isfinite(iou_threshold) or not 0 < iou_threshold <= 1:
        raise ValueError("iou_threshold must be in (0, 1]")
    if len({truth.behavior for truth in truths}) > 1:
        raise ValueError("Evaluate one behavior at a time")
    if any(not isfinite(p.score) for p in predictions) or any(
        a.score < b.score for a, b in zip(predictions, predictions[1:], strict=False)
    ):
        raise ValueError("Predictions must have finite scores in descending order")
    used = set()
    matched = []
    for prediction in predictions:
        candidates = [
            (prediction.window.interval.temporal_iou(truth.interval), i)
            for i, truth in enumerate(truths)
            if i not in used and truth.video_id == prediction.window.video_id
        ]
        score, index = max(candidates, key=lambda pair: (pair[0], -pair[1]), default=(0, None))
        if score >= iou_threshold:
            used.add(index)
            matched.append(index)
        else:
            matched.append(None)
    return np.asarray([index is not None for index in matched], dtype=bool), tuple(matched)
