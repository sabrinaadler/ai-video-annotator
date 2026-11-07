"""Event-level retrieval metrics for an already ranked result list."""

from collections.abc import Sequence

import numpy as np

from ai_video_annotator.evaluation.matching import match_predictions
from ai_video_annotator.types import BehaviorEvent, RetrievalResult


def retrieval_metrics(
    predictions: Sequence[RetrievalResult],
    truths: Sequence[BehaviorEvent],
    iou_threshold: float = 0.3,
) -> dict[str, float | int | None]:
    """P@K uses K as denominator even when fewer predictions are available.

    AP is the sum of precision at each true-positive rank divided by truth count.
    With no remaining truth, recall and AP are undefined and reported as None.
    """
    hits, _ = match_predictions(predictions, truths, iou_threshold)
    cumulative = np.cumsum(hits)
    ranks = np.arange(1, len(hits) + 1)
    result: dict[str, float | int | None] = {
        "truth_count": len(truths),
        "prediction_count": len(predictions),
        "average_precision": float(np.sum(cumulative / ranks * hits) / len(truths))
        if truths
        else None,
    }
    for k in (5, 10, 20):
        result[f"precision_at_{k}"] = int(hits[:k].sum()) / k
    for k in (10, 20, 50):
        result[f"recall_at_{k}"] = int(hits[:k].sum()) / len(truths) if truths else None
    return result
