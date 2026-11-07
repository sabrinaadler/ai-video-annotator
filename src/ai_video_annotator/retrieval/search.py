"""Map researcher examples to cached windows and retrieve unseen intervals."""

from collections.abc import Sequence
from math import isfinite

import numpy as np

from ai_video_annotator.retrieval.postprocess import temporal_nms
from ai_video_annotator.retrieval.similarity import similarity_scores
from ai_video_annotator.types import BehaviorEvent, RetrievalResult, TimeInterval, VideoWindow


def example_window_indices(
    windows: Sequence[VideoWindow],
    examples: Sequence[BehaviorEvent],
    reference_window_seconds: float | None = None,
) -> tuple[int, ...]:
    """Select one maximum-IoU cached window per example; break ties by row order."""
    if not windows or not examples:
        raise ValueError("Provide cached windows and at least one labeled example")
    if len({event.behavior for event in examples}) != 1:
        raise ValueError("Examples must describe one behavior")
    selected = []
    allowed = np.ones(len(windows), bool)
    if reference_window_seconds is not None:
        if not isfinite(reference_window_seconds) or reference_window_seconds <= 0:
            raise ValueError("Reference window duration must be positive")
        distances = np.array(
            [abs(w.interval.duration_seconds - reference_window_seconds) for w in windows]
        )
        allowed = np.isclose(distances, distances.min(), atol=1e-9, rtol=0)
    for example in examples:
        overlaps = [
            window.interval.temporal_iou(example.interval)
            if window.video_id == example.video_id and allowed[i]
            else 0.0
            for i, window in enumerate(windows)
        ]
        index = int(np.argmax(overlaps))
        if overlaps[index] == 0:
            raise ValueError("Example does not overlap a cached window in its video")
        selected.append(index)
    if len(set(selected)) != len(selected):
        raise ValueError("Multiple examples map to the same cached window; choose distinct events")
    return tuple(selected)


def retrieve(
    embeddings: np.ndarray,
    windows: Sequence[VideoWindow],
    examples: Sequence[BehaviorEvent],
    method: str = "nearest-example",
    nms_iou_threshold: float = 0.3,
    top_k: int = 20,
    *,
    negative_examples: Sequence[BehaviorEvent] = (),
    negative_weight: float = 0.5,
    excluded_intervals: Sequence[TimeInterval] = (),
    reference_window_seconds: float | None = None,
    candidate_window_seconds: float | None = None,
) -> tuple[RetrievalResult, ...]:
    """Rank unseen candidates, excluding all overlaps with labels and selected windows."""
    if embeddings.ndim != 2 or len(embeddings) != len(windows):
        raise ValueError("Embedding rows must correspond to cached windows")
    indices = example_window_indices(windows, examples, reference_window_seconds)
    negative_indices = []
    if negative_examples and negative_weight:
        for example in negative_examples:
            contained = [
                i
                for i, w in enumerate(windows)
                if w.video_id == example.video_id
                and w.interval.start_seconds >= example.interval.start_seconds - 1e-9
                and w.interval.end_seconds <= example.interval.end_seconds + 1e-9
            ]
            if not contained:
                raise ValueError(
                    "Counterexample needs a complete cached window inside its bounds; "
                    "mark a longer absent interval or turn counterexample influence off"
                )
            if reference_window_seconds is not None:
                distance = min(
                    abs(windows[i].interval.duration_seconds - reference_window_seconds)
                    for i in contained
                )
                contained = [
                    i
                    for i in contained
                    if abs(
                        abs(windows[i].interval.duration_seconds - reference_window_seconds)
                        - distance
                    )
                    < 1e-9
                ]
            negative_indices.append(
                max(
                    contained,
                    key=lambda i: (windows[i].interval.temporal_iou(example.interval), -i),
                )
            )
        if len(set(negative_indices)) != len(negative_indices):
            raise ValueError("Choose counterexamples in distinct cached windows")
    if set(indices) & set(negative_indices) or any(
        a.video_id == b.video_id and a.interval.intersection_seconds(b.interval) > 0
        for a in examples
        for b in negative_examples
    ):
        raise ValueError("Positive and negative examples must describe distinct intervals/windows")
    scores = similarity_scores(
        embeddings,
        embeddings[list(indices)],
        method,
        negative_examples=embeddings[list(negative_indices)] if negative_indices else None,
        negative_weight=negative_weight,
    )
    exclusions = [
        (example.video_id, example.interval) for example in (*examples, *negative_examples)
    ]
    exclusions.extend((windows[index].video_id, windows[index].interval) for index in indices)
    exclusions.extend((windows[i].video_id, windows[i].interval) for i in negative_indices)
    exclusions.extend((windows[0].video_id, interval) for interval in excluded_intervals)
    allowed_candidates = np.ones(len(windows), bool)
    if candidate_window_seconds is not None:
        if not isfinite(candidate_window_seconds) or candidate_window_seconds <= 0:
            raise ValueError("Candidate window duration must be positive")
        distances = np.array(
            [abs(w.interval.duration_seconds - candidate_window_seconds) for w in windows]
        )
        allowed_candidates = np.isclose(distances, distances.min(), atol=1e-9, rtol=0)
    results = (
        RetrievalResult(window, float(score))
        for i, (window, score) in enumerate(zip(windows, scores, strict=True))
        if allowed_candidates[i]
        and not any(
            window.video_id == video_id and window.interval.intersection_seconds(interval) > 0
            for video_id, interval in exclusions
        )
    )
    return temporal_nms(results, nms_iou_threshold, top_k)
