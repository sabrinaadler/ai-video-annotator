"""Paired few-shot benchmarks with fixed test truth and a random-ranking control."""

from collections import defaultdict
from dataclasses import asdict

import numpy as np

from ai_video_annotator.evaluation.matching import match_predictions
from ai_video_annotator.evaluation.metrics import retrieval_metrics
from ai_video_annotator.retrieval.search import example_window_indices
from ai_video_annotator.retrieval.similarity import similarity_scores
from ai_video_annotator.types import RetrievalResult


def budget_metrics(predictions, truths, threshold, budget):
    selected = predictions[:budget]
    hits, _ = match_predictions(selected, truths, threshold)
    tp = int(hits.sum())
    fp, fn = len(selected) - tp, len(truths) - tp
    precision = tp / len(selected) if selected else 0.0
    recall = tp / len(truths) if truths else None
    denominator = 2 * tp + fp + fn
    f1 = 2 * tp / denominator if denominator else None
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "reviewed": len(selected),
        "false_suggestions_per_hit": fp / tp if tp else None,
    }


def _suppression_matrix(windows, threshold):
    # Caches contain one recording. Reuse interval geometry across thousands of rankings.
    start = np.array([w.interval.start_seconds for w in windows])
    end = np.array([w.interval.end_seconds for w in windows])
    intersection = np.maximum(0, np.minimum(end[:, None], end) - np.maximum(start[:, None], start))
    union = (end - start)[:, None] + (end - start) - intersection
    return intersection / union > threshold


def _rank(windows, scores, candidates, suppression):
    indices = np.array([w.index for w in windows])
    start = np.array([w.interval.start_seconds for w in windows])
    order = np.lexsort((indices, start, -scores))
    remaining = np.zeros(len(windows), dtype=bool)
    remaining[candidates] = True
    ranked = []
    for i in order:
        if remaining[i]:
            ranked.append(RetrievalResult(windows[i], float(scores[i])))
            remaining[suppression[i]] = False
            remaining[i] = False
    return tuple(ranked)


def run_pair(
    source,
    target,
    source_events,
    target_events,
    *,
    shots=(1, 3, 5, 10),
    trials=30,
    seed=42,
    thresholds=(0.3, 0.5),
    budgets=(10, 20, 50),
    nms_iou=0.3,
    source_ignore_intervals=(),
    target_ignore_intervals=(),
):
    """Within-video: reserve a fixed max-shot pool for every shot count in a trial.

    Truth and candidate exclusions are identical across shots/methods. This controlled
    experiment deliberately withholds unused pool examples from scoring and testing.
    Across-video: examples come only from source, all target truth is held out.
    """
    if not shots or min(shots) < 1 or trials < 1:
        raise ValueError("Positive shots and trials required")
    within = source.windows[0].video_id == target.windows[0].video_id
    eligible = {}
    for i, event in enumerate(source_events):
        row = example_window_indices(source.windows, [event])[0]
        if not any(
            source.windows[row].interval.intersection_seconds(interval) > 0
            or event.interval.intersection_seconds(interval) > 0
            for interval in source_ignore_intervals
        ):
            eligible.setdefault(row, i)
    reserve = min(max(shots), len(eligible) - int(within))
    available = [shot for shot in shots if shot <= reserve]
    skipped = [
        {"shots": shot, "reason": "Insufficient distinct source examples/test events"}
        for shot in shots
        if shot not in available
    ]
    rows = []
    if not available:
        return {"trials": rows, "skipped": skipped, "eligible_seed_count": len(eligible)}
    suppression = _suppression_matrix(target.windows, nms_iou)
    for trial in range(trials):
        rng = np.random.default_rng(np.random.SeedSequence([seed, trial]))
        pool = rng.choice(list(eligible.values()), reserve, replace=False).tolist()
        truth = [event for i, event in enumerate(target_events) if not within or i not in pool]
        excluded = [source_events[i].interval for i in pool] if within else []
        excluded += (
            [
                source.windows[
                    example_window_indices(source.windows, [source_events[i]])[0]
                ].interval
                for i in pool
            ]
            if within
            else []
        )
        excluded.extend(target_ignore_intervals)
        truth = [
            event
            for event in truth
            if not any(
                event.interval.intersection_seconds(interval) > 0
                for interval in target_ignore_intervals
            )
        ]
        candidates = [
            i
            for i, window in enumerate(target.windows)
            if not any(window.interval.intersection_seconds(interval) > 0 for interval in excluded)
        ]
        coverage = {
            str(t): sum(
                any(
                    target.windows[i].interval.temporal_iou(event.interval) >= t for i in candidates
                )
                for event in truth
            )
            / len(truth)
            if truth
            else None
            for t in thresholds
        }
        random_scores = rng.random(len(target.windows))
        for shot in available:
            examples = [source_events[i] for i in pool[:shot]]
            indices = example_window_indices(source.windows, examples)
            for method in ("nearest-example", "prototype", "random"):
                scores = (
                    random_scores
                    if method == "random"
                    else similarity_scores(
                        target.embeddings, source.embeddings[list(indices)], method
                    )
                )
                predictions = _rank(target.windows, scores, candidates, suppression)
                for threshold in thresholds:
                    review_cases = []
                    if trial == 0 and shot == 5:
                        _, matches = match_predictions(predictions[:10], truth, threshold)
                        review_cases = [
                            {
                                "rank": rank + 1,
                                "prediction": asdict(prediction),
                                "matched_truth": asdict(truth[index])
                                if index is not None
                                else None,
                                "best_truth_iou": max(
                                    (
                                        prediction.window.interval.temporal_iou(event.interval)
                                        for event in truth
                                    ),
                                    default=0.0,
                                ),
                            }
                            for rank, (prediction, index) in enumerate(
                                zip(predictions[:10], matches, strict=True)
                            )
                        ]
                    rows.append(
                        {
                            "shots": shot,
                            "trial": trial,
                            "method": method,
                            "iou_threshold": threshold,
                            "seed_event_indices": pool[:shot],
                            "reserved_event_indices": pool,
                            "examples": [asdict(event) for event in examples],
                            "sample_review": review_cases,
                            "candidate_coverage": coverage[str(threshold)],
                            **retrieval_metrics(predictions, truth, threshold),
                            "budgets": {
                                str(k): budget_metrics(predictions, truth, threshold, k)
                                for k in budgets
                            },
                        }
                    )
    return {"trials": rows, "skipped": skipped, "eligible_seed_count": len(eligible)}


def summarize(rows):
    """Trial variability within each recording pair, not a population confidence interval."""
    groups = defaultdict(list)
    for row in rows:
        key = (row["shots"], row["method"], row["iou_threshold"])
        groups[key].append(row)
    result = []
    for (shots, method, threshold), group in groups.items():
        metrics = {
            "average_precision": [row["average_precision"] for row in group],
            "candidate_coverage": [row["candidate_coverage"] for row in group],
        }
        for k in group[0]["budgets"]:
            for metric in ("precision", "recall", "f1", "tp", "fp", "fn"):
                metrics[f"{metric}_at_{k}"] = [row["budgets"][k][metric] for row in group]
        aggregate = {}
        for name, values in metrics.items():
            defined = [v for v in values if v is not None]
            aggregate[name] = (
                {"mean": float(np.mean(defined)), "std": float(np.std(defined))}
                if defined
                else None
            )
        result.append(
            {
                "shots": shots,
                "method": method,
                "iou_threshold": threshold,
                "trial_count": len(group),
                **aggregate,
            }
        )
    return result
