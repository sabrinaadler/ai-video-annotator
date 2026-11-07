import numpy as np
import pytest

from ai_video_annotator.embeddings.cache import EmbeddingCache
from ai_video_annotator.evaluation.benchmark import budget_metrics, run_pair
from ai_video_annotator.types import BehaviorEvent, RetrievalResult, TimeInterval, VideoWindow


def fixture(video_id):
    windows = tuple(VideoWindow(video_id, i, TimeInterval(i * 4, i * 4 + 2)) for i in range(8))
    cache = EmbeddingCache(np.array([[1, 0]] * 4 + [[0, 1]] * 4, dtype=np.float32), windows, {})
    events = [BehaviorEvent(video_id, "mount", w.interval) for w in windows[:4]]
    return cache, events


def test_budget_counts_duplicates_and_missing_events():
    cache, truth = fixture("target")
    predictions = [
        RetrievalResult(cache.windows[0], 1),
        RetrievalResult(cache.windows[0], 0.9),
        RetrievalResult(cache.windows[5], 0.8),
    ]
    metrics = budget_metrics(predictions, truth, 0.5, 10)
    assert (metrics["tp"], metrics["fp"], metrics["fn"]) == (1, 2, 3)
    assert metrics["precision"] == pytest.approx(1 / 3)
    assert metrics["recall"] == 0.25
    assert metrics["f1"] == pytest.approx(2 / 7)


def test_fixed_truth_nested_examples_and_paired_random_baseline():
    cache, events = fixture("within")
    report = run_pair(cache, cache, events, events, shots=(1, 2), trials=2)
    rows = report["trials"]
    for trial in range(2):
        subset = [row for row in rows if row["trial"] == trial]
        assert {row["truth_count"] for row in subset} == {2}
        assert len({tuple(row["reserved_event_indices"]) for row in subset}) == 1
        one = next(row for row in subset if row["shots"] == 1)
        two = next(row for row in subset if row["shots"] == 2)
        assert two["seed_event_indices"][:1] == one["seed_event_indices"]
        random = [r for r in subset if r["method"] == "random" and r["iou_threshold"] == 0.5]
        assert random[0]["budgets"] == random[1]["budgets"]
        assert all(r["average_precision"] == 1 for r in subset if r["method"] != "random")


def test_cross_video_keeps_all_target_truth_and_reports_missing_shots():
    source, examples = fixture("source")
    target, truth = fixture("target")
    report = run_pair(source, target, examples, truth, shots=(1, 5), trials=1)
    assert report["skipped"][0]["shots"] == 5
    assert all(row["truth_count"] == 4 for row in report["trials"])
    assert all(
        row["average_precision"] == 1 for row in report["trials"] if row["method"] != "random"
    )


def test_no_positive_target_measures_false_alarms():
    cache, _ = fixture("negative")
    metrics = budget_metrics([RetrievalResult(cache.windows[0], 1)], [], 0.5, 10)
    assert metrics["fp"] == 1
    assert metrics["recall"] is None
    assert metrics["f1"] == 0
    assert budget_metrics([], [], 0.5, 10)["f1"] is None


def test_reused_suppression_geometry_matches_core_nms():
    from ai_video_annotator.evaluation.benchmark import _rank, _suppression_matrix
    from ai_video_annotator.retrieval.postprocess import temporal_nms
    from ai_video_annotator.video.windows import generate_windows

    windows = generate_windows("video", 17.3, 2, 0.5)
    rng = np.random.default_rng(7)
    for threshold in (0, 0.3, 0.5, 1):
        scores = rng.integers(0, 5, len(windows)).astype(float)
        candidates = list(range(0, len(windows), 2))
        expected = temporal_nms(
            [RetrievalResult(windows[i], scores[i]) for i in candidates], threshold
        )
        actual = _rank(windows, scores, candidates, _suppression_matrix(windows, threshold))
        assert actual == expected


def test_macro_report_balances_targets_and_separates_splits():
    import runpy

    aggregate = runpy.run_path("scripts/report_benchmark.py")["aggregate"]
    report = {
        "manifest": {
            "recordings": [
                {"id": "a", "split": "test"},
                {"id": "b", "split": "test"},
                {"id": "development", "split": "train"},
            ]
        },
        "reports": [],
    }
    for source, target, value in [
        ("s1", "a", 0.6),
        ("s2", "a", 0.7),
        ("s3", "a", 0.8),
        ("s1", "b", 0.1),
        ("development", "development", 0.99),
    ]:
        row = {"method": "nearest-example", "shots": 5, "iou_threshold": 0.3}
        for key in (
            "precision_at_10",
            "recall_at_10",
            "f1_at_10",
            "recall_at_50",
            "average_precision",
            "candidate_coverage",
        ):
            row[key] = {"mean": value, "std": 0}
        report["reports"].append(
            {
                "protocol": "cross-video",
                "behavior": "mount",
                "source": source,
                "target": target,
                "summary": [row],
            }
        )
    rows = aggregate(report)
    held_out = next(r for r in rows if r["evaluation_split"] == "test")
    assert held_out["precision_at_10"] == pytest.approx(0.4)
    assert held_out["target_recordings"] == 2
    assert held_out["recording_pairs"] == 4
    assert held_out["source_recordings"] == 3


def test_ignored_regions_neither_seed_examples_nor_scored_truth():
    source, examples = fixture("source")
    target, truth = fixture("target")
    report = run_pair(
        source,
        target,
        examples,
        truth,
        shots=(1,),
        trials=1,
        source_ignore_intervals=(TimeInterval(0, 2),),
        target_ignore_intervals=(TimeInterval(0, 2),),
    )
    assert report["eligible_seed_count"] == 3
    assert all(row["truth_count"] == 3 for row in report["trials"])
    assert all(row["seed_event_indices"][0] != 0 for row in report["trials"])
    assert all(row["prediction_count"] == 7 for row in report["trials"])
