"""Paired duration/counterexample ablation; negative labels come only from sources."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from ai_video_annotator.embeddings.cache import load_cache, source_fingerprint
from ai_video_annotator.evaluation.benchmark import summarize
from ai_video_annotator.evaluation.reporting import aggregate
from ai_video_annotator.evaluation.temporal_index import (
    EventIndex,
    SuppressionIndex,
    metrics_from_hits,
)
from ai_video_annotator.retrieval.search import example_window_indices
from ai_video_annotator.retrieval.similarity import similarity_scores
from ai_video_annotator.types import BehaviorEvent, TimeInterval


def intersects_any(window, intervals):
    return any(window.interval.intersection_seconds(i) > 0 for i in intervals)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shots", type=int, nargs="+", default=[1, 3, 5, 10])
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--negative-count", type=int, default=3)
    parser.add_argument("--negative-weight", type=float, default=0.5)
    parser.add_argument(
        "--negative-method", choices=["penalty", "prototype-difference"], default="penalty"
    )
    parser.add_argument("--adaptive-duration", action="store_true")
    parser.add_argument("--development", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; choose a new result path")
    if args.trials < 1 or not args.shots or min(args.shots) < 1 or args.negative_count < 1:
        parser.error("Positive trials, shot counts and negative count required")
    if not np.isfinite(args.negative_weight) or not 0 <= args.negative_weight <= 1:
        parser.error("Negative weight must be in [0,1]")
    manifest = json.loads(args.manifest.read_text())
    protocol = {
        "name": "source-negative-multiscale-v1",
        "manifest": str(args.manifest),
        "shots": args.shots,
        "trials": args.trials,
        "seed": 42,
        "negative_count": args.negative_count,
        "negative_weight": args.negative_weight,
        "negative_method": args.negative_method,
        "adaptive_duration": args.adaptive_duration,
        "development_only": args.development,
        "negative_selection": "top source-only false positives; zero target-behavior overlap; "
        "exclude unknown regions; source ground truth simulates review",
        "query_policy": "match source features to selected duration"
        if args.adaptive_duration
        else "same single-duration source vectors for every condition",
        "evaluation": "cross-video only; all target events except declared ignores",
        "thresholds": [0.3, 0.5],
        "review_budgets": [10, 20, 50],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".protocol.json").write_text(json.dumps(protocol, indent=2))
    prepared = {}
    for record in manifest["recordings"]:
        caches = {
            "single": load_cache(record["cache"]),
            "multiscale": load_cache(record["multiscale_cache"]),
        }
        for cache in caches.values():
            if (
                cache.metadata["source"] != source_fingerprint(record["video"])
                or cache.windows[0].video_id != record["id"]
            ):
                raise ValueError("Recording/cache identity mismatch")
        for key in ("model", "resolved_revision", "pooling", "preprocessing", "num_frames"):
            if caches["single"].metadata[key] != caches["multiscale"].metadata[key]:
                raise ValueError(f"Representation mismatch: {key}")
        ignores = [TimeInterval(i["start"], i["end"]) for i in record.get("ignore_intervals", [])]
        events = [
            BehaviorEvent(record["id"], e["behavior"], TimeInterval(e["start"], e["end"]))
            for e in record["events"]
        ]
        events = [e for e in events if not intersects_any(e, ignores)]
        prepared[record["id"]] = (record, caches, events, ignores)
    reports = []
    for pair in manifest["pairs"]:
        if pair["protocol"] != "cross-video":
            continue
        source_record, source_caches, all_source_events, source_ignore = prepared[pair["source"]]
        target_record, target_caches, all_target_events, target_ignore = prepared[pair["target"]]
        expected_target = "train" if args.development else "test"
        if (
            source_record["split"] != "train"
            or target_record["split"] != expected_target
            or pair["source"] == pair["target"]
        ):
            raise ValueError("Cross-video examples must come from official training sources")
        source = source_caches["single"]
        for key in (
            "model",
            "resolved_revision",
            "preprocessing",
            "pooling",
            "num_frames",
            "window_sizes",
            "stride_fraction",
        ):
            for policy in ("single", "multiscale"):
                if source_caches[policy].metadata.get(key) != target_caches[policy].metadata.get(
                    key
                ):
                    raise ValueError(f"Cross-recording representation mismatch: {key}")
        for behavior in manifest["behaviors"]:
            source_events = [e for e in all_source_events if e.behavior == behavior]
            truth = [e for e in all_target_events if e.behavior == behavior]
            eligible = {}
            for i, event in enumerate(source_events):
                row = example_window_indices(source.windows, [event])[0]
                if not intersects_any(source.windows[row], source_ignore):
                    eligible.setdefault(row, i)
            available = [s for s in args.shots if s <= len(eligible)]
            if not available:
                continue
            source_suppression = SuppressionIndex(source.windows)
            negative_candidates = np.array(
                [
                    not intersects_any(w, [*source_ignore, *[e.interval for e in source_events]])
                    for w in source.windows
                ]
            )
            target_indices = {
                policy: (SuppressionIndex(cache.windows), EventIndex(cache.windows, truth))
                for policy, cache in target_caches.items()
            }
            target_candidates = {
                policy: np.array([not intersects_any(w, target_ignore) for w in cache.windows])
                for policy, cache in target_caches.items()
            }
            conditions = {
                name: []
                for name in (
                    "single_positive",
                    "single_negative",
                    "multiscale_positive",
                    "multiscale_negative",
                    "single_random",
                    "multiscale_random",
                    "single_prototype",
                    "multiscale_prototype",
                )
            }
            coverages = {
                (policy, t): index.coverage(target_candidates[policy], t)
                for policy, (_, index) in target_indices.items()
                for t in (0.3, 0.5)
            }
            adaptive = {}
            print(
                f"{pair['source']} → {pair['target']} / {behavior}: "
                f"{len(truth)} events, {len(eligible)} eligible seeds",
                flush=True,
            )
            for trial in range(args.trials):
                rng = np.random.default_rng(np.random.SeedSequence([42, trial]))
                pool = rng.choice(
                    list(eligible.values()), min(max(args.shots), len(eligible)), replace=False
                ).tolist()
                random_scores = {p: rng.random(len(c.windows)) for p, c in target_caches.items()}
                for shot in available:
                    seeds = example_window_indices(
                        source.windows, [source_events[i] for i in pool[:shot]]
                    )
                    positives = source.embeddings[list(seeds)]
                    source_scores = similarity_scores(source.embeddings, positives)
                    negative_rows = source_suppression.rank(
                        source_scores, negative_candidates, limit=args.negative_count
                    )
                    negatives = source.embeddings[negative_rows]
                    for policy, target in target_caches.items():
                        suppress, events_index = target_indices[policy]
                        candidates = target_candidates[policy]
                        query, counter = positives, negatives
                        selected_duration = None
                        if args.adaptive_duration and policy == "multiscale":
                            median = float(
                                np.median(
                                    [
                                        source_events[i].interval.duration_seconds
                                        for i in pool[:shot]
                                    ]
                                )
                            )
                            selected_duration = min(
                                target.metadata["window_sizes"],
                                key=lambda d: (abs(np.log(d / median)), d),
                            )
                            key = (policy, selected_duration)
                            if key not in adaptive:
                                durations = np.array(
                                    [w.interval.duration_seconds for w in target.windows]
                                )
                                mask = candidates & np.isclose(
                                    durations, selected_duration, atol=1e-9, rtol=0
                                )
                                adaptive[key] = (
                                    mask,
                                    {t: events_index.coverage(mask, t) for t in (0.3, 0.5)},
                                )
                            candidates = adaptive[key][0]
                            source_multi = source_caches["multiscale"]
                            query_rows = example_window_indices(
                                source_multi.windows,
                                [source_events[i] for i in pool[:shot]],
                                selected_duration,
                            )
                            negative_events = [
                                BehaviorEvent(pair["source"], behavior, source.windows[i].interval)
                                for i in negative_rows
                            ]
                            query = source_multi.embeddings[list(query_rows)]
                            if selected_duration <= source.metadata["window_seconds"]:
                                counter_rows = (
                                    example_window_indices(
                                        source_multi.windows, negative_events, selected_duration
                                    )
                                    if negative_events
                                    else ()
                                )
                                counter = source_multi.embeddings[list(counter_rows)]
                            # A longer scale must not expand a confirmed-negative clip
                            # into surrounding, unreviewed behavior. Keep its original vector.
                        for mode in ("positive", "negative", "random", "prototype"):
                            if mode == "negative" and len(negatives) < args.negative_count:
                                continue
                            scores = (
                                random_scores[policy]
                                if mode == "random"
                                else similarity_scores(
                                    target.embeddings,
                                    query,
                                    method="prototype"
                                    if mode == "prototype"
                                    else "nearest-example",
                                    negative_examples=counter if mode == "negative" else None,
                                    negative_weight=args.negative_weight,
                                    negative_method=args.negative_method,
                                )
                            )
                            ranked = suppress.rank(scores, candidates)
                            name = policy + "_" + mode
                            for threshold in (0.3, 0.5):
                                hits, matches = events_index.match(ranked, threshold)
                                cases = []
                                if trial == 0 and shot == 5:
                                    cases = [
                                        {
                                            "rank": k + 1,
                                            "prediction": asdict(target.windows[row]),
                                            "score": float(scores[row]),
                                            "matched_truth": asdict(truth[matches[k]])
                                            if matches[k] is not None
                                            else None,
                                        }
                                        for k, row in enumerate(ranked[:10])
                                    ]
                                conditions[name].append(
                                    {
                                        "shots": shot,
                                        "trial": trial,
                                        "method": name,
                                        "iou_threshold": threshold,
                                        "seed_event_indices": pool[:shot],
                                        "reserved_event_indices": pool,
                                        "negative_source_window_indices": negative_rows.tolist()
                                        if mode == "negative"
                                        else [],
                                        "additional_negative_labels": len(negative_rows)
                                        if mode == "negative"
                                        else 0,
                                        "candidate_coverage": adaptive[(policy, selected_duration)][
                                            1
                                        ][threshold]
                                        if selected_duration is not None
                                        else coverages[(policy, threshold)],
                                        "selected_duration": selected_duration,
                                        "sample_review": cases,
                                        **metrics_from_hits(hits, len(truth)),
                                    }
                                )
            for condition, rows in conditions.items():
                reports.append(
                    {
                        **pair,
                        "behavior": behavior,
                        "condition": condition,
                        "source_event_count": len(source_events),
                        "target_event_count": len(truth),
                        "eligible_seed_count": len(eligible),
                        "trials": rows,
                        "summary": summarize(rows),
                        "skipped": [s for s in args.shots if s not in available],
                    }
                )
    report = {"protocol": protocol, "manifest": manifest, "reports": reports}
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False))
    args.output.with_suffix(".summary.json").write_text(json.dumps(aggregate(report), indent=2))
    print(f"Saved paired ablation to {args.output}", flush=True)


if __name__ == "__main__":
    main()
