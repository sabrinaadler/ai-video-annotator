"""Benchmark prepared dataset-independent event manifests against verified caches."""

import argparse
import json
from pathlib import Path

from ai_video_annotator.datasets.event_csv import load_event_csv
from ai_video_annotator.embeddings.cache import load_cache, source_fingerprint
from ai_video_annotator.evaluation.benchmark import run_pair, summarize
from ai_video_annotator.types import BehaviorEvent, TimeInterval


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shots", type=int, nargs="+", default=[1, 3, 5, 10])
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; choose a new report path")
    manifest = json.loads(args.manifest.read_text())
    prepared = {}
    for record in manifest["recordings"]:
        if record["id"] in prepared:
            raise ValueError("Recording IDs must be unique")
        cache = load_cache(record["cache"])
        if cache.metadata["source"] != source_fingerprint(record["video"]):
            raise ValueError("Cache does not match original video")
        video_id = cache.windows[0].video_id
        if video_id != record["id"]:
            raise ValueError("Manifest ID must match the cache video ID")
        if "events_csv" in record:
            events = [e for e in load_event_csv(record["events_csv"]) if e.video_id == video_id]
        else:
            events = [
                BehaviorEvent(video_id, row["behavior"], TimeInterval(row["start"], row["end"]))
                for row in record["events"]
            ]
        if any(
            e.interval.end_seconds > max(w.interval.end_seconds for w in cache.windows)
            for e in events
        ):
            raise ValueError("Ground truth exceeds recording")
        prepared[record["id"]] = (cache, events)
    reports = []
    for pair in manifest["pairs"]:
        source, source_events = prepared[pair["source"]]
        target, target_events = prepared[pair["target"]]
        # Cross-video scores require exactly the same representation configuration.
        for key in (
            "model",
            "resolved_revision",
            "preprocessing",
            "window_seconds",
            "stride_seconds",
            "num_frames",
            "pooling",
            "sampling",
        ):
            if source.metadata.get(key) != target.metadata.get(key):
                raise ValueError(f"Incompatible caches: {key}")
        for behavior in manifest["behaviors"]:
            print(f"Evaluating {pair['source']} → {pair['target']} {behavior}…", flush=True)
            report = run_pair(
                source,
                target,
                [e for e in source_events if e.behavior == behavior],
                [e for e in target_events if e.behavior == behavior],
                shots=args.shots,
                trials=args.trials,
                seed=args.seed,
                source_ignore_intervals=tuple(
                    TimeInterval(i["start"], i["end"])
                    for i in next(
                        r for r in manifest["recordings"] if r["id"] == pair["source"]
                    ).get("ignore_intervals", [])
                ),
                target_ignore_intervals=tuple(
                    TimeInterval(i["start"], i["end"])
                    for i in next(
                        r for r in manifest["recordings"] if r["id"] == pair["target"]
                    ).get("ignore_intervals", [])
                ),
            )
            reports.append(
                {
                    **pair,
                    "behavior": behavior,
                    "source_event_count": sum(e.behavior == behavior for e in source_events),
                    "target_event_count": sum(e.behavior == behavior for e in target_events),
                    "source_cache_metadata": source.metadata,
                    "target_cache_metadata": target.metadata,
                    **report,
                    "summary": summarize(report["trials"]),
                }
            )
            print(
                f"{pair['source']} → {pair['target']} {behavior}: "
                f"{len(report['trials'])} evaluations",
                flush=True,
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "protocol": "fixed-pool-v1",
                "nms_iou": 0.3,
                "evaluation_iou": [0.3, 0.5],
                "review_budgets": [10, 20, 50],
                "shot_counts": args.shots,
                "seed_trials": args.trials,
                "seed": args.seed,
                "manifest": manifest,
                "reports": reports,
            },
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
