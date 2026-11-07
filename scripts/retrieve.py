"""Retrieve ranked timestamps using only cached embeddings and researcher examples."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from ai_video_annotator.embeddings import load_cache
from ai_video_annotator.retrieval import example_window_indices, retrieve
from ai_video_annotator.types import BehaviorEvent, TimeInterval


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument(
        "--examples",
        required=True,
        type=Path,
        help="JSON array of {start_seconds, end_seconds} objects",
    )
    parser.add_argument("--behavior", required=True)
    parser.add_argument(
        "--method", default="nearest-example", choices=["nearest-example", "prototype"]
    )
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--nms-iou", type=float, default=0.3)
    parser.add_argument(
        "--output", type=Path, help="Save JSON report; existing files are preserved"
    )
    args = parser.parse_args()
    try:
        cache = load_cache(args.cache)
        video_id = cache.windows[0].video_id
        raw = json.loads(args.examples.read_text(encoding="utf-8"))
        if not isinstance(raw, list) or not raw or any(not isinstance(row, dict) for row in raw):
            raise ValueError("Examples must be a nonempty JSON array of interval objects")
        examples = tuple(
            BehaviorEvent(
                video_id,
                args.behavior,
                TimeInterval(float(row["start_seconds"]), float(row["end_seconds"])),
            )
            for row in raw
        )
        duration = max(window.interval.end_seconds for window in cache.windows)
        if any(example.interval.end_seconds > duration for example in examples):
            raise ValueError("Example exceeds the cached video timeline")
        selected = example_window_indices(cache.windows, examples)
        results = retrieve(
            cache.embeddings, cache.windows, examples, args.method, args.nms_iou, args.top_k
        )
        report = {
            "video_id": video_id,
            "behavior": args.behavior,
            "method": args.method,
            "nms_iou_threshold": args.nms_iou,
            "requested_top_k": args.top_k,
            "examples": [asdict(example) for example in examples],
            "selected_window_indices": list(selected),
            "results": [asdict(result) for result in results],
        }
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as handle:
                json.dump(report, handle, indent=2)
        print(
            f"{args.behavior}: {len(examples)} examples, {len(results)} suggestions ({args.method})"
        )
        for rank, result in enumerate(results, 1):
            interval = result.window.interval
            print(
                f"{rank:2}. {interval.start_seconds:.2f}–{interval.end_seconds:.2f} s "
                f"score={result.score:.4f}"
            )
        if args.output:
            print(f"Saved report to {args.output}")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
