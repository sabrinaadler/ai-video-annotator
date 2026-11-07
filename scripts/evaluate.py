"""Repeated within-video few-shot experiments using cached embeddings."""

import argparse
import json
from pathlib import Path

import numpy as np

from ai_video_annotator.datasets import CalMS21Dataset
from ai_video_annotator.embeddings import load_cache, source_fingerprint
from ai_video_annotator.evaluation import retrieval_metrics
from ai_video_annotator.retrieval import example_window_indices, retrieve


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--videos-root", type=Path, required=True)
    parser.add_argument("--behavior", default="mount")
    parser.add_argument("--shots", type=int, nargs="+", default=[1, 3, 5, 10])
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nms-iou", type=float, default=0.3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.trials <= 0 or any(shot <= 0 for shot in args.shots):
            raise ValueError("Trials and shot counts must be positive")
        if args.output.exists():
            raise FileExistsError(f"Output exists: {args.output}")
        cache = load_cache(args.cache)
        video_id = cache.windows[0].video_id
        dataset = CalMS21Dataset(args.annotations, args.videos_root)
        metadata = dataset.video_metadata(video_id)
        if source_fingerprint(metadata.path) != cache.metadata.get("source"):
            raise ValueError("Cache source does not match the annotated recording")
        events = dataset.behavior_events(video_id, args.behavior)
        # One seed per distinct cached row; short nearby bouts can map to the same row.
        eligible = {}
        for i, event in enumerate(events):
            index = example_window_indices(cache.windows, [event])[0]
            eligible.setdefault(index, i)
        if len(eligible) <= max(args.shots):
            raise ValueError(
                f"Only {len(eligible)} distinct seed windows; "
                "choose fewer shots or another recording"
            )
        rng = np.random.default_rng(args.seed)
        rows = []
        for shot in args.shots:
            for trial in range(args.trials):
                chosen = sorted(rng.choice(list(eligible.values()), shot, replace=False).tolist())
                examples = [events[i] for i in chosen]
                # All nonseed truth remains in denominator, including excluded neighborhoods.
                truth = [event for i, event in enumerate(events) if i not in chosen]
                for method in ("nearest-example", "prototype"):
                    predictions = retrieve(
                        cache.embeddings,
                        cache.windows,
                        examples,
                        method,
                        args.nms_iou,
                        top_k=len(cache.windows),
                    )
                    for threshold in (0.3, 0.5):
                        rows.append(
                            {
                                "shots": shot,
                                "trial": trial,
                                "method": method,
                                "iou_threshold": threshold,
                                "seed_event_indices": chosen,
                                **retrieval_metrics(predictions, truth, threshold),
                            }
                        )
        summary = []
        metrics = [key for key in rows[0] if key.startswith(("precision_", "recall_", "average_"))]
        for shot in args.shots:
            for method in ("nearest-example", "prototype"):
                for threshold in (0.3, 0.5):
                    subset = [
                        row
                        for row in rows
                        if row["shots"] == shot
                        and row["method"] == method
                        and row["iou_threshold"] == threshold
                    ]
                    aggregate = {"shots": shot, "method": method, "iou_threshold": threshold}
                    for key in metrics:
                        values = [row[key] for row in subset if row[key] is not None]
                        aggregate[key] = (
                            {"mean": float(np.mean(values)), "std": float(np.std(values))}
                            if values
                            else None
                        )
                    summary.append(aggregate)
        report = {
            "video_id": video_id,
            "behavior": args.behavior,
            "seed": args.seed,
            "nms_iou": args.nms_iou,
            "truth_count": len(events),
            "eligible_seed_count": len(eligible),
            "cache_metadata": cache.metadata,
            "trials": rows,
            "summary": summary,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        print(f"Saved {len(rows)} evaluations for {len(events)} events to {args.output}")
        for row in summary:
            print(
                f"{row['shots']:2} shots {row['method']:15} IoU={row['iou_threshold']} "
                f"P@10={row['precision_at_10']['mean']:.3f} "
                f"AP={row['average_precision']['mean']:.3f}"
            )
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
