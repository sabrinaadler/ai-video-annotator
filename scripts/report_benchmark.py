"""Export recording-balanced benchmark tables and example-count learning curves."""

import argparse
import csv
import json
from pathlib import Path

from ai_video_annotator.evaluation.reporting import aggregate


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    rows = [{**row, "target_status": "all"} for row in aggregate(report)]
    for status, positive in (("positive", True), ("negative", False)):
        subset = {
            **report,
            "reports": [
                pair
                for pair in report["reports"]
                if (pair.get("target_event_count", 0) > 0) == positive
            ],
        }
        rows.extend({**row, "target_status": status} for row in aggregate(subset))
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "macro_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "macro_metrics.json").write_text(json.dumps(rows, indent=2, allow_nan=False))
    behaviors = sorted({row["behavior"] for row in rows})
    fig, axes = plt.subplots(2, len(behaviors), figsize=(13, 8), squeeze=False, sharey=True)
    colors = {"nearest-example": "#285846", "prototype": "#815208", "random": "#777777"}
    for i, protocol in enumerate(("within-video", "cross-video")):
        for j, behavior in enumerate(behaviors):
            ax = axes[i, j]
            for method, color in colors.items():
                subset = sorted(
                    [
                        r
                        for r in rows
                        if r["protocol"] == protocol
                        and r["evaluation_split"] == "test"
                        and r["target_status"] == "positive"
                        and r["behavior"] == behavior
                        and r["method"] == method
                        and r["iou_threshold"] == 0.3
                    ],
                    key=lambda r: r["examples"],
                )
                if not subset:
                    continue
                ax.plot(
                    [r["examples"] for r in subset],
                    [100 * r["precision_at_10"] for r in subset],
                    marker="o",
                    label=method,
                    color=color,
                    linestyle="--" if method == "random" else "-",
                )
                counts = ", ".join(
                    f"{r['examples']}→{r['target_recordings']}/{r['source_recordings']}"
                    for r in subset
                )
                if method == "nearest-example":
                    ax.text(
                        0.02,
                        0.98,
                        "Examples→targets/sources: " + counts,
                        transform=ax.transAxes,
                        va="top",
                        fontsize=8,
                    )
            ax.set_title(f"{behavior} · {protocol}")
            ax.set_xticks([1, 3, 5, 10])
            ax.set_ylim(0, 100)
            ax.grid(alpha=0.2)
            ax.set_xlabel("Labeled reference examples (frozen encoder)")
            if j == 0:
                ax.set_ylabel("Precision among first 10 proposals (%)")
    axes[0, 0].legend(fontsize=9, loc="lower right")
    fig.suptitle(
        "CalMS21 positive test-recording pilot · temporal IoU ≥ 0.3 · recording-balanced means",
        fontsize=15,
    )
    fig.text(
        0.5,
        0.015,
        "30 seed trials/pair; within-video uses a fixed held-out pool. "
        "Positive official-test targets only; no-behavior targets reported separately in CSV. "
        "Not a multi-dataset validation.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=[0, 0.055, 1, 0.95])
    fig.savefig(args.output / "learning_curves.png", dpi=180)
    fig.savefig(args.output / "learning_curves.pdf")


if __name__ == "__main__":
    main()
