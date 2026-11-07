"""Audit paired trials and report recording-balanced encoder comparisons."""

import argparse
import csv
import json
from pathlib import Path

from ai_video_annotator.evaluation.reporting import aggregate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--directory", type=Path, default=Path("outputs/benchmarks/encoder_comparison")
    )
    args = parser.parse_args()
    inventory = json.loads((args.directory / "inventory.json").read_text())
    rows, audits = [], []
    for dataset in sorted({item["dataset"] for item in inventory}):
        items = [item for item in inventory if item["dataset"] == dataset]
        baseline = json.loads(
            Path(next(i["report"] for i in items if i["encoder"] == "videomae")).read_text()
        )
        for item in items:
            current = json.loads(Path(item["report"]).read_text())
            if len(current["reports"]) != len(baseline["reports"]):
                raise ValueError("Recording pair count mismatch")
            for reference, pair in zip(baseline["reports"], current["reports"], strict=True):
                for key in ("source", "target", "behavior", "eligible_seed_count", "skipped"):
                    if reference[key] != pair[key]:
                        raise ValueError(f"Unpaired comparison: {key}")
                if len(reference["trials"]) != len(pair["trials"]):
                    raise ValueError("Trial count mismatch")
                for a, b in zip(reference["trials"], pair["trials"], strict=True):
                    for key in (
                        "shots",
                        "trial",
                        "method",
                        "iou_threshold",
                        "seed_event_indices",
                        "reserved_event_indices",
                        "truth_count",
                        "candidate_coverage",
                    ):
                        if a[key] != b[key]:
                            raise ValueError(f"Unpaired trial: {key}")
                    if a["method"] == "random":
                        for key in ("average_precision", "budgets"):
                            if a[key] != b[key]:
                                raise ValueError("Random controls differ across encoders")
            audits.append(
                {
                    "dataset": dataset,
                    "encoder": item["encoder"],
                    "paired_examples_truth_coverage_random_controls": True,
                }
            )
            for status, pairs in (
                ("all", current["reports"]),
                ("positive", [p for p in current["reports"] if p["target_event_count"] > 0]),
                ("negative", [p for p in current["reports"] if p["target_event_count"] == 0]),
            ):
                for row in aggregate({**current, "reports": pairs}):
                    rows.append(
                        {
                            "dataset": dataset,
                            "encoder": item["encoder"],
                            "target_status": status,
                            **row,
                        }
                    )
            del current
        del baseline
    (args.directory / "paired_audit.json").write_text(json.dumps(audits, indent=2))
    (args.directory / "macro_metrics.json").write_text(json.dumps(rows, indent=2, allow_nan=False))
    with (args.directory / "macro_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    # Plot only the well-supported main behaviors, preserving dataset/task geometry.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tasks = [
        ("CalMS21", "attack"),
        ("CalMS21", "mount"),
        ("CalMS21", "investigation"),
        ("Fly-vs-Fly", "lunge"),
    ]
    colors = {
        "videomae": "#263f37",
        "dinov2_mean": "#356f9b",
        "dinov2_ordered4": "#83a2b7",
        "bioclip_mean": "#895720",
        "bioclip_ordered4": "#c7975f",
        "bioclip2_mean": "#8b4086",
        "bioclip2_ordered4": "#c68dc2",
        "clip_mean": "#b34449",
        "clip_ordered4": "#db9598",
        "resnet50_mean": "#426f68",
        "resnet50_ordered4": "#8bb2aa",
    }
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharey=True)
    for i, protocol in enumerate(("within-video", "cross-video")):
        for j, (dataset, behavior) in enumerate(tasks):
            ax = axes[i, j]
            for encoder, color in colors.items():
                subset = sorted(
                    [
                        r
                        for r in rows
                        if r["dataset"] == dataset
                        and r["encoder"] == encoder
                        and r["target_status"] == "positive"
                        and r["evaluation_split"] == "test"
                        and r["protocol"] == protocol
                        and r["behavior"] == behavior
                        and r["iou_threshold"] == 0.3
                        and r["method"] == "nearest-example"
                    ],
                    key=lambda r: r["examples"],
                )
                if not subset:
                    continue
                ax.plot(
                    [r["examples"] for r in subset],
                    [100 * r["precision_at_10"] for r in subset],
                    marker="o",
                    color=color,
                    label=encoder,
                    linestyle="--" if "ordered4" in encoder else "-",
                )
            control = sorted(
                [
                    r
                    for r in rows
                    if r["dataset"] == dataset
                    and r["encoder"] == "videomae"
                    and r["target_status"] == "positive"
                    and r["evaluation_split"] == "test"
                    and r["protocol"] == protocol
                    and r["behavior"] == behavior
                    and r["iou_threshold"] == 0.3
                    and r["method"] == "random"
                ],
                key=lambda r: r["examples"],
            )
            if control:
                ax.plot(
                    [r["examples"] for r in control],
                    [100 * r["precision_at_10"] for r in control],
                    color="#777777",
                    linestyle=":",
                    label="random control",
                )
            ax.set_title(f"{dataset} / {behavior}\n{protocol}", fontsize=11)
            ax.set_xticks([1, 3, 5, 10])
            ax.set_ylim(0, 100)
            ax.grid(alpha=0.2)
            ax.set_xlabel("Reference examples")
            if j == 0:
                ax.set_ylabel("Precision among first 10 proposals (%)")
    legend = {}
    for ax in axes.flat:
        handles, labels = ax.get_legend_handles_labels()
        legend.update(zip(labels, handles, strict=True))
    fig.legend(
        list(legend.values()),
        list(legend),
        fontsize=8,
        loc="upper center",
        ncol=5,
        bbox_to_anchor=(0.5, 0.935),
    )
    fig.suptitle("Frozen encoder comparison · nearest-example · event IoU ≥ 0.3", fontsize=16)
    fig.text(
        0.5,
        0.01,
        "30 paired seed trials; positive test targets only. "
        "Mouse windows 2s; insect windows 0.25s. "
        "Small pilot; absolute mouse/insect scores are different tasks.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=[0, 0.045, 1, 0.87])
    fig.savefig(args.directory / "encoder_learning_curves.png", dpi=180)
    fig.savefig(args.directory / "encoder_learning_curves.pdf")
    print("Paired audit, metrics and learning curves saved.", flush=True)


if __name__ == "__main__":
    main()
