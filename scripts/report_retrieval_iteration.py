"""Audit and report proposal/counterexample experiments, including fresh full video."""

import csv
import json
from collections import defaultdict
from pathlib import Path

from ai_video_annotator.embeddings.cache import load_cache
from ai_video_annotator.evaluation.reporting import aggregate
from ai_video_annotator.types import TimeInterval

DIRECTORY = Path("outputs/benchmarks")
STUDIES = {
    "fly_pilot": "retrieval_iteration",
    "training_alternative": "retrieval_development_prototype",
    "fresh_full_fly": "fresh_fly_validation",
    "mouse_all_durations": "retrieval_mouse_raw",
    "mouse_example_duration": "retrieval_mouse_adaptive",
}
NAMES = {
    "single_positive": "Fixed / nearest example",
    "single_negative": "Fixed / counterexamples",
    "multiscale_positive": "Multiple durations / nearest example",
    "multiscale_negative": "Multiple durations / counterexamples",
    "single_prototype": "Fixed / positive prototype",
    "multiscale_prototype": "Multiple durations / positive prototype",
    "single_random": "Fixed / random control",
    "multiscale_random": "Multiple durations / random control",
}


def audit(report):
    grouped = defaultdict(list)
    records = {r["id"]: r for r in report["manifest"]["recordings"]}
    for pair in report["reports"]:
        grouped[pair["source"], pair["target"], pair["behavior"]].append(pair)
    checked = negative_labels = 0
    for (source_id, target_id, behavior), pairs in grouped.items():
        source, target = records[source_id], records[target_id]
        expected = "train" if report["protocol"].get("development_only") else "test"
        assert source["split"] == "train" and target["split"] == expected
        baseline = next(p for p in pairs if p["condition"] == "single_positive")
        base_rows = {(r["shots"], r["trial"], r["iou_threshold"]): r for r in baseline["trials"]}
        cache = load_cache(source["cache"])
        blocked = [
            TimeInterval(e["start"], e["end"])
            for e in source["events"]
            if e["behavior"] == behavior
        ]
        blocked += [TimeInterval(i["start"], i["end"]) for i in source.get("ignore_intervals", [])]
        for pair in pairs:
            for row in pair["trials"]:
                base = base_rows[row["shots"], row["trial"], row["iou_threshold"]]
                for key in ("seed_event_indices", "reserved_event_indices", "truth_count"):
                    assert row[key] == base[key], f"Unpaired {key}"
                if pair["condition"].endswith("negative"):
                    ids = row["negative_source_window_indices"]
                    assert len(ids) == report["protocol"]["negative_count"]
                    assert row["additional_negative_labels"] == len(ids)
                    for i in ids:
                        assert not any(
                            cache.windows[i].interval.intersection_seconds(b) > 0 for b in blocked
                        ), "Counterexample contains target/unknown"
                    negative_labels += len(ids)
                checked += 1
        for policy in ("single", "multiscale"):
            reference = next(p for p in pairs if p["condition"] == policy + "_positive")
            coverage = {
                (r["shots"], r["trial"], r["iou_threshold"]): r["candidate_coverage"]
                for r in reference["trials"]
            }
            for pair in pairs:
                if pair["condition"].startswith(policy + "_"):
                    assert all(
                        r["candidate_coverage"]
                        == coverage[r["shots"], r["trial"], r["iou_threshold"]]
                        for r in pair["trials"]
                    )
    return {
        "paired_scored_rows": checked,
        "verified_source_only_negative_labels": negative_labels,
        "same_examples_truth_and_within_policy_coverage": True,
    }


def baseline_audit(current):
    old = json.loads((DIRECTORY / "encoder_comparison/flyvsfly_clip_mean.json").read_text())
    previous = {
        (p["source"], p["target"], p["behavior"]): p
        for p in old["reports"]
        if p["protocol"] == "cross-video"
    }
    changed, maximum = 0, 0.0
    for pair in current["reports"]:
        if pair["condition"] not in {"single_positive", "single_random"}:
            continue
        reference = previous[pair["source"], pair["target"], pair["behavior"]]
        method = "nearest-example" if pair["condition"] == "single_positive" else "random"
        rows = {
            (r["shots"], r["trial"], r["iou_threshold"]): r
            for r in reference["trials"]
            if r["method"] == method
        }
        for row in pair["trials"]:
            a = rows[row["shots"], row["trial"], row["iou_threshold"]]
            for key in (
                "seed_event_indices",
                "reserved_event_indices",
                "truth_count",
                "candidate_coverage",
            ):
                assert row[key] == a[key], f"Original baseline pairing changed: {key}"
            if method == "random":
                assert row["budgets"] == a["budgets"]
                assert row["average_precision"] == a["average_precision"]
            else:
                changed += row["budgets"] != a["budgets"]
                if a["average_precision"] is not None:
                    maximum = max(maximum, abs(row["average_precision"] - a["average_precision"]))
    return {
        "random_controls_exactly_reproduced": True,
        "positive_baseline_changed_budget_rows": changed,
        "positive_baseline_max_absolute_AP_delta": maximum,
    }


def main():
    rows, audits = [], {}
    for study, name in STUDIES.items():
        report = json.loads((DIRECTORY / f"{name}.json").read_text())
        audits[study] = audit(report)
        if study == "fly_pilot":
            audits["original_CLIP_baseline"] = baseline_audit(report)
        for row in aggregate(report):
            rows.append({"study": study, **row})
        del report
    output = DIRECTORY / "retrieval_iteration_report"
    output.mkdir(exist_ok=True)
    (output / "paired_audit.json").write_text(json.dumps(audits, indent=2))
    (output / "metrics.json").write_text(json.dumps(rows, indent=2, allow_nan=False))
    with (output / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    def pct(value):
        return "—" if value is None else f"{100 * value:.1f}"

    text = [
        """# Retrieval and workflow iteration — 18 October 2026

This iteration implements real duration-specific proposals, explicit counterexamples,
safe reranking, frame boundary controls, and resumable frame features. The first
fly experiment did **not** support enabling duration mixing or the tested negative
penalty by default. The application therefore starts with a fixed duration and
counterexample influence Off. Experimental controls are available explicitly.
The tables below include unfavorable results and the fresh full-recording test.

## Experiments and interpretation

All experiments use the frozen general CLIP ViT-B/16 checkpoint pinned in the prior
comparison. Source videos are official training recordings. The four-condition
ablation compares fixed proposals, a negative penalty, multiple durations, and both.
Thirty paired seed draws use nested 1/3/5/10 positive references. The penalty uses
three additional, simulated reviewed negative clips and weight 0.5. These negatives
are top-ranked source-only false positives with zero overlap with any source target
event or unknown region. No target labels are consulted to pick counterexamples.
This simulated review assumes reliable source labels; its annotation cost is not free.

The fly pilot reuses movie2/3 training and movie6/8 test excerpts of 120 seconds each.
Its baseline is 0.25 seconds at stride 0.125; duration mixing adds 0.125/0.5/1 seconds.
Actual frames are re-sampled for each duration. We do not shorten a timestamp while
reusing an old longer-window vector. Fixed source reference vectors stay identical
across the original four conditions, isolating changes in candidates/negative scoring.

The fresh test locks **all 30 minutes of movie7**, the first unused official test movie,
before reading its labels. It uses the same two training excerpts as sources.
The selection and method/default decision are saved before fresh scoring, and this
recording is not used for tuning. It contains 342 eligible lunge events, so reviewing
50 proposals can cover at most 14.6% even with perfect ranking and one match per event.
The fixed baseline retrieves 3.6% at that budget. Unknown-labeled intervals remain excluded. One
fresh recording is stronger evidence than another excerpt, but is not a general
insect/laboratory validation or an independent confidence interval.

The mouse experiment uses complete mouse025 training and mouse085 test recordings
already inspected in the earlier pilot; it is exploratory. The baseline is 2 seconds
at stride 0.5. Additional durations are 0.25/0.5/1/4 seconds with stride duration/4.
A second mouse experiment chooses one cached duration nearest the median positive
reference duration in log space. It also matches positive source features to that
duration; negative representations never expand past the confirmed-negative clip.

A separate alternative used only movie2/3 (both official training recordings):
example-based duration selection and a normalized positive-minus-negative prototype,
weight 1, at five examples. Its results did not support promoting that alternative.
The encoder is never trained or fine-tuned. No fresh-test hyperparameter search occurs.

## Five-reference results

Nearest-example conditions, event IoU at least 0.3. All metrics are percentages.
Precision@10 measures useful suggestions among the first ten; recall@50 measures
events found after reviewing fifty; F1@10 balances precision and recall at ten.
Coverage is a geometric upper bound before ranking/NMS, not achieved recall. AP uses
the full ranked proposal list. Means first average sources/trials per target, then
weight targets equally. F1 is not recomputed from macro precision/recall.
"""
    ]
    for study, title in (
        ("fly_pilot", "Earlier fly excerpts — two positive test targets"),
        ("fresh_full_fly", "Fresh full fly recording — one test target"),
        ("mouse_all_durations", "Mouse duration mixing — one previously inspected test target"),
        ("mouse_example_duration", "Mouse example-based duration — same exploratory target"),
    ):
        text.append(
            f"### {title}\n\n| Behavior | Condition | P10 | R10 | F1@10 | R50 | AP | Coverage |\n"
            "|---|---|---:|---:|---:|---:|---:|---:|"
        )
        for r in rows:
            if (
                r["study"] == study
                and r["examples"] == 5
                and r["iou_threshold"] == 0.3
                and r["method"]
                in {
                    "single_positive",
                    "single_negative",
                    "multiscale_positive",
                    "multiscale_negative",
                }
            ):
                label = NAMES[r["method"]]
                if study == "mouse_example_duration":
                    label = label.replace("Multiple durations", "Example-selected duration")
                text.append(
                    f"| {r['behavior']} | {label} | "
                    + " | ".join(
                        pct(r[k])
                        for k in (
                            "precision_at_10",
                            "recall_at_10",
                            "f1_at_10",
                            "recall_at_50",
                            "average_precision",
                            "candidate_coverage",
                        )
                    )
                    + " |"
                )
        text.append("")
    text.append("""The mouse duration-selected counterexample condition improves mount precision@10
from 8.0% to 26.0% in this exploratory recording. However, its duration-matched random
control reaches 17.0% precision@10 and 28.0% recall@50, versus 26.2% recall@50 for the
method. Part of the apparent gain comes from the duration prior and proposal geometry;
it is not evidence of better general behavior recognition. Neither duration mixing nor
counterexamples improved the fresh fly result, so neither becomes a default.

## Audit, artifacts and practical limits

The paired audit checks identical seed pools and evaluation truth across conditions,
identical candidate coverage within each duration policy, and the purity and training
origin of every negative label. It also checks reproduction of the old fixed-CLIP
random controls. Sparse event matching and NMS are tested against the original greedy
implementations, including ties and cross-video labels. They avoid quadratic memory
growth on the full recording. All model/bank/cache identities and source hashes are
verified before reuse; frame features save in independently validated chunks.

The application supports counterexample marking (X), explicitly enabled negative
influence, fixed/example-selected/all-duration options, frame boundary nudging,
original-proposal restoration, and repeated retrieval. Accepted/rejected/deferred
intervals and edited or noted pending intervals survive reranking. Both original and
edited bounds are excluded from new proposals. Score and per-search provenance stay
server-owned. JSON retains negative examples and search history; annotation CSV
contains positive examples, accepted suggestions, and manually coded state/point events
across all behaviors. The ethogram and behavior-specific review contexts persist in JSON.
Negative/positive conflicts
are rejected, saves are atomic, and revisions prevent stale-tab overwrites.

Browser QA used an isolated QA session with arbitrary test intervals, not ground-truth
autofilled annotations. It checked decisions, notes, boundary edits, counterexamples,
reranking and reload. The user preview starts with no labeled reference intervals.
Alt+Left/Right nudges the in point, Shift+Left/Right nudges the out point; normal
Left/Right steps playback. Nudge operations move to an actual next/previous frame
boundary. Apply or a review decision saves boundary changes.

There is no calibrated automatic confidence/abstention threshold. Forced top-K ranking
does not establish specificity or autonomous false alarms/hour. The human time saved
has not been measured: that requires a timed researcher study. Expert review of label
conversion and additional untouched animals, full negative-only recordings, and labs
remain necessary. Do not interpret repeated example draws as independent recordings.

Artifacts:

- [Paired audit](../outputs/benchmarks/retrieval_iteration_report/paired_audit.json)
- [All metrics CSV](../outputs/benchmarks/retrieval_iteration_report/metrics.csv)
- [All metrics JSON](../outputs/benchmarks/retrieval_iteration_report/metrics.json)
- [Learning curves PNG](../outputs/benchmarks/retrieval_iteration_report/learning_curves.png)
- [Learning curves PDF](../outputs/benchmarks/retrieval_iteration_report/learning_curves.pdf)
- Raw trials, manifests, selection locks and protocols remain under `outputs/benchmarks/`.

Reproduce the pilot:

```sh
export PYTHONPATH=src
export HF_HOME=outputs/model-cache
.venv/bin/python scripts/prepare_retrieval_iteration.py --device mps
.venv/bin/python scripts/benchmark_retrieval_iteration.py \\
  --manifest outputs/benchmarks/retrieval_iteration_manifest.json \\
  --output outputs/benchmarks/new_pilot_report.json
.venv/bin/python scripts/report_retrieval_iteration.py
```

The report generator requires all recorded studies above. Raw output paths are never
silently overwritten. The fresh test preparation uses `prepare_fresh_fly_validation.py`,
then the frame-bank preparer with its manifest. For mice, use baseline duration 2,
durations .25/.5/1/2/4 and stride fraction .25. `--adaptive-duration` selects example-based
duration; `--development` requires train-to-train pairs. Alternative scoring is exposed
as `--negative-method prototype-difference --negative-weight 1` for research, not the default.
""")
    Path("docs/retrieval_iteration_results.md").write_text("\n".join(text))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharey=True)
    tasks = [
        ("fly_pilot", "lunge", "Earlier fly excerpts"),
        ("fresh_full_fly", "lunge", "Fresh full fly recording"),
        ("mouse_all_durations", "mount", "Mouse mount"),
        ("mouse_all_durations", "investigation", "Mouse investigation"),
    ]
    colors = {
        "single_positive": "#263f37",
        "single_negative": "#a65a42",
        "multiscale_positive": "#477b9b",
        "multiscale_negative": "#91afc1",
    }
    for j, (study, behavior, title) in enumerate(tasks):
        for method, color in colors.items():
            selected = sorted(
                [
                    r
                    for r in rows
                    if r["study"] == study
                    and r["behavior"] == behavior
                    and r["method"] == method
                    and r["iou_threshold"] == 0.3
                ],
                key=lambda r: r["examples"],
            )
            for i, metric in enumerate(("precision_at_10", "recall_at_50")):
                axes[i, j].plot(
                    [r["examples"] for r in selected],
                    [100 * r[metric] for r in selected],
                    color=color,
                    marker="o",
                    label=NAMES[method],
                )
        if study.startswith("mouse"):
            for method, color in (
                ("multiscale_positive", "#788b3a"),
                ("multiscale_negative", "#b28e42"),
            ):
                selected = sorted(
                    [
                        r
                        for r in rows
                        if r["study"] == "mouse_example_duration"
                        and r["behavior"] == behavior
                        and r["method"] == method
                        and r["iou_threshold"] == 0.3
                    ],
                    key=lambda r: r["examples"],
                )
                for i, metric in enumerate(("precision_at_10", "recall_at_50")):
                    axes[i, j].plot(
                        [r["examples"] for r in selected],
                        [100 * r[metric] for r in selected],
                        color=color,
                        linestyle="--",
                        marker="o",
                        label=NAMES[method].replace("Multiple durations", "Example duration"),
                    )
        for i in range(2):
            axes[i, j].set_title(title)
            axes[i, j].set_xticks([1, 3, 5, 10])
            axes[i, j].set_xlabel("Positive reference examples")
            axes[i, j].set_ylim(0, 100)
            axes[i, j].grid(alpha=0.2)
    axes[0, 0].set_ylabel("Precision@10 (%)")
    axes[1, 0].set_ylabel("Recall@50 (%)")
    legend = {}
    for ax in axes.flat:
        handles, labels = ax.get_legend_handles_labels()
        legend.update(zip(labels, handles, strict=True))
    fig.legend(
        list(legend.values()),
        list(legend),
        loc="upper center",
        ncol=3,
        fontsize=8,
        bbox_to_anchor=(0.5, 0.94),
    )
    fig.suptitle("Proposal duration and counterexample experiments · event IoU ≥ 0.3")
    fig.text(
        0.5,
        0.01,
        "30 paired seed draws. Fly pilot: 2 test excerpts; fresh fly: 1 full recording; "
        "mice: 1 previously inspected target. Negative conditions add 3 source labels.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.87])
    fig.savefig(output / "learning_curves.png", dpi=180)
    fig.savefig(output / "learning_curves.pdf")
    print("All paired audits passed; results document, metrics and curves saved.", flush=True)


if __name__ == "__main__":
    main()
