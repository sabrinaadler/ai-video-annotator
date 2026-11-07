"""Recording-balanced aggregation independent of datasets and encoders."""

from collections import defaultdict

import numpy as np


def aggregate(report):
    grouped = defaultdict(lambda: defaultdict(list))
    pair_counts = defaultdict(set)
    splits = {r["id"]: r.get("split", "unspecified") for r in report["manifest"]["recordings"]}
    for pair in report["reports"]:
        for row in pair["summary"]:
            key = (
                pair["protocol"],
                splits[pair["target"]],
                pair["behavior"],
                row["method"],
                row["shots"],
                row["iou_threshold"],
            )
            grouped[key][pair["target"]].append(row)
            pair_counts[key].add((pair["source"], pair["target"]))
    rows = []
    for (protocol, split, behavior, method, shots, threshold), targets in grouped.items():
        row = {
            "protocol": protocol,
            "evaluation_split": split,
            "behavior": behavior,
            "method": method,
            "examples": shots,
            "iou_threshold": threshold,
            "target_recordings": len(targets),
            "recording_pairs": len(
                pair_counts[(protocol, split, behavior, method, shots, threshold)]
            ),
            "source_recordings": len(
                {s for s, _ in pair_counts[(protocol, split, behavior, method, shots, threshold)]}
            ),
        }
        for metric in (
            "precision_at_10",
            "recall_at_10",
            "f1_at_10",
            "recall_at_50",
            "average_precision",
            "candidate_coverage",
        ):
            means = [
                np.mean([r[metric]["mean"] for r in group if r[metric] is not None])
                for group in targets.values()
                if any(r[metric] is not None for r in group)
            ]
            row[metric] = float(np.mean(means)) if means else None
            row[metric + "_recording_std"] = float(np.std(means)) if means else None
            row[metric + "_defined_targets"] = len(means)
        rows.append(row)
    return rows
