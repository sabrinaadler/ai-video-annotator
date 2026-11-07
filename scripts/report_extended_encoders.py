"""Write the insect extension report from audited recording-balanced metrics."""

import json
from collections import defaultdict
from pathlib import Path

NAMES = {
    "videomae": "VideoMAE Base",
    "dinov2_mean": "DINOv2 Small mean",
    "dinov2_ordered4": "DINOv2 Small ordered4",
    "bioclip_mean": "BioCLIP mean",
    "bioclip_ordered4": "BioCLIP ordered4",
    "clip_mean": "General CLIP mean",
    "clip_ordered4": "General CLIP ordered4",
    "bioclip2_mean": "BioCLIP 2 mean",
    "bioclip2_ordered4": "BioCLIP 2 ordered4",
    "resnet50_mean": "ResNet-50 mean",
    "resnet50_ordered4": "ResNet-50 ordered4",
}


def main():
    directory = Path("outputs/benchmarks/encoder_comparison")
    rows = json.loads((directory / "macro_metrics.json").read_text())
    audit = json.loads((directory / "paired_audit.json").read_text())
    checked = {
        a["encoder"]
        for a in audit
        if a["dataset"] == "Fly-vs-Fly" and a["paired_examples_truth_coverage_random_controls"]
    }
    if not {e for e in NAMES if not e.startswith("bioclip2")}.issubset(checked):
        raise ValueError("Run the full insect extension and paired report audit first")
    names = {encoder: label for encoder, label in NAMES.items() if encoder in checked}

    def row(encoder, protocol, shots=5, threshold=0.3, method="nearest-example"):
        return next(
            r
            for r in rows
            if r["dataset"] == "Fly-vs-Fly"
            and r["encoder"] == encoder
            and r["behavior"] == "lunge"
            and r["target_status"] == "positive"
            and r["evaluation_split"] == "test"
            and r["protocol"] == protocol
            and r["examples"] == shots
            and r["iou_threshold"] == threshold
            and r["method"] == method
        )

    def pct(value):
        return "—" if value is None else f"{100 * value:.1f}"

    summary = "\n".join(
        f"- {NAMES[encoder]}: P10 **{pct(row(encoder, 'cross-video')['precision_at_10'])}%**, "
        f"R50 **{pct(row(encoder, 'cross-video')['recall_at_50'])}%**."
        for encoder in ("clip_mean", "resnet50_mean", "bioclip_mean", "videomae")
    )
    out = [
        "# Additional model tests — 15 October 2026\n\n"
        "For fly lunges with five references from another recording, event IoU 0.3:\n\n"
        + summary
        + "\n",
        """

This extends the [original encoder pilot](encoder_comparison_results.md) with
**general CLIP (ViT-B/16, LAION-2B)** and **ResNet-50 (ImageNet-1K V2)**.
BioCLIP 2 (ViT-L/14) was attempted but did not complete under the current Mac's
resource pressure; it has no accuracy scores in this report.
The additional models are evaluated on all four existing Fly-vs-Fly excerpts,
not on the mouse recordings. The image models use both mean and ordered4 pooling.
No model is trained or fine-tuned; example counts are retrieval references.

There are two training and two test excerpts, each 120 seconds (eight minutes total).
Windows, sampled frames, full-frame letterboxing, labels, examples, exclusions,
NMS and scoring are unchanged from the [protocol](encoder_comparison_protocol.md).
All completed insect encoder/pooling configurations passed the paired audit, including
identical random controls. Each reference count uses 30 paired seed draws, seed 42.
Means are balanced by target recording, not pooled over all events.

General CLIP shares the original BioCLIP's ViT-B/16 architecture, making it a useful
general-image baseline. ResNet-50 adds a smaller supervised CNN; we use its 2048-D
pooled features before the classification head. Differences between architectures
and training data cannot be attributed solely to biological specialization. These models
are image encoders applied to video frames; ordered4 is coarse temporal pooling,
not a learned motion model. This is an exploratory extension on previously inspected
test excerpts, not fresh confirmation of the selected winner.

## Five-reference results

Nearest-example retrieval, positive test targets, temporal IoU at least 0.3.
All values are percentages. P10 is precision among ten proposals, R10/R50 are event
recall after ten/fifty proposals, F1@10 combines precision and recall at ten, and AP
uses the full ranked list. Results are event retrieval scores, not frame accuracy.
Both test excerpts contain lunges; cross-video has four source/target pairs and
within-video has two. The reserved reference pool is excluded within-video.
""",
    ]
    for protocol in ("within-video", "cross-video"):
        out.append(
            f"### {protocol.capitalize()}\n\n"
            "| Encoder | P10 | R10 | F1@10 | R50 | AP |\n"
            "|---|---:|---:|---:|---:|---:|"
        )
        for encoder, name in names.items():
            r = row(encoder, protocol)
            out.append(
                f"| {name} | "
                + " | ".join(
                    pct(r[k])
                    for k in (
                        "precision_at_10",
                        "recall_at_10",
                        "f1_at_10",
                        "recall_at_50",
                        "average_precision",
                    )
                )
                + " |"
            )
        control = row("videomae", protocol, method="random")
        out.append(
            "| Random control | "
            + " | ".join(
                pct(control[k])
                for k in (
                    "precision_at_10",
                    "recall_at_10",
                    "f1_at_10",
                    "recall_at_50",
                    "average_precision",
                )
            )
            + " |\n"
        )
    out.append(
        "## Variation between test excerpts\n\n"
        "Cross-video, five references, IoU 0.3. Mean pooling image models and VideoMAE. "
        "Each cell averages the two training sources and 30 seed draws.\n\n"
        "| Encoder | movie6 P10 | movie8 P10 |\n|---|---:|---:|"
    )
    for encoder, name in names.items():
        if "ordered4" in encoder:
            continue
        raw = json.loads((directory / f"flyvsfly_{encoder}.json").read_text())
        targets = defaultdict(list)
        for pair in raw["reports"]:
            if pair["protocol"] != "cross-video" or pair["behavior"] != "lunge":
                continue
            for summary in pair["summary"]:
                if (
                    summary["shots"] == 5
                    and summary["iou_threshold"] == 0.3
                    and summary["method"] == "nearest-example"
                ):
                    targets[pair["target"]].append(summary["precision_at_10"]["mean"])
        out.append(
            f"| {name} | "
            + " | ".join(
                pct(sum(targets[target]) / len(targets[target])) for target in sorted(targets)
            )
            + " |"
        )

    out.append("""\n## More examples and stricter boundary matching

Cross-video P10, nearest-example retrieval. Reference counts use the same test set
within each trial. The last column uses five references at stricter IoU 0.5.

| Encoder | 1 example | 3 examples | 5 examples | 10 examples | 5 examples, IoU 0.5 |
|---|---:|---:|---:|---:|---:|""")
    for encoder, name in names.items():
        values = [row(encoder, "cross-video", n)["precision_at_10"] for n in (1, 3, 5, 10)]
        values.append(row(encoder, "cross-video", threshold=0.5)["precision_at_10"])
        out.append(f"| {name} | " + " | ".join(map(pct, values)) + " |")
    out.append("""
## Interpretation

At five references, general CLIP has the highest cross-video P10 and AP among
completed models. ResNet-50 has the highest macro R10 and F1@10, but varies sharply
between the two excerpts. DINOv2 mean still leads within-video P10. F1 is averaged
per target after source/trial averaging; it is not computed from macro P and R.
There is no universal winner across metrics and protocols.

Ordered4 lowers five-reference P10 for every tested image model in both protocols.
More examples do not reliably help: CLIP mean cross-video P10 is 47.5%, 44.9%, 47.1%,
and 46.2% at 1/3/5/10 examples. These are retrieval reference counts, not fine-tuning
learning curves. Even the leading cross-video R50 is only 28.4%, so the tools still
miss most labeled events at that review budget.

## Limits and reproducibility

The test set has only two insect excerpts. Thirty seed draws measure example-choice
sensitivity; they are not thirty independent animals or sessions. Stronger top-ten
precision can coexist with low recall. All scores need human review. There is no
calibrated confidence/abstention threshold or validated automatic annotation accuracy.
Sparse wing-threat/tussle results and unsupported reference counts remain in the raw
reports; lunge is the only well-supported insect behavior in these excerpts.

No additional mouse results are implied. No-positive mouse targets and the original
mouse scores remain in the combined exports. Fresh full recordings and expert checks
of the scene-level fly labels are required before broader claims. The same preprocessing
uses bilinear letterboxing for all models instead of each checkpoint's default crop.

Pinned checkpoints for completed scores:

- [General CLIP](https://huggingface.co/laion/CLIP-ViT-B-16-laion2B-s34B-b88K), commit
  `7288da5a0d6f0b51c4a2b27c624837a9236d0112`.
- [ResNet-50](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html),
  `IMAGENET1K_V2`, official `resnet50-11ad3fa6.pth`; TorchVision checks the hash on download.

```sh
PYTHONPATH=src .venv/bin/python scripts/extend_encoder_comparison.py
```

For explicit GPU selection and small batches, append `--device mps --frame-batch-size 4`.

```sh
PYTHONPATH=src .venv/bin/python scripts/report_extended_encoders.py
```

The runner saves both pooling variants and resumes using checked caches. Use
`--datasets calms21` or `--datasets all` to run the longer mouse experiment separately.
Run only one GPU encoder at a time on the 16GB Mac. The main paired report generator
must finish before generating this document; it is called by the extension runner.

Inspect the [combined CSV](../outputs/benchmarks/encoder_comparison/macro_metrics.csv),
[paired audit](../outputs/benchmarks/encoder_comparison/paired_audit.json),
[learning curves](../outputs/benchmarks/encoder_comparison/encoder_learning_curves.png),
or [PDF figure](../outputs/benchmarks/encoder_comparison/encoder_learning_curves.pdf).
Per-recording timing, device, source hash, exact model revision, and batch size are
stored in each embedding cache's metadata. Raw trial reports and per-model manifests
remain in `outputs/benchmarks/encoder_comparison/`.

Completed new models used batches of 8 and float32 inference. BioCLIP 2's downloaded
checkpoint is pinned to `2957b322090f9cb17ae72c71981c7218a28d81e0`. Both its original
full-model loader (batch 8) and a visual-only memory-mapped loader (batch 4) were
attempted. Partial extraction remained too slow under current memory/GPU pressure;
both attempts were stopped and no incomplete features were scored. Process sampling
showed waiting for Metal completion; macOS also reported about 8GB of swap in use.
These observations do not isolate the cause or prove BioCLIP 2 cannot run on this
hardware under other conditions. This is a resource limitation, not an accuracy
result. No BioCLIP 2 worker remains running. To retry it explicitly, add
`--models bioclip2 --frame-batch-size 4` to the runner command. The visual-only loader
uses the native CLIP architecture and strict visual key matching; an equality test
checks its outputs against a complete native CLIP. Completed scores are preserved.
""")
    Path("docs/extended_encoder_results.md").write_text("\n".join(out))
    print("Saved docs/extended_encoder_results.md", flush=True)


if __name__ == "__main__":
    main()
