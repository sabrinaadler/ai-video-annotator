# Additional model tests — 15 October 2026

For fly lunges with five references from another recording, event IoU 0.3:

- General CLIP mean: P10 **47.1%**, R50 **28.4%**.
- ResNet-50 mean: P10 **41.8%**, R50 **27.2%**.
- BioCLIP mean: P10 **41.5%**, R50 **22.7%**.
- VideoMAE Base: P10 **33.5%**, R50 **16.5%**.



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

### Within-video

| Encoder | P10 | R10 | F1@10 | R50 | AP |
|---|---:|---:|---:|---:|---:|
| VideoMAE Base | 27.7 | 4.5 | 7.4 | 15.6 | 17.4 |
| DINOv2 Small mean | 39.2 | 6.5 | 10.7 | 26.1 | 24.5 |
| DINOv2 Small ordered4 | 23.3 | 3.9 | 6.4 | 17.8 | 17.8 |
| BioCLIP mean | 34.2 | 5.6 | 9.3 | 25.1 | 24.4 |
| BioCLIP ordered4 | 27.3 | 3.9 | 6.5 | 20.6 | 21.2 |
| General CLIP mean | 38.0 | 6.1 | 10.1 | 27.8 | 24.7 |
| General CLIP ordered4 | 29.2 | 4.4 | 7.4 | 17.5 | 18.4 |
| ResNet-50 mean | 34.2 | 5.6 | 9.4 | 25.1 | 23.4 |
| ResNet-50 ordered4 | 27.8 | 5.0 | 8.2 | 22.5 | 22.1 |
| Random control | 15.8 | 2.0 | 3.5 | 9.4 | 11.0 |

### Cross-video

| Encoder | P10 | R10 | F1@10 | R50 | AP |
|---|---:|---:|---:|---:|---:|
| VideoMAE Base | 33.5 | 4.4 | 7.7 | 16.5 | 18.5 |
| DINOv2 Small mean | 29.0 | 4.1 | 7.0 | 19.0 | 18.8 |
| DINOv2 Small ordered4 | 18.0 | 2.6 | 4.5 | 13.7 | 14.8 |
| BioCLIP mean | 41.5 | 6.3 | 10.8 | 22.7 | 25.3 |
| BioCLIP ordered4 | 33.3 | 4.9 | 8.3 | 17.1 | 19.6 |
| General CLIP mean | 47.1 | 6.4 | 11.1 | 28.4 | 28.7 |
| General CLIP ordered4 | 40.3 | 5.2 | 9.0 | 21.0 | 22.9 |
| ResNet-50 mean | 41.8 | 7.1 | 11.9 | 27.2 | 28.1 |
| ResNet-50 ordered4 | 35.9 | 6.3 | 10.6 | 24.3 | 25.5 |
| Random control | 16.8 | 1.9 | 3.3 | 9.2 | 12.1 |

## Variation between test excerpts

Cross-video, five references, IoU 0.3. Mean pooling image models and VideoMAE. Each cell averages the two training sources and 30 seed draws.

| Encoder | movie6 P10 | movie8 P10 |
|---|---:|---:|
| VideoMAE Base | 38.8 | 28.2 |
| DINOv2 Small mean | 30.3 | 27.7 |
| BioCLIP mean | 35.3 | 47.7 |
| General CLIP mean | 51.5 | 42.7 |
| ResNet-50 mean | 24.8 | 58.7 |

## More examples and stricter boundary matching

Cross-video P10, nearest-example retrieval. Reference counts use the same test set
within each trial. The last column uses five references at stricter IoU 0.5.

| Encoder | 1 example | 3 examples | 5 examples | 10 examples | 5 examples, IoU 0.5 |
|---|---:|---:|---:|---:|---:|
| VideoMAE Base | 29.0 | 33.3 | 33.5 | 33.8 | 21.3 |
| DINOv2 Small mean | 33.4 | 28.9 | 29.0 | 30.2 | 17.8 |
| DINOv2 Small ordered4 | 23.9 | 18.1 | 18.0 | 18.2 | 13.2 |
| BioCLIP mean | 41.7 | 41.6 | 41.5 | 41.7 | 25.5 |
| BioCLIP ordered4 | 33.2 | 31.6 | 33.3 | 35.2 | 22.1 |
| General CLIP mean | 47.5 | 44.9 | 47.1 | 46.2 | 28.7 |
| General CLIP ordered4 | 39.2 | 40.4 | 40.3 | 39.6 | 24.7 |
| ResNet-50 mean | 41.4 | 41.2 | 41.8 | 42.3 | 23.8 |
| ResNet-50 ordered4 | 36.4 | 37.4 | 35.9 | 34.5 | 21.1 |

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
