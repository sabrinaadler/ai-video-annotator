# Encoder comparison results — 15 October 2026

**BioCLIP helps in this insect pilot, but does not replace VideoMAE across behaviors.**
For cross-video fly lunges with five references, BioCLIP mean achieves 41.5% precision
among the first ten proposals, versus 33.5% for VideoMAE and 29.0% for DINOv2 mean.
For cross-video mouse attacks, VideoMAE leads at 58.0%, versus BioCLIP mean 47.0%
and DINOv2 mean 17.7%. Within-video fly lunges favor DINOv2 mean (39.2%).
Ordered pooling is not a consistent improvement. More references do not reliably
improve performance; these references are examples for retrieval, not training data
for fitting the encoder.

This is a completed **small pilot**, not a validated species-general model or a
single application-wide accuracy estimate. Keep the encoder selectable for research;
retain VideoMAE as the existing default while testing alternatives on fresh recordings.
Do not automatically choose an encoder using these test results and then claim an
unbiased final performance estimate on the same recordings.

## What was measured

All encoders are frozen: original BioCLIP ViT-B/16, DINOv2 Small, and VideoMAE Base.
The two image encoders each have mean and four-bin ordered pooling variants.
Six CalMS21 recordings use 2-second windows at 0.5-second stride. Four fixed
120-second Fly-vs-Fly excerpts (eight minutes total) use 0.25-second windows at
0.125-second stride. Two recordings/excerpts per dataset are official test targets.
Mouse attack and mount each occur in only ONE positive test recording; investigation
and fly lunge have TWO. See the [protocol](encoder_comparison_protocol.md) for source
selection, pinned models, sampling, label conversion and exclusions.

Trials use nested 1/3/5/10 examples, 30 seed draws, nearest-example or mean-prototype
retrieval, and a paired random control. Within-video means examples and evaluation
come from one recording with the reserved example pool withheld. Cross-video means
examples come from official training recordings and evaluation from test recordings.
This does not establish generalization across laboratories, species, or recording setups.

The audit passed for all ten dataset/encoder runs: identical selected examples,
reserved pools, eligible source counts, evaluation truth, candidate coverage and
random controls. Cache geometry/source checks passed before scoring.

## How to read video metrics

A proposal is a true positive if its time interval overlaps an unmatched labeled
behavior event at the required temporal intersection-over-union (tIoU). tIoU is
intersection duration divided by union duration. Duplicate detections of the same
event count as false positives. tIoU 0.5 requires tighter boundaries than 0.3.

- **P10:** fraction of the first ten proposals that match events; review efficiency.
- **R10/R50:** fraction of eligible labeled events found after reviewing 10/50 proposals.
- **F1@10:** harmonic mean of precision and recall at the ten-proposal budget.
- **AP:** precision accumulated at true-positive ranks over the full ranked proposal list,
  divided by the number of labeled events; not classification accuracy.
- **Coverage:** fraction of events with ANY eligible cached window capable of meeting tIoU,
  before ranking/NMS. This is a geometric upper bound, not achieved recall.

All table metric values below are percentages. Metrics are averaged over trials and
source recordings within each target, then equally over targets. Consequently the
reported macro F1 need not equal the harmonic mean of the reported macro P and R.
The ten-proposal budget can have low recall simply because there are many events;
R50 shows whether reviewing more helps. Frame accuracy is omitted because dominant
background frames can conceal poor event detection.

## Five-reference results at tIoU 0.3

Positive test targets only; nearest-example retrieval. These tables include every
encoder variant, not only the strongest configuration.

### Within-video

| Dataset / behavior | Encoder | P10 | R10 | F1@10 | R50 | AP | Positive targets |
|---|---|---:|---:|---:|---:|---:|---:|
| CalMS21 / attack | VideoMAE | 67.0 | 22.3 | 33.5 | 61.9 | 44.5 | 1 |
| CalMS21 / attack | DINOv2 mean | 67.3 | 22.4 | 33.7 | 63.1 | 43.5 | 1 |
| CalMS21 / attack | DINOv2 ordered4 | 69.3 | 23.1 | 34.7 | 59.0 | 41.5 | 1 |
| CalMS21 / attack | BioCLIP mean | 64.7 | 21.6 | 32.3 | 58.3 | 38.5 | 1 |
| CalMS21 / attack | BioCLIP ordered4 | 51.0 | 17.0 | 25.5 | 49.7 | 31.5 | 1 |
| CalMS21 / mount | VideoMAE | 13.3 | 7.8 | 9.9 | 18.8 | 6.5 | 1 |
| CalMS21 / mount | DINOv2 mean | 8.0 | 4.7 | 5.9 | 20.8 | 4.8 | 1 |
| CalMS21 / mount | DINOv2 ordered4 | 6.0 | 3.5 | 4.4 | 17.5 | 4.3 | 1 |
| CalMS21 / mount | BioCLIP mean | 14.3 | 8.4 | 10.6 | 22.0 | 6.5 | 1 |
| CalMS21 / mount | BioCLIP ordered4 | 12.0 | 7.1 | 8.9 | 18.6 | 5.2 | 1 |
| CalMS21 / investigation | VideoMAE | 18.5 | 3.9 | 6.3 | 13.5 | 7.0 | 2 |
| CalMS21 / investigation | DINOv2 mean | 20.3 | 4.9 | 7.8 | 17.9 | 9.1 | 2 |
| CalMS21 / investigation | DINOv2 ordered4 | 16.3 | 3.5 | 5.7 | 14.0 | 6.6 | 2 |
| CalMS21 / investigation | BioCLIP mean | 19.2 | 4.6 | 7.3 | 15.5 | 8.4 | 2 |
| CalMS21 / investigation | BioCLIP ordered4 | 20.3 | 4.7 | 7.5 | 13.2 | 7.2 | 2 |
| Fly-vs-Fly / lunge | VideoMAE | 27.7 | 4.5 | 7.4 | 15.6 | 17.4 | 2 |
| Fly-vs-Fly / lunge | DINOv2 mean | 39.2 | 6.5 | 10.7 | 26.1 | 24.5 | 2 |
| Fly-vs-Fly / lunge | DINOv2 ordered4 | 23.3 | 3.9 | 6.4 | 17.8 | 17.8 | 2 |
| Fly-vs-Fly / lunge | BioCLIP mean | 34.2 | 5.6 | 9.3 | 25.1 | 24.4 | 2 |
| Fly-vs-Fly / lunge | BioCLIP ordered4 | 27.3 | 3.9 | 6.5 | 20.6 | 21.2 | 2 |

### Cross-video

| Dataset / behavior | Encoder | P10 | R10 | F1@10 | R50 | AP | Positive targets |
|---|---|---:|---:|---:|---:|---:|---:|
| CalMS21 / attack | VideoMAE | 58.0 | 14.5 | 23.2 | 58.3 | 43.3 | 1 |
| CalMS21 / attack | DINOv2 mean | 17.7 | 4.4 | 7.1 | 23.1 | 13.4 | 1 |
| CalMS21 / attack | DINOv2 ordered4 | 16.3 | 4.1 | 6.5 | 20.7 | 12.3 | 1 |
| CalMS21 / attack | BioCLIP mean | 47.0 | 11.8 | 18.8 | 36.8 | 25.3 | 1 |
| CalMS21 / attack | BioCLIP ordered4 | 28.7 | 7.2 | 11.5 | 28.3 | 16.6 | 1 |
| CalMS21 / mount | VideoMAE | 3.6 | 1.3 | 1.9 | 10.0 | 2.9 | 1 |
| CalMS21 / mount | DINOv2 mean | 7.1 | 2.6 | 3.8 | 13.2 | 3.8 | 1 |
| CalMS21 / mount | DINOv2 ordered4 | 8.3 | 3.1 | 4.5 | 15.6 | 4.3 | 1 |
| CalMS21 / mount | BioCLIP mean | 6.4 | 2.4 | 3.5 | 7.9 | 3.5 | 1 |
| CalMS21 / mount | BioCLIP ordered4 | 9.7 | 3.6 | 5.2 | 13.0 | 4.9 | 1 |
| CalMS21 / investigation | VideoMAE | 13.7 | 2.4 | 4.0 | 10.4 | 6.8 | 2 |
| CalMS21 / investigation | DINOv2 mean | 15.0 | 3.0 | 5.0 | 14.3 | 8.3 | 2 |
| CalMS21 / investigation | DINOv2 ordered4 | 10.8 | 2.2 | 3.6 | 10.7 | 6.1 | 2 |
| CalMS21 / investigation | BioCLIP mean | 10.3 | 2.0 | 3.3 | 11.0 | 6.7 | 2 |
| CalMS21 / investigation | BioCLIP ordered4 | 9.6 | 1.7 | 2.9 | 9.8 | 5.6 | 2 |
| Fly-vs-Fly / lunge | VideoMAE | 33.5 | 4.4 | 7.7 | 16.5 | 18.5 | 2 |
| Fly-vs-Fly / lunge | DINOv2 mean | 29.0 | 4.1 | 7.0 | 19.0 | 18.8 | 2 |
| Fly-vs-Fly / lunge | DINOv2 ordered4 | 18.0 | 2.6 | 4.5 | 13.7 | 14.8 | 2 |
| Fly-vs-Fly / lunge | BioCLIP mean | 41.5 | 6.3 | 10.8 | 22.7 | 25.3 | 2 |
| Fly-vs-Fly / lunge | BioCLIP ordered4 | 33.3 | 4.9 | 8.3 | 17.1 | 19.6 | 2 |

## Stricter boundaries and the random control

Five examples, cross-video, positive test targets. P10 is shown at both thresholds;
the random control uses identical candidate windows and NMS, with randomized ranking.

| Task | Encoder | P10 at 0.3 | P10 at 0.5 | Coverage at 0.3 | Coverage at 0.5 |
|---|---|---:|---:|---:|---:|
| CalMS21 / attack | VideoMAE | 58.0 | 32.7 | 77.5 | 60.0 |
| CalMS21 / attack | DINOv2 mean | 17.7 | 8.3 | 77.5 | 60.0 |
| CalMS21 / attack | DINOv2 ordered4 | 16.3 | 7.3 | 77.5 | 60.0 |
| CalMS21 / attack | BioCLIP mean | 47.0 | 30.0 | 77.5 | 60.0 |
| CalMS21 / attack | BioCLIP ordered4 | 28.7 | 12.0 | 77.5 | 60.0 |
| CalMS21 / attack | Random | 10.3 | 5.7 | 77.5 | 60.0 |
| CalMS21 / mount | VideoMAE | 3.6 | 0.2 | 48.1 | 33.3 |
| CalMS21 / mount | DINOv2 mean | 7.1 | 3.9 | 48.1 | 33.3 |
| CalMS21 / mount | DINOv2 ordered4 | 8.3 | 3.3 | 48.1 | 33.3 |
| CalMS21 / mount | BioCLIP mean | 6.4 | 2.6 | 48.1 | 33.3 |
| CalMS21 / mount | BioCLIP ordered4 | 9.7 | 3.1 | 48.1 | 33.3 |
| CalMS21 / mount | Random | 3.9 | 1.7 | 48.1 | 33.3 |
| CalMS21 / investigation | VideoMAE | 13.7 | 6.7 | 51.6 | 30.3 |
| CalMS21 / investigation | DINOv2 mean | 15.0 | 5.7 | 51.6 | 30.3 |
| CalMS21 / investigation | DINOv2 ordered4 | 10.8 | 4.0 | 51.6 | 30.3 |
| CalMS21 / investigation | BioCLIP mean | 10.3 | 1.7 | 51.6 | 30.3 |
| CalMS21 / investigation | BioCLIP ordered4 | 9.6 | 3.6 | 51.6 | 30.3 |
| CalMS21 / investigation | Random | 8.3 | 3.3 | 51.6 | 30.3 |
| Fly-vs-Fly / lunge | VideoMAE | 33.5 | 21.3 | 100.0 | 94.7 |
| Fly-vs-Fly / lunge | DINOv2 mean | 29.0 | 17.8 | 100.0 | 94.7 |
| Fly-vs-Fly / lunge | DINOv2 ordered4 | 18.0 | 13.2 | 100.0 | 94.7 |
| Fly-vs-Fly / lunge | BioCLIP mean | 41.5 | 25.5 | 100.0 | 94.7 |
| Fly-vs-Fly / lunge | BioCLIP ordered4 | 33.3 | 22.1 | 100.0 | 94.7 |
| Fly-vs-Fly / lunge | Random | 16.8 | 9.2 | 100.0 | 94.7 |

## What happens with more examples

Cross-video P10 at tIoU 0.3, positive test targets, nearest-example retrieval.
All reference counts use a fixed test set within each trial.

| Task | Encoder | 1 example | 3 examples | 5 examples | 10 examples |
|---|---|---:|---:|---:|---:|
| CalMS21 / attack | VideoMAE | 57.7 | 57.0 | 58.0 | 52.0 |
| CalMS21 / attack | DINOv2 mean | 20.0 | 22.3 | 17.7 | 12.7 |
| CalMS21 / attack | DINOv2 ordered4 | 21.7 | 20.0 | 16.3 | 12.0 |
| CalMS21 / attack | BioCLIP mean | 45.3 | 48.7 | 47.0 | 50.3 |
| CalMS21 / attack | BioCLIP ordered4 | 30.0 | 31.0 | 28.7 | 37.3 |
| CalMS21 / mount | VideoMAE | 4.0 | 3.2 | 3.6 | 0.0 |
| CalMS21 / mount | DINOv2 mean | 6.2 | 7.7 | 7.1 | 11.8 |
| CalMS21 / mount | DINOv2 ordered4 | 8.6 | 10.6 | 8.3 | 13.2 |
| CalMS21 / mount | BioCLIP mean | 4.8 | 5.6 | 6.4 | 9.3 |
| CalMS21 / mount | BioCLIP ordered4 | 6.6 | 7.8 | 9.7 | 15.0 |
| CalMS21 / investigation | VideoMAE | 12.0 | 13.6 | 13.7 | 15.2 |
| CalMS21 / investigation | DINOv2 mean | 15.8 | 14.6 | 15.0 | 13.4 |
| CalMS21 / investigation | DINOv2 ordered4 | 11.9 | 10.6 | 10.8 | 9.4 |
| CalMS21 / investigation | BioCLIP mean | 11.9 | 11.1 | 10.3 | 10.1 |
| CalMS21 / investigation | BioCLIP ordered4 | 9.7 | 9.6 | 9.6 | 11.1 |
| Fly-vs-Fly / lunge | VideoMAE | 29.0 | 33.3 | 33.5 | 33.8 |
| Fly-vs-Fly / lunge | DINOv2 mean | 33.4 | 28.9 | 29.0 | 30.2 |
| Fly-vs-Fly / lunge | DINOv2 ordered4 | 23.9 | 18.1 | 18.0 | 18.2 |
| Fly-vs-Fly / lunge | BioCLIP mean | 41.7 | 41.6 | 41.5 | 41.7 |
| Fly-vs-Fly / lunge | BioCLIP ordered4 | 33.2 | 31.6 | 33.3 | 35.2 |

The [learning-curve figure](../outputs/benchmarks/encoder_comparison/encoder_learning_curves.png)
includes both within-video and cross-video curves. Complete machine-readable
[CSV](../outputs/benchmarks/encoder_comparison/macro_metrics.csv) and
[JSON](../outputs/benchmarks/encoder_comparison/macro_metrics.json) include prototype
retrieval, both thresholds, all supported reference counts, and target-status groups.

## Limits and next decisions

The weakest mouse results remain weak with all tested encoders, especially mount
across videos. Fixed 2-second windows also fail to geometrically cover about half
of mount/investigation events at tIoU 0.3. A better ranking model alone cannot fix
those missing boundaries; duration-aware proposals and boundary refinement deserve
separate controlled experiments.

Fly lunges have almost complete geometric coverage at tIoU 0.3, yet achieved recall
remains modest. This points to ranking/representation as an additional bottleneck.
Do not infer the same performance for other insects or other behaviors. Wing-threat
and tussle are retained in raw reports, but sparse source examples make many 5/10
reference configurations unavailable; they are not suitable for the main curves.

No-positive target rows are included separately in the exported metrics. A ranker
forced to return ten proposals will return false positives on such a target. It has
no calibrated abstention threshold, so these results do not establish specificity,
false alarms per hour, or calibrated confidence. A development-set threshold and
independent negative recordings are needed for those claims.

Thirty seed draws measure sensitivity to example choice, not population uncertainty.
The curves are paired descriptively; the pilot is too small for a convincing general
winner or species-wide confidence interval. Mouse test recordings were previously
inspected. The insect excerpts are scene-level annotations converted from actor
bouts; expert review and full-recording evaluation remain necessary. No fine-tuning,
BioCLIP 2/2.5, optical flow, pose baseline, or alternate video model was tested here.

Next validation should lock the proposal policy and model selection using development
recordings, then evaluate fresh full recordings from multiple animals/sessions and
labs, including genuine no-event videos. Report performance by behavior and dataset,
annotation agreement, missed-event duration, false alarms per hour, and review time.

## Reproduce and inspect

With the caches prepared as described in the protocol:

```sh
PYTHONPATH=src .venv/bin/python scripts/compare_encoders.py
PYTHONPATH=src .venv/bin/python scripts/report_encoder_comparison.py
```

The comparison runner reuses completed reports only when their manifests and seed
trial settings match. The report command audits pairing before writing aggregates.
[Audit output](../outputs/benchmarks/encoder_comparison/paired_audit.json), raw reports,
and manifests are saved in `outputs/benchmarks/encoder_comparison/`; exact model
revisions/environment are in `outputs/model-cache/comparison_*.json`. These are local
artifacts, not a published benchmark. Scientific figure is also available as
[PDF](../outputs/benchmarks/encoder_comparison/encoder_learning_curves.pdf).
