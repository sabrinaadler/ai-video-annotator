# Reliability evaluation protocol

A working annotation interface and passing software tests do not establish model
accuracy. Retrieval reliability must be measured against labels unavailable to the
retrieval process. User-approved exports measure reviewed annotations, not raw AI
performance. Keep both measurements distinct.

## What a video prediction means

One proposal is a half-open interval in seconds. It matches a ground-truth event
when temporal intersection / temporal union meets a threshold. At 0.3 a coarse
proposal may count; 0.5 requires more temporal agreement. Each event may match only
once; duplicate suggestions are false positives. These event-level scores differ
from frame-level classification metrics published for CalMS21 and must not be
compared directly. Our AP is non-interpolated ranked retrieval AP, not ActivityNet's
interpolated AP.

For the first K suggestions, TP is matched events, FP is unmatched suggestions,
FN is unmatched ground truth. Precision = TP/(TP+FP); recall = TP/(TP+FN);
F1 = 2TP/(2TP+FP+FN). A review budget is necessary: all three change as a researcher
reviews more of the ranking. Report budgets 10/20/50, AP over the complete ranking,
false suggestions per hit, and candidate coverage. The new budget metrics divide
precision by suggestions actually returned; legacy Precision@K uses K even if
fewer suggestions survive. Recall on no-positive targets is null; F1 is zero if false proposals are returned
and null if neither truth nor proposals exist. False positives remain measurable. Accuracy with true negatives is not defined for
an arbitrary list of event proposals. Frame accuracy can be dominated by background.

## Controlled learning curves

Use nested 1/3/5/10 reference examples, paired random seeds and methods, 30 trials.
The frozen encoder is not trained by these examples. Within a recording, reserve a
fixed pool of up to 10 distinct source events per trial; all pool events/windows
are excluded from every setting's candidates and truth. A 1-shot setting uses the
first pool example, 3-shot the first three, etc. Unused pool events are withheld.
This controlled setup holds the test set fixed across shot counts, unlike the
original deployment-like experiment that excluded only each setting's seeds.
These protocols answer different questions and their numbers should not be mixed.

For cross-video experiments, all examples come from the source recording and the
entire target recording is held out. Source and target caches must have compatible
representation metadata. No target labels reach similarity scoring. Include a
uniform random-ranking control with the same candidate set and NMS, paired across
shot counts. It measures ranking lift over random proposals, not random manual
watching. Audit missing shot counts instead of silently removing difficult cases.

## Current pilot

Dataset: CalMS21 Task 1. Retain previously observed mouse054 as development data.
Select two additional train and two test MP4s using random seed 20261013 from sorted
archive member lists, before inspecting behavior labels. Selected IDs are mouse059,
mouse025 (train), mouse085, mouse077 (test). This is not the full official test set.

Test mount, attack, investigation; nearest-example, mean prototype, random;
IoU 0.3/0.5; 1/3/5/10 references; 30 seed trials; budgets 10/20/50.
Use fixed VideoMAE commit dc740ceda42fce44faed2ea03c6d447db72f6af9,
letterbox-v1, two-second windows, half-second stride, 16 frames, temporal NMS 0.3.
Within-video pairs initially cover five recordings; cross-video pairs use train
sources against two fixed test targets. Add a reproducibly selected training source
with at least 11 attack bouts to provide attack examples (the random train sample
contains none). Record this source-label stratification separately; do not modify
the random test sample. Verify original source hashes, decoded frame
counts against labels, and representation compatibility. Keep exact examples,
cache provenance, counts, skipped cases, per-trial values and recording IDs.

Trial standard deviations quantify sensitivity to example selection within a
recording; they are not confidence intervals for unseen animals or laboratories.
Aggregate by target recording, then average targets equally, keeping behavior,
protocol and IoU separate. Keep official train and test targets separate. Do not pool thousands of seed
trials as independent videos. Publish per-recording distributions as well as macro means.

## Requirements before a broad reliability claim

- Expand to all feasible official CalMS21 test recordings, with configuration frozen
  on development recordings and actual animal/session grouping audited.
- Add an independent pixel-video + event-label dataset. Fly-vs-Fly is a concrete
  candidate with expert labels and a different species/visual scale. An adapter
  must verify annotation vocabulary, timestamps, available videos, and splits;
  do not substitute pose-only feature files for video. It has not been evaluated.
- Pre-register window-length comparisons (0.5/1/2/4 seconds) on development data.
  Short bouts may be mathematically undetectable at strict IoU with two-second
  windows; coverage separates this limit from ranking failure.
- Compare frozen representations and preprocessing using paired trials. Preserve
  held-out targets until model/window choices are finalized.
- Report target-recording/animal-cluster bootstrap 95% confidence intervals once
  the independent sample size supports them, failure cases, rare behaviors,
  no-positive recordings, lighting/camera differences and boundary error.
- Measure human utility: randomized/counterbalanced manual-vs-assisted annotation
  on matched clips; researcher minutes per hour, review actions per true event,
  accepted/rejected rates, missed-event audit, and final annotation agreement.
  Include example-labeling and boundary-edit time. Reliable AI rankings alone do
  not establish time saved or final annotation completeness.

Sources: CalMS21 authors' paper https://pmc.ncbi.nlm.nih.gov/articles/PMC11067713/;
ActivityNet proposal evaluation https://activity-net.org/challenges/2018/tasks/anet_proposals.html;
Fly-vs-Fly original dataset https://data.caltech.edu/records/zrznw-w7386.

## Reproduce

`scripts/benchmark.py` accepts a dataset-independent JSON manifest with
`recordings` ({id, video, cache, events:[{behavior,start,end}]}), `behaviors`, and
`pairs` ({source,target,protocol}). Paths are interpreted relative to the working
directory. IDs must be unique, and cache video IDs must identify distinct recordings.

```sh
PYTHONPATH=src .venv/bin/python scripts/benchmark.py \
  --manifest outputs/benchmarks/calms21_pilot_manifest.json \
  --trials 30 --shots 1 3 5 10 --seed 42 \
  --output outputs/benchmarks/calms21_pilot.json
```

The explicit output path must be new. Do not overwrite prior reports. Labels are
converted by the dataset adapter beforehand; the core evaluator has no CalMS21
assumptions. Only source examples are supplied to the scorer. This pilot is broader
than the initial single-recording experiment, but not a comprehensive multi-dataset
validation or comparison against published classifiers.

To prepare the exact pilot from the existing archives (and encode missing caches):

```sh
PYTHONPATH=src .venv/bin/python scripts/prepare_calms21_benchmark.py --encode --device mps
```

Then create a recording-balanced CSV/JSON and PNG/PDF learning curves:

```sh
PYTHONPATH=src .venv/bin/python scripts/report_benchmark.py \
  --report outputs/benchmarks/calms21_pilot.json \
  --output outputs/benchmarks/calms21_pilot_report
```

Plots show only official-test target recordings; CSV also includes development
recordings separately. Each setting reports target/source counts and defined-metric
counts. Missing examples can change which sources/targets contribute across shot
counts, so use the per-pair paired curves for controlled conclusions. Representative
trial-zero five-shot top-ten cases include matched truth and best event IoU for
manual audit. Benchmark suppression reuses a vectorized temporal overlap matrix;
regression tests verify identical ranking to the core temporal NMS, including ties
and threshold extremes.

## Custom species / insect data

Each recording can provide `events_csv` instead of inline `events`:

```json
{
  "recordings": [
    {"id": "fly-session-01", "video": "data/fly01.mp4",
     "cache": "outputs/embeddings/fly01", "events_csv": "data/insect_ground_truth.csv"}
  ],
  "behaviors": ["wing extension", "circling"],
  "pairs": [{"source": "fly-session-01", "target": "fly-session-01", "protocol": "within-video"}]
}
```

CSV columns: `video_id,behavior,start_seconds,end_seconds`. IDs must match the cache;
behavior names are arbitrary, and overlapping behavior labels are supported.
Additional columns are ignored. Duplicate labels and invalid/nonfinite intervals
are rejected. Convert frame-based labels with the recording's actual FPS and verify
alignment; do not assume insect videos use CalMS21 frame rates. Use independently
reviewed ground truth, never raw system suggestions as their own evaluation labels.

The CSV adapter does not assert that the frozen encoder transfers successfully to
insects. It provides a shared evaluation interface; an actual insect pixel-video
benchmark and domain-specific annotation audit are still required.
