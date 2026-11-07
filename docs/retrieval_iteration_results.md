# Retrieval and workflow iteration — 18 October 2026

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

### Earlier fly excerpts — two positive test targets

| Behavior | Condition | P10 | R10 | F1@10 | R50 | AP | Coverage |
|---|---|---:|---:|---:|---:|---:|---:|
| lunge | Fixed / nearest example | 47.1 | 6.4 | 11.1 | 28.4 | 28.7 | 100.0 |
| lunge | Fixed / counterexamples | 33.8 | 4.9 | 8.4 | 21.5 | 23.4 | 100.0 |
| lunge | Multiple durations / nearest example | 18.1 | 2.3 | 4.0 | 12.3 | 16.0 | 100.0 |
| lunge | Multiple durations / counterexamples | 21.1 | 2.9 | 5.0 | 12.3 | 15.5 | 100.0 |

### Fresh full fly recording — one test target

| Behavior | Condition | P10 | R10 | F1@10 | R50 | AP | Coverage |
|---|---|---:|---:|---:|---:|---:|---:|
| lunge | Fixed / nearest example | 25.5 | 0.7 | 1.4 | 3.6 | 12.2 | 100.0 |
| lunge | Fixed / counterexamples | 24.0 | 0.7 | 1.4 | 3.0 | 10.3 | 100.0 |
| lunge | Multiple durations / nearest example | 13.5 | 0.4 | 0.8 | 1.7 | 6.9 | 100.0 |
| lunge | Multiple durations / counterexamples | 13.8 | 0.4 | 0.8 | 2.0 | 6.8 | 100.0 |

### Mouse duration mixing — one previously inspected test target

| Behavior | Condition | P10 | R10 | F1@10 | R50 | AP | Coverage |
|---|---|---:|---:|---:|---:|---:|---:|
| mount | Fixed / nearest example | 8.0 | 3.0 | 4.3 | 7.7 | 2.6 | 48.1 |
| mount | Fixed / counterexamples | 9.7 | 3.6 | 5.2 | 9.4 | 4.2 | 48.1 |
| mount | Multiple durations / nearest example | 3.7 | 1.4 | 2.0 | 5.3 | 1.5 | 74.1 |
| mount | Multiple durations / counterexamples | 3.0 | 1.1 | 1.6 | 4.7 | 1.3 | 74.1 |
| investigation | Fixed / nearest example | 16.3 | 3.5 | 5.7 | 17.5 | 9.4 | 68.1 |
| investigation | Fixed / counterexamples | 12.3 | 2.6 | 4.3 | 14.6 | 8.0 | 68.1 |
| investigation | Multiple durations / nearest example | 15.7 | 3.3 | 5.5 | 10.4 | 6.5 | 100.0 |
| investigation | Multiple durations / counterexamples | 11.0 | 2.3 | 3.9 | 7.5 | 4.7 | 100.0 |

### Mouse example-based duration — same exploratory target

| Behavior | Condition | P10 | R10 | F1@10 | R50 | AP | Coverage |
|---|---|---:|---:|---:|---:|---:|---:|
| mount | Fixed / nearest example | 8.0 | 3.0 | 4.3 | 7.7 | 2.6 | 48.1 |
| mount | Fixed / counterexamples | 9.7 | 3.6 | 5.2 | 9.4 | 4.2 | 48.1 |
| mount | Example-selected duration / nearest example | 10.0 | 3.7 | 5.4 | 24.6 | 9.0 | 72.3 |
| mount | Example-selected duration / counterexamples | 26.0 | 9.6 | 14.1 | 26.2 | 15.2 | 72.3 |
| investigation | Fixed / nearest example | 16.3 | 3.5 | 5.7 | 17.5 | 9.4 | 68.1 |
| investigation | Fixed / counterexamples | 12.3 | 2.6 | 4.3 | 14.6 | 8.0 | 68.1 |
| investigation | Example-selected duration / nearest example | 22.7 | 4.8 | 8.0 | 16.1 | 8.9 | 73.8 |
| investigation | Example-selected duration / counterexamples | 21.7 | 4.6 | 7.6 | 15.6 | 8.8 | 73.8 |

The mouse duration-selected counterexample condition improves mount precision@10
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
across all behaviors. The ethogram and behavior-specific review contexts persist in JSON. Negative/positive conflicts
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
.venv/bin/python scripts/benchmark_retrieval_iteration.py \
  --manifest outputs/benchmarks/retrieval_iteration_manifest.json \
  --output outputs/benchmarks/new_pilot_report.json
.venv/bin/python scripts/report_retrieval_iteration.py
```

The report generator requires all recorded studies above. Raw output paths are never
silently overwritten. The fresh test preparation uses `prepare_fresh_fly_validation.py`,
then the frame-bank preparer with its manifest. For mice, use baseline duration 2,
durations .25/.5/1/2/4 and stride fraction .25. `--adaptive-duration` selects example-based
duration; `--development` requires train-to-train pairs. Alternative scoring is exposed
as `--negative-method prototype-difference --negative-weight 1` for research, not the default.
