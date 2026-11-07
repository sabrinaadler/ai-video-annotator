# CalMS21 reliability pilot — 13 October 2026

The frozen VideoMAE retrieval system is not consistently reliable across the tested
behaviors. Attack ranking shows promise; mounting and investigation need substantial
improvement. This is an exploratory pilot, not a general accuracy guarantee.

## Experimental scope

Six recordings (approximately 43 minutes), four training/development sources and two
previously untested official-test recordings. Two train and two test recordings were
randomly selected before inspecting labels; mouse054 was retained from development.
Mouse070 was added through explicit source-label stratification to provide attack
examples. No target selection was changed. Configuration was fixed before test results.

1/3/5/10 references, 30 random seed trials, nearest-example and prototype methods,
random-ranking control, IoU 0.3/0.5, review budgets 10/20/50. Predictions are scored
against one-to-one matched event intervals. Model weights remain frozen.

The controlled within-video protocol reserves a fixed maximum-shot pool per trial,
keeping test truth/exclusions identical across shot counts. Across-video examples
come exclusively from training sources. These controlled results must not be mixed
with the earlier deployment-like mouse054 experiment, which changed test truth as
shot count increased. Different recordings and behavior durations also change results.

There are 19,440 metric configurations, **not that many independent videos**. Mount and attack each have only one positive test recording; investigation has two.
Macro means weight target recordings equally, after averaging source pairs/trials.
Plots and the following tables use positive target recordings. Negative recordings
are reported separately. All-source counts can change when shots are infeasible.

## Five-reference results

Nearest-example, IoU ≥ 0.3; precision, recall and F1 use the first ten proposals.
Recall@50 evaluates the deeper benchmark ranking; the current UI displays 20 proposals.

| Search | Behavior | Precision@10 | Recall@10 | F1@10 | Recall@50 | Random P@10 | Positive targets |
|---|---|---:|---:|---:|---:|---:|---:|
| within-video | attack | 67.0% | 22.3% | 33.5% | 61.9% | 8.7% | 1 |
| within-video | mount | 13.3% | 7.8% | 9.9% | 18.8% | 3.3% | 1 |
| within-video | investigation | 18.5% | 3.9% | 6.3% | 13.5% | 6.5% | 2 |
| cross-video | attack | 58.0% | 14.5% | 23.2% | 58.3% | 10.3% | 1 |
| cross-video | mount | 3.6% | 1.3% | 1.9% | 10.0% | 3.9% | 1 |
| cross-video | investigation | 13.7% | 2.4% | 4.0% | 10.4% | 8.3% | 2 |

Cross-video mounting is no better than random in this small positive-target pilot.
Attack retrieval beats random substantially, but misses many events at small review
budgets. Low F1@10 partly reflects the fixed ten-proposal budget: a video with 30 test
events cannot reach perfect recall by reviewing only ten proposals.

## Number of references

Same-video nearest-example P@10, IoU ≥ 0.3; test truth stays fixed within each trial.

| Behavior | 1 reference | 3 references | 5 references | 10 references |
|---|---:|---:|---:|---:|
| attack | 66.3% | 67.0% | 67.0% | 68.3% |
| mount | 7.7% | 15.0% | 13.3% | 11.0% |
| investigation | 17.3% | 18.0% | 18.5% | 16.2% |

More examples do not consistently improve this representation. Prototype P@10 with
five references is 70.7% for attack, 17.3% for mount, and 23.5% for investigation;
methods must be compared per behavior rather than choosing the best post hoc and
presenting it as a preselected test result. Trial standard deviations and per-pair
metrics are retained in the raw report. They are not unseen-animal confidence intervals.

## Boundary strictness and candidate geometry

Five-reference same-video nearest-example:

| Behavior | P@10 at IoU 0.3 | P@10 at IoU 0.5 | Candidate coverage at 0.3 | Candidate coverage at 0.5 |
|---|---:|---:|---:|---:|
| attack | 67.0% | 38.7% | 76.9% | 57.8% |
| mount | 13.3% | 7.3% | 48.0% | 34.1% |
| investigation | 18.5% | 7.2% | 51.7% | 30.1% |

Coverage is the fraction of test bouts with any eligible two-second window meeting
the IoU threshold, before NMS. It is a geometric upper bound, not model recall;
short/long events and excluded reference neighborhoods can make events unreachable.
Use window-length ablations to address this limitation before interpreting failures
as representation failures alone. Timing hints and accurate event boundaries should
remain distinct evaluation goals.

## Negative recordings

Mouse077 has no mounting; mouse085 has no attacks. Cross-video ranking still returns
proposals for these requested behaviors. At budgets 10/20/50 all returned proposals
are false positives. At ten proposals, precision and F1 are zero; recall is undefined.
The current system has no calibrated absence/abstention mechanism. Cosine similarity
near 1.0 is not confidence that a behavior occurred. User review remains essential.

## Accuracy and final annotations

Mount occupies 30.53 of 577.3 seconds in the original mouse054 recording. Always
predicting no mount yields 94.7% binary frame accuracy and zero event recall.
A single accuracy percentage is therefore a poor headline for this retrieval tool.
These scores evaluate raw suggestions, not accuracy of researcher-approved exports.
No annotation speed study or expert reliability audit has been completed.

## Adaptability and remaining validation

The encoder accepts RGB video windows; retrieval accepts vectors; evaluation accepts
seconds-based events. CalMS21 assumptions remain in its dataset adapter. A generic
CSV adapter now supports custom insect, rodent, or other behavior labels and overlapping
behaviors. This adapter has integration tests, **not an insect accuracy benchmark**.

Fly-vs-Fly is a concrete second-dataset candidate with expert insect behavior labels:
https://data.caltech.edu/records/zrznw-w7386. Its pixels and annotations have not yet
been downloaded/aligned/evaluated here. Next validation should include an independent
insect dataset, more official CalMS21 targets, multi-scale windows, and researcher
annotation-time measurements. Use development recordings for choices; keep new test
recordings held out until those choices are fixed. Audit animal/session independence
before claiming cross-animal generalization.

## Artifacts

- `outputs/benchmarks/calms21_pilot.json`: exact examples, provenance, counts, skipped settings,
  all trial metrics and representative top-ten matching cases.
- `outputs/benchmarks/calms21_pilot_report/macro_metrics.csv`: train/test and
  positive/negative/all-target summaries with sample counts.
- `outputs/benchmarks/calms21_pilot_report/learning_curves.png` and `.pdf`: positive-target curves.
- `docs/reliability_protocol.md`: scoring definitions, reproduction and validation requirements.

The original single-recording report remains preserved. New caches can be reused
for future experiments without running the encoder again.
