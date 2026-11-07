# First real CalMS21 experiment

## Recording and configuration

- Training recording: `task1/train/mouse054_task1_annotator1`.
- 577.3 seconds, 30 fps, 17,319 decoded frames matching annotations exactly.
- 20 contiguous mounting events; 20 distinct seed windows.
- Frozen `MCG-NJU/videomae-base`, revision
  `dc740ceda42fce44faed2ea03c6d447db72f6af9`.
- Transformers 4.57.6; float32 inference on Apple MPS.
- Standard checkpoint resize and center crop; mean of all final tokens.
- 2-second windows, 0.5-second stride, 16 uniformly sampled frames.
- 1,152 cached 768-dimensional vectors. Verified cache reuse skipped model loading.
- 1/3/5/10 examples, five trials each, seed 42; both retrieval methods.
- NMS IoU 0.3; evaluation IoU 0.3 and 0.5, one-to-one matches.
- Ground-truth seeds excluded from evaluation; all other events remain in recall denominator.

## Results

Means over five random seed selections; full per-trial metrics and standard deviations
are in `outputs/metrics/mouse054_mount.json`.

| Examples | Method | P@10, IoU 0.3 | AP, IoU 0.3 | P@10, IoU 0.5 | AP, IoU 0.5 |
|---|---|---:|---:|---:|---:|
| 1 | Nearest example | .24 | .135 | .22 | .083 |
| 1 | Prototype | .24 | .135 | .22 | .083 |
| 3 | Nearest example | .32 | .179 | .28 | .104 |
| 3 | Prototype | .26 | .177 | .20 | .103 |
| 5 | Nearest example | .36 | .254 | .26 | .151 |
| 5 | Prototype | .40 | .276 | .18 | .101 |
| 10 | Nearest example | .34 | .305 | .26 | .207 |
| 10 | Prototype | .22 | .177 | .20 | .128 |

At five examples and IoU 0.3, P@5 is .48 for nearest example and .56 for prototype;
Recall@50 is .493 and .440 respectively. P@10 population standard deviations are
.120 and .141, illustrating variation with seed selection. Larger shot counts also
leave fewer unseen events, so AP comparisons across shot counts change the truth set.

## Interpretation and limitations

This establishes an executable end-to-end baseline on real pixels and annotations.
It does not establish practical usefulness, generalization, or reduced annotation time.
No random-ranking baseline or held-out recording comparison has been run yet.

Only 16/20 events have any candidate at IoU 0.3, and 12/20 at IoU 0.5, before seed
exclusion and NMS. Very short bouts cannot achieve the thresholds with fixed
2-second windows. Candidate coverage is recorded in
`outputs/mouse054_window_coverage.json`.

Frame spot checks (`outputs/mouse054_mount_check.jpg`) show animals near image edges.
The model's center crop can remove these regions. Full-frame preprocessing should be
checked before drawing conclusions about video representation quality. Exact frame
counts verify basic alignment; behavior boundaries still merit expert clip review.

Cosine scores for the trial-zero five-shot nearest-example report are tightly
clustered near 1.0. These are ranking similarities, not calibrated probabilities.
High numerical similarity is insufficient evidence of a correct event. NMS also
allows some residual duplicate windows around long bouts; one-to-one matching
penalizes these in evaluation.

## Local artifacts

- `outputs/embeddings/mouse054_videomae/`: reusable embeddings, windows, provenance.
- `outputs/metrics/mouse054_mount.json`: 80 evaluations and aggregate results.
- `outputs/retrieval/mouse054_mount_examples.json`: five trial-zero seed intervals.
- `outputs/retrieval/mouse054_mount_nearest.json`: top-20 ranked suggestions.
- `outputs/mouse054_inspection.json`: parsed metadata and mounting intervals.

The original archives remain intact. Only the training annotations, the 3.2-second
`mouse036` smoke-test recording, and `mouse054` were extracted. Videos, weights,
and generated outputs remain outside Git via existing ignore rules.
