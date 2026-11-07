# Within-video evaluation

```bash
python scripts/evaluate.py \
  --cache outputs/embeddings/mouse054_videomae \
  --annotations data/raw/calms21/task1_classic_classification/calms21_task1_train.json \
  --videos-root data/raw/calms21/task1_videos_mp4 \
  --behavior mount --shots 1 3 5 10 --trials 5 --seed 42 \
  --output outputs/metrics/mouse054_mount.json
```

The evaluator verifies that the cache source fingerprint matches the annotated
recording. Each trial samples distinct seed windows without replacement. The same
examples are used for both methods and both IoU thresholds. Only seed events are
passed to retrieval; remaining annotations are used exclusively for evaluation.

Seeds are excluded from the ground-truth denominator. All other events remain,
including events near excluded seed neighborhoods. When multiple bouts map to the
same cached window, only the first is eligible for seed selection; the report records
eligible counts and exact seed event indices. This policy should be revisited for
very fragmented behaviors. The CLI requires at least one remaining eligible event.

Matching proceeds in descending prediction score order. Each prediction matches the
highest-IoU unmatched ground-truth event in the same video, provided IoU is at least
0.3 or 0.5. Each truth can match once; residual duplicate predictions are false positives.

Precision@5/10/20 divides hits by K, even when fewer than K suggestions survive.
Recall@10/20/50 divides by all nonseed events. Average Precision sums precision at
true-positive ranks and divides by nonseed truth count; it uses the full NMS-ranked
list, not just the top 50. With no truth, recall/AP are null. Summary standard
deviations are population standard deviations over the five trials.

These are development results for one recording, not a held-out test benchmark or
an estimate of researcher time saved. A fixed 2-second prediction cannot achieve
high IoU with a very long or very short bout even if it points to the right behavior.
Review example durations and candidate coverage before interpreting low metrics.

## First real-data validation

The initial selected training recording is `mouse054_task1_annotator1`: 17,319
frames at 30 fps (577.3 seconds), with 20 mounting bouts. Its annotation count
matches the decoded source exactly. With the fixed 2-second / 0.5-second candidate
grid, only 16 of 20 bouts have any candidate at IoU ≥ 0.3, and only 12 at IoU ≥ 0.5.
These are candidate coverage ceilings before seed exclusions and NMS, not model
performance. Several bouts are much shorter than two seconds.

Frame spot checks also show animals near lateral image edges. The checkpoint's
standard resize and center crop may exclude them. The first run retains this
baseline for reproducibility; full-frame preprocessing is a clear follow-up
experiment if retrieval is weak. Do not infer an annotation speed benefit from
this single development recording.
