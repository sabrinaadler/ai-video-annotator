# Few-shot retrieval from a cached recording

Create a JSON file containing only researcher-selected example intervals, in seconds:

```json
[
  {"start_seconds": 31.2, "end_seconds": 34.8},
  {"start_seconds": 82.1, "end_seconds": 84.27}
]
```

Save it as `outputs/retrieval/mount_examples.json`. These numbers are illustrative;
replace them with actual examples from your recording. The cache supplies the video
ID and `--behavior` supplies the behavior name. Retrieval does not read dataset
annotations, source video, or encoder weights.

```bash
python scripts/retrieve.py \
  --cache outputs/embeddings/mouse001_videomae \
  --examples outputs/retrieval/mount_examples.json \
  --behavior mount \
  --method nearest-example \
  --top-k 20 \
  --nms-iou 0.3 \
  --output outputs/retrieval/mount_nearest.json
```

Repeat with `--method prototype` and a different output filename to compare methods.
Existing reports are preserved; choose a new filename to rerun.

Each example maps to the cached window with maximum temporal IoU. Ties use the first
cache row. Examples mapping to the same window are rejected to prevent accidental
weighting of duplicate seeds. The report includes the selected row indices so this
approximation can be inspected. Long examples still contribute one window; no
additional model inference occurs. For the first experiment, prefer examples close
to the configured window length.

Nearest-example scores are the maximum cosine similarity to any selected example
vector. Prototype scores are cosine similarity to the mean of the raw example
vectors. Vectors are normalized during cosine calculation, not overwritten in the
cache. Scores are similarities, not probabilities. Zero vectors and a cancelling
zero prototype fail explicitly.

Candidates overlapping a labeled interval or its selected cached window are excluded,
including adjacent overlapping windows that would otherwise rediscover the seeds.
This also removes unseen events inside that immediate neighborhood; keep the same
exclusion policy when evaluating later.

Greedy temporal NMS sorts by descending score and suppresses same-video suggestions
whose temporal IoU with a retained suggestion exceeds the chosen threshold. Equal
scores use video ID, start time, and window index. Top-K applies after suppression.
The default 0.3 threshold suppresses adjacent 2-second windows with 0.5-second stride.
NMS reduces duplicate suggestions but cannot guarantee one prediction per biological
event: sufficiently distant windows within a long event can survive. Suggestions
retain window boundaries for researcher review rather than claiming precise event
boundaries. One-to-one event matching will handle residual duplicates in evaluation.

Tests establish scoring formulas, deterministic ranking, exclusion of seed regions,
NMS behavior, and an offline CLI round trip. Scientific performance still requires
pretrained embeddings from real CalMS21 recordings. Temporal matching and retrieval
metrics are the next milestone.
