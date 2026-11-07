# Full-frame VideoMAE preprocessing

The default is now `--preprocessing letterbox`. Each frame is resized with bilinear
interpolation to fit inside the model's square input while preserving aspect ratio,
then centered with padding filled with the checkpoint's RGB mean. Checkpoint
rescaling and normalization still apply. A second resize or center crop is explicitly
disabled, so content at the frame edges is retained.

For the 1024 × 570 CalMS21 video and 224 × 224 checkpoint input, the image becomes
224 × 125 pixels with 49 pixels of padding above and 50 below. Padding is nearly
zero after normalization (uint8 mean rounding introduces at most one-level error).
This preserves the whole arena but gives mice fewer pixels than a center crop.
Letterboxing fixes lost field of view; it does not guarantee better retrieval.

```bash
python scripts/encode_video.py \
  --video data/raw/calms21/task1_videos_mp4/train/mouse054_task1_annotator1.mp4 \
  --video-id task1/train/mouse054_task1_annotator1 \
  --cache outputs/embeddings/mouse054_videomae_letterbox \
  --preprocessing letterbox --batch-size 4 --device mps \
  --revision dc740ceda42fce44faed2ea03c6d447db72f6af9 --local-files-only
```

Use `--preprocessing center-crop` to reproduce the original checkpoint transform in
a new cache. Existing baseline artifacts are preserved. New cache identity includes
a versioned preprocessing mode; a mismatch fails before model loading. Legacy
caches without that field still work for retrieval/evaluation, but are rejected by
the encoding CLI rather than silently reused under a new preprocessing default.

Tests verify that colored content on opposite edges survives for wide and tall
frames, square images receive no padding, mean padding stays neutral after actual
checkpoint normalization, and a local VideoMAE model accepts the resulting tensors.

## Controlled comparison on mouse054

The full-frame run reused the same source video, model commit, windows, random
seed, and exact seed event indices for all 80 evaluations. Both caches remain available.

| Five examples | Center-crop P@10 | Letterbox P@10 | Center-crop AP | Letterbox AP |
|---|---:|---:|---:|---:|
| nearest-example | 0.36 | 0.46 | 0.254 | 0.383 |
| prototype | 0.40 | 0.42 | 0.276 | 0.392 |

Values are means across five trials at temporal IoU ≥ 0.3. At IoU ≥ 0.5,
nearest-example P@10 improved from .26 to .34 and prototype from .18 to .24.
The result supports retaining the full frame for this recording; one recording
and five trials do not establish generalization. One-shot P@10 decreased, so
the improvement is not uniform across settings.

Full results: `outputs/metrics/mouse054_mount_letterbox.json`.
Visual check: `outputs/preprocessing_comparison.jpg`.
New embeddings: `outputs/embeddings/mouse054_videomae_letterbox/`.
