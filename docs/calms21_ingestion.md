# CalMS21 Task 1 ingestion

Download and extract `task1_classic_classification.zip` and the corresponding
MP4 recordings from the [official CaltechDATA release](https://data.caltech.edu/records/s0vdx-0k302).
Keep downloads under `data/raw/calms21/`, which is ignored by Git.
The annotation archive alone contains trajectories and labels, not video pixels.
The MP4 archive is approximately 28 GB; download it separately when ready.

This adapter reads the official nested JSON format:

```json
{
  "annotator-id_0": {
    "task1/train/mouse001_task1_annotator1": {
      "annotations": [3, 2, 2, 3],
      "metadata": {"vocab": {"attack": 0, "investigation": 1, "mount": 2, "other": 3}}
    }
  }
}
```

Use the original `calms21_task1_train.json`, not `taskprog_features` files.
Converted pickled `.npy` dictionaries and Tasks 2/3 are not supported yet.
JSON parsing streams labels and vocabulary, skipping unused pose fields to keep memory bounded.

After installing the package with its `ml` extras:

```bash
python scripts/inspect_dataset.py \
  --annotations data/raw/calms21/calms21_task1_train.json \
  --videos-root data/raw/calms21/videos \
  --video-id task1/train/mouse001_task1_annotator1 \
  --behavior mount
```

Adjust paths and sequence ID to match your extracted files. Omit `--behavior` to
show all behaviors, omit `--video-id` to inspect all recordings, or add `--json`
for structured output. Full sequence IDs prevent mixing train/test recordings.
Videos are located recursively by exact sequence basename; duplicate matches fail.

The adapter uses per-record `metadata.vocab`, falling back to the documented Task 1
mapping when absent. Each contiguous run becomes a half-open interval
`[first_frame / fps, (last_frame + 1) / fps)`. Background `other` is preserved.
FPS comes from the video stream, never a hardcoded 30. This initial adapter assumes
constant frame rate and frame-zero alignment. It decodes the video once per adapter
instance to count frames exactly and rejects label/video length mismatches rather
than silently rescaling or truncating annotations.

Before an experiment, inspect several reported intervals in a video player to
verify alignment and behavior identity. Synthetic tests establish parsing and codec
integration; they do not establish alignment on an actual CalMS21 recording.
