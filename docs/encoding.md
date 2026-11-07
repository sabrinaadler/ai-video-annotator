# Frozen VideoMAE encoding and caching

Install the package and ML extras in the project environment:

```bash
source .venv/bin/activate
pip install -e ".[ml,dev]"
python scripts/encode_video.py \
  --video data/raw/calms21/videos/mouse001_task1_annotator1.mp4 \
  --video-id task1/train/mouse001_task1_annotator1 \
  --cache outputs/embeddings/mouse001_videomae
```

Adjust the video path to the extracted archive layout. The CLI also works with any
local constant-frame-rate video and does not load behavior annotations.

Defaults: 2-second windows, 0.5-second stride, 16 frames, batches of 8, and
`MCG-NJU/videomae-base`. Device selection prefers MPS, then CUDA, then CPU.
Use `--batch-size 1` if memory is tight. Use `--device cpu` for explicit CPU runs.

On first use, Transformers downloads the pretrained model and image processor.
The encoder freezes parameters, switches to evaluation mode, and uses inference
mode. The default full-frame letterbox preserves aspect ratio and pads to the model input
size; the checkpoint processor then applies rescaling and normalization. Use
`--preprocessing center-crop` for the original transform. Embeddings are the mean of final spatial/temporal token vectors,
returned as float32. No classifier, training, or fine-tuning is involved.
See [preprocessing details](preprocessing.md) for edge retention and the input
resolution tradeoff. Preprocessing mode is part of each new cache identity.

The output directory contains:

- `embeddings.npy`: N × D float32 vectors (unnormalized).
- `windows.json`: ordered window IDs and intervals in seconds.
- `metadata.json`: video content fingerprint, sampling settings, model identity,
  resolved model revision, library version, frame rate, and embedding shape.

Run the same command again to reuse the cache without loading the model or
sampling video frames. Reuse reads the video to verify its SHA-256 fingerprint.
Different input or encoding settings fail explicitly; choose another output
folder for a new experiment. Existing caches are never overwritten. A new cache
is published only after all files have been written successfully.

For reproducible experiments, pass a model commit hash via `--revision` rather
than the mutable default `main`. Cache reuse preserves the already encoded
revision even if `main` subsequently changes. `--local-files-only` prevents model
downloads once weights are available locally.

The CLI currently probes frame counts with one decoding pass, then samples in a
second sequential pass. It retains only an overlap buffer and the current batch
of pixel arrays; the comparatively small output embedding matrix stays in memory.
Float32 inference is the initial baseline. Pretrained weights and MPS execution
have been validated on real CalMS21 data; see [the first experiment](first_experiment.md)
for retrieval results. Unit tests use a tiny randomly initialized local VideoMAE model to verify the
actual Transformers processor, tensor shapes, pooling, and frozen inference.

Reference: [Hugging Face VideoMAE documentation](https://huggingface.co/docs/transformers/model_doc/videomae).
