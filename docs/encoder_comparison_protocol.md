# Frozen biological/image/video encoder comparison

This experiment evaluates original BioCLIP (imageomics/bioclip, ViT-B/16),
DINOv2 Small (facebook/dinov2-small), and the existing VideoMAE Base.
BioCLIP 2/2.5 and other newer encoders are not included in this first comparison.
No encoder is trained or fine-tuned. Both image models use native image features,
not species predictions, text prompts, or a behavior classifier.

## Matched inputs and pooling

For each dataset, all encoders use the same source pixels, window boundaries,
16 uniformly sampled frames, full-frame letterboxing to 224 square, fixed example
selection and evaluation truth. Each model uses its own checkpoint normalization.
VideoMAE uses the existing mean of final spatiotemporal tokens. Image models use
DINO's CLS feature or BioCLIP's projected image feature, with per-frame L2 normalization.
Two clip variants are computed from the SAME frame features:

- `mean`: mean over the 16 frames, then clip L2 normalization.
- `ordered4`: means over four consecutive equal-sized frame bins, concatenated,
  then clip L2 normalization. This retains coarse temporal position, not learned
  motion or optical flow. Temporal phase changes may hurt retrieval.

Image encoders reuse features for identical sampled source-frame indices across
windows. Feature reuse does not change which frames contribute to a window or
its multiplicity: repeated frames in a short clip still receive their sampled weight.
Both variants are cached atomically using the existing cache format. Different
models/poolings have separate cache paths. Model commit IDs and exact dependency
versions are stored under `outputs/model-cache/comparison_*.json`.

The comparison runner verifies identical source hashes, full window lists, FPS,
frame counts, duration, sampling, preprocessing geometry and temporal settings
across encoders. Different feature dimensions and pooling rules are intentional.
No vector comparisons are performed between different encoders' feature spaces.

## Mouse dataset

Reuse all six verified CalMS21 pilot recordings and their original splits, annotations,
2-second windows and 0.5-second stride. Keep the four development/training sources
separate from the two previously evaluated test targets. Because test outcomes from
these recordings have already been inspected, encoder selection based on this
comparison requires fresh held-out recordings for a final generalization claim.

## Insect dataset

Fly-vs-Fly Aggression.zip is downloaded from CaltechDATA and verified against its
published MD5 `db2b527b800f4013d1e583f53b116c18`. Ten MP4 videos, action MATLAB files,
and the official `traintest.mat` are extracted. The labels use one-based inclusive
frame endpoints, confirmed in the authors' `visualizer.m` where bouts are indexed
`start:end`. Conversion to half-open seconds is `(first-1)/fps` through `last/fps`.
Decoded source frame counts are checked against annotation bounds.

This first pilot uses fixed, continuous 600–720 second excerpts from movie2 and
movie3 (official training), and movie6 and movie8 (official test). Movies/offsets
were chosen before inspecting excerpt event counts; the choices are recorded in
`flyvsfly_manifest.json`. Each full source has 54,000 frames at 30 fps, 144×144.
Excerpts use lossless H.264 at the same FPS, reset to frame-zero with original source,
frame offset and derivative hashes retained. They are excerpts, not evaluations
of whole recordings or the full insect dataset.

The videos contain extremely short lunges. A 2-second proposal has inadequate
geometric coverage for these events, so insect windows are predeclared at 0.25s,
stride 0.125s, for EVERY encoder. Sixteen samples repeat available frames when
fewer than sixteen original frames exist. These windows are held fixed across
models; insect absolute scores should not be compared directly with mouse scores
as if the tasks had identical geometry.

This is scene-level behavior retrieval, not actor identification. Overlapping bouts
of the same behavior across the two actors are unioned so simultaneous actors do
not inflate event counts. Distinct touching bouts remain separate. Third-column
annotation flags are retained in the original MATLAB data/provenance; they are
handled as in the authors' visualizer, which uses both endpoints without filtering
these flags. We do not invent an interpretation of those flags.

Unknown-labeled regions and boundary-truncated events are ignored; candidates and
truth overlapping these regions are excluded. Source examples whose mapped cached
windows intersect ignored regions are ineligible. Only fully contained, known
behavior bouts enter truth. This avoids treating ambiguous or partial bouts as
reliable negatives. An expert audit of this scene-level conversion remains needed.

Lunge has enough source examples for the main curve. Wing-threat/tussle have sparse
examples in the selected excerpts, and unavailable shot counts must be reported,
not silently replaced with handpicked richer clips. Secondary annotator labels in
`movie6_second_actions.mat` are preserved but not combined with primary labels.

## Scoring

Use the fixed-pool protocol: nested 1/3/5/10 references, 30 trials, seed 42,
nearest-example and mean-prototype retrieval, a paired uniform random-ranking
control, temporal NMS 0.3, event IoU 0.3/0.5, review budgets 10/20/50, full-ranking
non-interpolated AP. Within-video test truth/exclusions remain fixed across shot
counts in each trial. Across-video examples come only from train sources.

Random controls, selected reference indices and target truth counts must match
across encoders. Report positive-target and no-positive-target results separately,
source/target counts, missing shots, temporal coverage and recording-balanced means.
Repeated seed trials do not turn a small number of recordings into independent
population observations. Keep per-pair variability and representative matched cases.

## Commands

```sh
HF_HOME=outputs/model-cache PYTHONPATH=src .venv/bin/python scripts/encode_image_video.py \
  --video data/raw/calms21/task1_videos_mp4/train/mouse054_task1_annotator1.mp4 \
  --video-id task1/train/mouse054_task1_annotator1 \
  --backend bioclip --model imageomics/bioclip \
  --revision ce901ab3c6a913f9e9ef94ce6d27761069f4f01c \
  --cache-prefix outputs/embeddings/comparison/mouse054_task1_annotator1_bioclip \
  --device mps --local-files-only

PYTHONPATH=src .venv/bin/python scripts/prepare_flyvsfly_benchmark.py
PYTHONPATH=src .venv/bin/python scripts/compare_encoders.py
```

The image command writes both `_mean` and `_ordered4` caches. DINO uses backend
`dinov2`, model `facebook/dinov2-small`, revision
`ed25f3a31f01632728cabb09d1542f84ab7b0056`. For insect excerpts use window .25 and
stride .125 in both image and VideoMAE encoding commands; VideoMAE must use the
existing frozen commit. All required caches must exist before comparison starts.

Sources: https://github.com/Imageomics/bioclip,
https://github.com/facebookresearch/dinov2,
https://data.caltech.edu/records/zrznw-w7386,
https://github.com/MCG-NJU/VideoMAE.

## Additional-model insect extension

`scripts/extend_encoder_comparison.py` adds general CLIP
(`laion/CLIP-ViT-B-16-laion2B-s34B-b88K`, commit
`7288da5a0d6f0b51c4a2b27c624837a9236d0112`) and TorchVision ResNet-50
(`IMAGENET1K_V2`, official hash-checked `resnet50-11ad3fa6.pth`). Both use mean
and ordered4 pooling, all four existing insect excerpts, identical geometry,
examples, ignored regions and paired controls. It adds no mouse results.
General CLIP uses native projected image features (512 dimensions). ResNet-50
uses 2048-D features after average pooling, with the classification head replaced
by identity. Both use checkpoint normalization and the shared 224-pixel bilinear
letterbox; the checkpoint's default crop is intentionally replaced.

Selection follows inspection of the first pilot, so this is an exploratory
comparison rather than fresh confirmation. General CLIP controls for the original
BioCLIP architecture; ResNet adds a smaller supervised CNN. Architecture/training
changes do not isolate a causal effect of biology pretraining. Completed new models
use batch size 8 and float32 inference. Batch size and weight-loading method are
stored in cache metadata. ResNet metadata also pins the TorchVision version.

BioCLIP 2 (`imageomics/bioclip-2`, commit
`2957b322090f9cb17ae72c71981c7218a28d81e0`) was downloaded and attempted with
batches of 8 and 4. Partial processing remained too slow under current resource
pressure; both workers were stopped, with no incomplete features or accuracy scores
reported. Process sampling showed waiting on Metal completion; macOS also reported
about 8GB of swap in use. This does not establish the cause or general hardware
incompatibility. The optional model remains available through `--models bioclip2`.

The new OpenCLIP loader constructs the native visual tower on the meta device and
assigns memory-mapped visual checkpoint tensors, avoiding unused text weights. A
test checks exact equality with a complete native CLIP and rejects missing visual
weights. MPS allocations for the new OpenCLIP and ResNet adapters are capped at
half the recommended working set. This does not change the representation or scoring.

```sh
PYTHONPATH=src .venv/bin/python scripts/extend_encoder_comparison.py --device mps
PYTHONPATH=src .venv/bin/python scripts/report_extended_encoders.py
```

The extension defaults to general CLIP and ResNet, runs models serially, and resumes
from validated caches. For the longer mouse experiment, pass `--datasets calms21`
or `--datasets all`. The comparison runner accepts `--variants` and preserves
other inventory entries. The combined report audits all completed configurations.
Sources: https://huggingface.co/laion/CLIP-ViT-B-16-laion2B-s34B-b88K,
https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html,
and https://huggingface.co/imageomics/bioclip-2.
