# AI Video Annotator

AI-assisted behavioral video annotation through few-shot temporal retrieval.

The initial research question is:

> Given a handful of researcher-selected behavior intervals in a continuous video, can a frozen pretrained video encoder retrieve additional occurrences of that behavior from the remaining video?

The project is intentionally **not** an automatic behavior classifier. The model ranks candidate temporal regions; a researcher remains responsible for accepting, rejecting, and adjusting annotations.

## Current scope

The application includes a local, video-centered annotation workspace and an ML evaluation pipeline:

```text
continuous video
    -> overlapping temporal windows
    -> frame sampling
    -> frozen video or image encoder
    -> cached window embeddings
    -> few-shot similarity retrieval
    -> temporal de-duplication
    -> evaluation against hidden annotations
```

The first dataset target is CalMS21. The first encoder target is VideoMAE.

Not yet in scope:

- database
- authentication or multi-user support
- AWS/cloud deployment
- fine-tuning or custom model training
- FAISS/vector database
- automatic zero-shot behavior naming

## Local annotation workflow

The workspace includes an editable ethogram, a behavior dropdown, and a clickable
coding pad. Each behavior has a definition, a configurable single-character shortcut,
and a state (duration) or point (instantaneous) event type. Press a state key to start
and again to stop; press a point key once. Unclosed states survive reload and appear
in the ledger until stopped or canceled. They are not included in annotation CSV.
Playback/review shortcuts are reserved; duplicate behavior keys are rejected.

Fly and mouse starter lists are optional and editable. They are operational examples,
not an automatically inferred species or a validated annotation protocol. Ethograms
can be exported/imported as Fieldnote JSON; native BORIS project import is not implemented.
This workflow draws on [BORIS's ethogram and coding model](https://www.boris.unito.it/user_guide/create_project/).

Switching behavior preserves separate reference examples, review decisions, notes,
counterexamples, and search provenance. CSV exports all behaviors' reference examples,
accepted suggestions, and manually coded events; `event_type` distinguishes point and
state annotations. JSON includes the full project, unfinished states, and search history.
Existing sessions retain their behavior names rather than being relabeled by species.
Similarity search uses duration examples; point events currently use manual coding.
Subjects, modifiers, exclusion rules, and a graphical multitrack timeline remain future work.

See [retrieval iteration results](docs/retrieval_iteration_results.md) for paired
1/3/5/10-reference experiments, negative examples, duration proposals, and fresh
full-recording validation. Experimental scoring/duration options remain off by default.

## Repository layout

```text
ai-video-annotator/
├── configs/                  # experiment configuration
├── docs/                     # architecture + experiment protocol
├── src/ai_video_annotator/   # reusable library code
├── tests/                    # fast unit tests
├── data/                     # local datasets; gitignored
└── outputs/                  # generated artifacts; gitignored
```

## Setup

Python 3.12 is the target development version.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[ml,dev]"
```

Run tests:

```bash
pytest
```

Run linting:

```bash
ruff check .
```

## Initial milestones

### M0 — Foundation

- project/package structure
- validated temporal data types
- device selection
- YAML configuration loading
- unit tests

### M1 — CalMS21 ingestion

- locate source videos and annotations
- convert framewise behavior labels into temporal intervals
- inspect dataset statistics
- visually verify several parsed events

### M2 — Video preprocessing + VideoMAE

- generate overlapping temporal windows
- uniformly sample frames from each window
- encode windows with frozen VideoMAE
- cache embeddings locally

### M3 — Few-shot retrieval

- select 1/3/5/10 known examples
- nearest-example retrieval
- prototype retrieval
- temporal non-maximum suppression

### M4 — Evaluation

- temporal IoU matching
- Precision@K / Recall@K
- repeated random few-shot trials
- compare behaviors, window durations, and retrieval strategies

Only after M4 should this become an annotation application.

## Inspect CalMS21 recordings

Milestone 1 now provides a Task 1 JSON adapter, exact video frame counting,
metadata, and contiguous behavior intervals in seconds.
See [dataset setup and inspection](docs/calms21_ingestion.md) for download instructions
and the supported format.

```bash
python scripts/inspect_dataset.py \
  --annotations data/raw/calms21/calms21_task1_train.json \
  --videos-root data/raw/calms21/videos \
  --video-id task1/train/mouse001_task1_annotator1 \
  --behavior mount
```

Add `--json` for machine-readable output. Real recording alignment still requires
expert review of parsed behavior boundaries; basic frame-count alignment and
several frame spot checks have passed on the initial recording.

Milestones 2–3 provide overlapping windows and uniform RGB frame sampling with
sequential decoding and bounded memory. See [video preprocessing](docs/video_preprocessing.md)
for library usage and boundary conventions.

Milestones 4–5 add frozen VideoMAE encoding and validated local embedding caches.
See [encoding instructions](docs/encoding.md) for the command, model download,
cache reuse, and reproducibility settings. Pretrained-model and MPS execution
have been validated on real CalMS21 recordings.

Milestones 6–7 add nearest-example/prototype similarity retrieval and temporal NMS.
The [retrieval guide](docs/retrieval.md) explains example JSON, ranked timestamp
reports, seed exclusion, and the cache-only CLI.

Milestone 8 adds repeated few-shot experiments and one-to-one temporal matching.
See [evaluation instructions](docs/evaluation.md) for metrics and experimental conventions.

VideoMAE now defaults to full-frame letterboxing to retain animals near image edges.
See [preprocessing details](docs/preprocessing.md) for the controlled comparison.

## Local annotation workspace

The working Fieldnote interface plays local recordings, retrieves suggestions from
cached embeddings, and saves annotation decisions to disk. See
[local workspace instructions](docs/local_annotation.md) for the launch command,
keyboard controls, exports, and current scope.

## Reliability benchmarks

[Evaluation protocol](docs/reliability_protocol.md) describes event-level
precision/recall/F1, fixed test sets across example counts, random-ranking controls,
within/cross-video experiments, and what remains needed for multi-dataset validation.
Use `scripts/prepare_calms21_benchmark.py`, `scripts/benchmark.py`, and
`scripts/report_benchmark.py` to prepare, run, and report the local pilot.

The [13 October reliability pilot](docs/reliability_results.md) reports measured
results and limitations. The [15 October encoder comparison](docs/encoder_comparison_results.md)
adds measured BioCLIP, DINOv2 and VideoMAE results on mice and fly excerpts, including
paired learning curves and precision/recall/F1. See the
[comparison protocol](docs/encoder_comparison_protocol.md) for reproducibility.
Custom species can use the generic event CSV adapter.

The [additional model tests](docs/extended_encoder_results.md) add measured general
CLIP and ResNet-50 insect results, per-video variation and an explicitly incomplete
BioCLIP 2 resource trial. Run `scripts/extend_encoder_comparison.py` to reproduce
the additional models; it defaults to the four fly excerpts and preserves existing
results when adding variants.
