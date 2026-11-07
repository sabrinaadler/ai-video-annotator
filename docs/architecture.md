# Architecture

## Current boundary

The repository is an ML experiment system, not yet an annotation application.

The first complete data path is:

```text
source video
    -> temporal windows
    -> sampled frames
    -> frozen VideoMAE encoder
    -> cached embeddings
    -> few-shot similarity search
    -> temporal postprocessing
    -> evaluation
```

## Design rules

1. Dataset adapters must not depend on model code.
2. Video decoding/sampling must not depend on a particular encoder.
3. Retrieval must consume embeddings without knowing which encoder produced them.
4. Expensive video encoding should happen once and be cached.
5. Ground-truth labels are available to evaluation but hidden from retrieval except for the simulated few-shot examples.
6. Candidate windows are temporal segments of full videos, not separately maintained cropped-video datasets.
7. Generated data, videos, embeddings, and experiment outputs remain outside Git.

## Deferred components

The following should not be introduced until retrieval quality is understood:

- web frontend
- FastAPI backend
- SQL database
- cloud storage
- authentication
- multi-user collaboration
- vector databases
- custom model training
