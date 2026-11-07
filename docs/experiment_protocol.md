# Initial Experiment Protocol

## Research question

Given a small number of researcher-selected examples of one behavior in a continuous video, can a frozen pretrained video representation retrieve additional occurrences from the remaining video?

## Initial system

- Dataset: CalMS21
- Encoder: VideoMAE Base, frozen
- Candidate window duration: 2.0 seconds
- Candidate stride: 0.5 seconds
- Frames sampled per candidate: 16
- Few-shot counts: 1, 3, 5, 10
- Retrieval strategies: nearest example and prototype
- Postprocessing: temporal non-maximum suppression
- Evaluation: event-level temporal matching

## Leakage rules

- A selected example interval must not also be counted as a retrieved event.
- Overlapping windows representing the same ground-truth event must not count as multiple discoveries.
- Cross-video evaluation must split at the video/session level rather than randomly splitting neighboring windows.

## First success criterion

The first milestone is not a production accuracy target. It is a reproducible experiment that can:

1. parse one real annotated video,
2. generate candidate temporal windows,
3. produce one cached embedding per window,
4. rank candidates from a small example set, and
5. compute event-level retrieval metrics against hidden labels.
