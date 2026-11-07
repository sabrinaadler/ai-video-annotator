"""Experiment configuration loading and validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    name: str
    root: Path


@dataclass(frozen=True, slots=True)
class VideoConfig:
    window_seconds: float
    stride_seconds: float
    num_frames: int


@dataclass(frozen=True, slots=True)
class EncoderConfig:
    type: str
    model_name: str
    batch_size: int
    device: str = "auto"


@dataclass(frozen=True, slots=True)
class RetrievalConfig:
    method: str
    num_examples: int


@dataclass(frozen=True, slots=True)
class PostprocessingConfig:
    nms_iou_threshold: float


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    match_iou_thresholds: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    seed: int


@dataclass(frozen=True, slots=True)
class AppConfig:
    dataset: DatasetConfig
    video: VideoConfig
    encoder: EncoderConfig
    retrieval: RetrievalConfig
    postprocessing: PostprocessingConfig
    evaluation: EvaluationConfig
    experiment: ExperimentConfig


def _require_mapping(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Configuration section {key!r} must be a mapping")
    return value


def _probability(value: Any, name: str) -> float:
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return number


def load_config(path: str | Path) -> AppConfig:
    """Load and validate a YAML experiment configuration."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    if not isinstance(raw, dict):
        raise ValueError("Top-level configuration must be a mapping")

    dataset = _require_mapping(raw, "dataset")
    video = _require_mapping(raw, "video")
    encoder = _require_mapping(raw, "encoder")
    retrieval = _require_mapping(raw, "retrieval")
    postprocessing = _require_mapping(raw, "postprocessing")
    evaluation = _require_mapping(raw, "evaluation")
    experiment = _require_mapping(raw, "experiment")

    window_seconds = float(video["window_seconds"])
    stride_seconds = float(video["stride_seconds"])
    num_frames = int(video["num_frames"])
    batch_size = int(encoder["batch_size"])
    num_examples = int(retrieval["num_examples"])

    if window_seconds <= 0:
        raise ValueError("video.window_seconds must be positive")
    if stride_seconds <= 0:
        raise ValueError("video.stride_seconds must be positive")
    if num_frames <= 0:
        raise ValueError("video.num_frames must be positive")
    if batch_size <= 0:
        raise ValueError("encoder.batch_size must be positive")
    if num_examples <= 0:
        raise ValueError("retrieval.num_examples must be positive")

    thresholds = tuple(
        _probability(value, "evaluation.match_iou_thresholds")
        for value in evaluation["match_iou_thresholds"]
    )
    if not thresholds:
        raise ValueError("evaluation.match_iou_thresholds must contain at least one value")

    return AppConfig(
        dataset=DatasetConfig(
            name=str(dataset["name"]),
            root=Path(dataset["root"]),
        ),
        video=VideoConfig(
            window_seconds=window_seconds,
            stride_seconds=stride_seconds,
            num_frames=num_frames,
        ),
        encoder=EncoderConfig(
            type=str(encoder["type"]),
            model_name=str(encoder["model_name"]),
            batch_size=batch_size,
            device=str(encoder.get("device", "auto")),
        ),
        retrieval=RetrievalConfig(
            method=str(retrieval["method"]),
            num_examples=num_examples,
        ),
        postprocessing=PostprocessingConfig(
            nms_iou_threshold=_probability(
                postprocessing["nms_iou_threshold"],
                "postprocessing.nms_iou_threshold",
            ),
        ),
        evaluation=EvaluationConfig(match_iou_thresholds=thresholds),
        experiment=ExperimentConfig(seed=int(experiment["seed"])),
    )
