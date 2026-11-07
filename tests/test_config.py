from pathlib import Path

import pytest

from ai_video_annotator.config import load_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_load_debug_config() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "debug.yaml")

    assert config.dataset.name == "calms21"
    assert config.video.window_seconds == pytest.approx(2.0)
    assert config.video.num_frames == 16
    assert config.encoder.type == "videomae"
    assert config.encoder.device == "auto"
    assert config.retrieval.num_examples == 2
    assert config.evaluation.match_iou_thresholds == (0.5,)
