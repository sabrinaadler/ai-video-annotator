import json
import os
import subprocess
import sys

import numpy as np
import pytest

from ai_video_annotator.embeddings import save_cache
from ai_video_annotator.retrieval import (
    example_window_indices,
    retrieve,
    similarity_scores,
    temporal_nms,
)
from ai_video_annotator.types import BehaviorEvent, RetrievalResult, TimeInterval, VideoWindow


def window(index, start, end, video_id="mouse"):
    return VideoWindow(video_id, index, TimeInterval(start, end))


def test_nearest_example_cosine_and_scale():
    candidates = np.array([[1, 0], [0, 1], [-1, 0], [1, 1]])
    examples = np.array([[2, 0], [0, 3]])
    np.testing.assert_allclose(similarity_scores(candidates, examples), [1, 1, 0, 1 / np.sqrt(2)])


def test_prototype_uses_mean_raw_embeddings():
    candidates = np.eye(2)
    examples = np.array([[2, 0], [0, 1]])
    np.testing.assert_allclose(
        similarity_scores(candidates, examples, "prototype"), [2 / np.sqrt(5), 1 / np.sqrt(5)]
    )
    assert similarity_scores(np.array([[1, 0]]), np.array([[-1, 0]]))[0] == -1


@pytest.mark.parametrize(
    "candidates,examples,method",
    [
        ([[0, 0]], [[1, 0]], "nearest-example"),
        ([[1, 0]], [[1, 0], [-1, 0]], "prototype"),
        ([[1, 0]], np.empty((0, 2)), "nearest-example"),
        ([[1, 0]], [[1]], "nearest-example"),
        ([[np.nan, 0]], [[1, 0]], "prototype"),
        ([[1, 0]], [[1, 0]], "unknown"),
    ],
)
def test_invalid_similarity(candidates, examples, method):
    with pytest.raises(ValueError):
        similarity_scores(np.asarray(candidates), np.asarray(examples), method)


def test_nms_ranks_before_suppressing_and_limits_afterward():
    results = [
        RetrievalResult(window(0, 0, 2), 0.8),
        RetrievalResult(window(1, 0.5, 2.5), 0.9),
        RetrievalResult(window(2, 5, 7), 0.7),
    ]
    kept = temporal_nms(results, 0.3, 2)
    assert [result.window.index for result in kept] == [1, 2]


def test_nms_ties_touching_and_separate_videos():
    results = [
        RetrievalResult(window(1, 2, 4), 0.8),
        RetrievalResult(window(0, 0, 2), 0.8),
        RetrievalResult(window(0, 0, 2, "other"), 0.8),
    ]
    kept = temporal_nms(results, 0)
    assert len(kept) == 3
    assert kept[0].window.index == 0


@pytest.mark.parametrize("threshold,k", [(float("nan"), 10), (-1, 10), (0.3, 0)])
def test_invalid_nms(threshold, k):
    with pytest.raises(ValueError):
        temporal_nms([], threshold, k)


def test_example_mapping_and_duplicate_rejection():
    windows = [window(0, 0, 2), window(1, 1, 3), window(2, 2, 4)]
    examples = [BehaviorEvent("mouse", "mount", TimeInterval(1.1, 2.9))]
    assert example_window_indices(windows, examples) == (1,)
    with pytest.raises(ValueError, match="same cached window"):
        example_window_indices(windows, examples * 2)
    with pytest.raises(ValueError, match="does not overlap"):
        example_window_indices(windows, [BehaviorEvent("wrong", "mount", TimeInterval(1, 2))])


def test_retrieval_excludes_seed_neighborhood_and_keeps_unseen_events():
    windows = [
        window(0, 0, 2),
        window(1, 0.5, 2.5),
        window(2, 5, 7),
        window(3, 5.5, 7.5),
        window(4, 9, 11),
    ]
    vectors = np.array([[1, 0], [1, 0], [0.9, 0.1], [0.8, 0.2], [0, 1]], dtype=np.float32)
    examples = [BehaviorEvent("mouse", "mount", TimeInterval(0, 2))]
    for method in ("nearest-example", "prototype"):
        results = retrieve(vectors, windows, examples, method, top_k=2)
        assert [result.window.index for result in results] == [2, 4]


def test_no_unseen_candidates_returns_empty():
    windows = [window(0, 0, 2)]
    examples = [BehaviorEvent("mouse", "mount", TimeInterval(0, 2))]
    assert retrieve(np.array([[1, 0]]), windows, examples) == ()


def test_cli_uses_only_cache_and_examples(tmp_path):
    windows = (window(0, 0, 2), window(1, 5, 7), window(2, 10, 12))
    cache = tmp_path / "cache"
    save_cache(cache, np.array([[1, 0], [0.9, 0.1], [0, 1]], dtype=np.float32), windows, {})
    examples = tmp_path / "examples.json"
    examples.write_text(json.dumps([{"start_seconds": 0, "end_seconds": 2}]))
    output = tmp_path / "report.json"
    command = [
        sys.executable,
        "scripts/retrieve.py",
        "--cache",
        str(cache),
        "--examples",
        str(examples),
        "--behavior",
        "mount",
        "--output",
        str(output),
    ]
    result = subprocess.run(
        command, env={**os.environ, "PYTHONPATH": "src"}, capture_output=True, text=True, check=True
    )
    assert "2 suggestions" in result.stdout
    report = json.loads(output.read_text())
    assert report["selected_window_indices"] == [0]
    assert report["results"][0]["window"]["index"] == 1
    before = output.read_bytes()
    result = subprocess.run(
        command, env={**os.environ, "PYTHONPATH": "src"}, capture_output=True, text=True
    )
    assert result.returncode == 2
    assert output.read_bytes() == before
