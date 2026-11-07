import json
import os
import subprocess
import sys

import numpy as np
import pytest

from ai_video_annotator.embeddings import save_cache, source_fingerprint
from ai_video_annotator.evaluation import match_predictions, retrieval_metrics
from ai_video_annotator.types import BehaviorEvent, RetrievalResult, TimeInterval, VideoWindow
from ai_video_annotator.video import generate_windows


def prediction(index, start, end, score, video_id="mouse"):
    return RetrievalResult(VideoWindow(video_id, index, TimeInterval(start, end)), score)


def truth(start, end, video_id="mouse"):
    return BehaviorEvent(video_id, "mount", TimeInterval(start, end))


def test_duplicates_are_false_positives_and_ap_accounts_for_rank():
    predictions = [prediction(0, 0, 2, 1), prediction(1, 0.5, 2.5, 0.9), prediction(2, 5, 7, 0.8)]
    truths = [truth(0, 2), truth(5, 7)]
    hits, matches = match_predictions(predictions, truths)
    assert hits.tolist() == [True, False, True]
    assert matches == (0, None, 1)
    metrics = retrieval_metrics(predictions, truths)
    assert metrics["average_precision"] == pytest.approx((1 + 2 / 3) / 2)
    assert metrics["precision_at_5"] == 0.4
    assert metrics["recall_at_10"] == 1


def test_exact_threshold_and_video_identity():
    p = prediction(0, 10, 14, 1)
    assert match_predictions([p], [truth(12, 16)], 1 / 3)[0].tolist() == [True]
    assert match_predictions([p], [truth(12, 16)], 0.5)[0].tolist() == [False]
    assert match_predictions([p], [truth(10, 14, "other")])[0].tolist() == [False]


def test_empty_predictions_and_truth():
    assert retrieval_metrics([], [truth(0, 1)])["average_precision"] == 0
    assert retrieval_metrics([], [truth(0, 1)])["recall_at_20"] == 0
    assert retrieval_metrics([], [])["average_precision"] is None


def test_score_order_required():
    with pytest.raises(ValueError, match="descending"):
        match_predictions([prediction(0, 0, 2, 0.5), prediction(1, 5, 7, 1)], [truth(0, 2)])


def test_repeated_experiment_cli(tmp_path):
    import av

    video_id = "task1/train/mouse_demo"
    video = tmp_path / "mouse_demo.mp4"
    with av.open(str(video), "w") as container:
        stream = container.add_stream("mpeg4", rate=1)
        stream.width = stream.height = 32
        stream.pix_fmt = "yuv420p"
        for _ in range(16):
            frame = av.VideoFrame.from_ndarray(
                np.zeros((32, 32, 3), dtype=np.uint8), format="rgb24"
            )
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    labels = [2, 2, 3, 3] * 4
    annotations = tmp_path / "annotations.json"
    annotations.write_text(json.dumps({"annotator-id_0": {video_id: {"annotations": labels}}}))
    cache = tmp_path / "cache"
    windows = generate_windows(video_id, 16, 2, 2)
    vectors = np.array([[1, 0] if i % 2 == 0 else [0, 1] for i in range(8)], dtype=np.float32)
    save_cache(cache, vectors, windows, {"source": source_fingerprint(video)})
    output = tmp_path / "metrics.json"
    command = [
        sys.executable,
        "scripts/evaluate.py",
        "--cache",
        str(cache),
        "--annotations",
        str(annotations),
        "--videos-root",
        str(tmp_path),
        "--shots",
        "1",
        "--trials",
        "2",
        "--output",
        str(output),
    ]
    subprocess.run(
        command, env={**os.environ, "PYTHONPATH": "src"}, capture_output=True, check=True
    )
    report = json.loads(output.read_text())
    assert len(report["trials"]) == 8
    assert report["truth_count"] == 4
    assert all(row["truth_count"] == 3 for row in report["trials"])
    assert all(row["average_precision"] == 1 for row in report["trials"])
    assert report["trials"][0]["seed_event_indices"] == report["trials"][2]["seed_event_indices"]
