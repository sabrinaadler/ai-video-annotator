from fractions import Fraction

import av
import numpy as np
import pytest

from ai_video_annotator.datasets.calms21 import probe_video
from ai_video_annotator.embeddings.cache import load_cache, save_cache
from ai_video_annotator.embeddings.frame_bank import encode_frame_bank, pool_frame_bank
from ai_video_annotator.encoders.image import pool_frame_features
from ai_video_annotator.retrieval.search import example_window_indices, retrieve
from ai_video_annotator.retrieval.similarity import similarity_scores
from ai_video_annotator.types import BehaviorEvent, TimeInterval, VideoWindow
from ai_video_annotator.video.sampling import sample_frame_indices
from ai_video_annotator.video.windows import generate_multiscale_windows


def test_counterexamples_demote_lookalikes_and_exclude_reviewed_intervals():
    vectors = np.array([[1, 0], [1, 0.8], [0.95, 0.31], [0.8, -0.6]], np.float32)
    windows = tuple(VideoWindow("v", i, TimeInterval(2 * i, 2 * i + 1)) for i in range(4))
    positive = BehaviorEvent("v", "target", windows[0].interval)
    negative = BehaviorEvent("v", "target", windows[1].interval)
    assert retrieve(vectors, windows, [positive])[0].window.index == 2
    ranked = retrieve(vectors, windows, [positive], negative_examples=[negative], negative_weight=1)
    assert [r.window.index for r in ranked] == [3, 2]
    with pytest.raises(ValueError, match="distinct"):
        retrieve(vectors, windows, [positive], negative_examples=[positive])
    with pytest.raises(ValueError):
        similarity_scores(vectors, vectors[:1], negative_weight=float("nan"))
    np.testing.assert_array_equal(
        similarity_scores(vectors, vectors[:1]),
        similarity_scores(vectors, vectors[:1], negative_examples=vectors[1:2], negative_weight=0),
    )


def test_counterexample_features_never_expand_outside_confirmed_negative_clip():
    windows = generate_multiscale_windows("v", 4, [0.25, 1])
    vectors = np.array([[1, w.interval.start_seconds + 0.1] for w in windows], np.float32)
    positive = BehaviorEvent("v", "target", TimeInterval(0, 1))
    short_negative = BehaviorEvent("v", "target", TimeInterval(2, 2.25))
    results = retrieve(
        vectors,
        windows,
        [positive],
        reference_window_seconds=1,
        negative_examples=[short_negative],
        negative_weight=0.5,
    )
    selected_negative = next(
        i for i, w in enumerate(windows) if w.interval == short_negative.interval
    )
    selected_positive = example_window_indices(windows, [positive], 1)[0]
    expected = similarity_scores(
        vectors, vectors[[selected_positive]], negative_examples=vectors[[selected_negative]]
    )
    for result in results:
        assert result.score == expected[result.window.index]
    too_short = BehaviorEvent("v", "target", TimeInterval(2.01, 2.05))
    with pytest.raises(ValueError, match="complete cached window"):
        retrieve(vectors, windows, [positive], negative_examples=[too_short])
    assert retrieve(vectors, windows, [positive], negative_examples=[too_short], negative_weight=0)


def test_frame_bank_resumes_nondivisible_chunks_and_reuses_completed_work(tmp_path):
    path = tmp_path / "tiny.mp4"
    with av.open(str(path), "w") as output:
        stream = output.add_stream("libx264", rate=8)
        stream.width = stream.height = 16
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "0"}
        for i in range(20):
            frame = av.VideoFrame.from_ndarray(
                np.full((16, 16, 3), i * 10, np.uint8), format="rgb24"
            )
            frame.pts, frame.time_base = i, Fraction(1, 8)
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    metadata = probe_video(path, "v")

    class Encoder:
        embedding_dim, frame_batch_size = 2, 4

        def __init__(self, stop=False):
            self.calls, self.stop = 0, stop

        def encode_frames(self, frames):
            self.calls += 1
            if self.stop and self.calls == 3:
                raise RuntimeError("interrupted")
            return np.array([[1, f.mean() / 255] for f in frames], np.float32)

    with pytest.raises(RuntimeError, match="interrupted"):
        encode_frame_bank(Encoder(True), metadata, tmp_path / "resume", {}, chunk_size=6)
    assert len(list((tmp_path / "resume").glob("chunk-*.npy"))) == 1
    resumed = encode_frame_bank(Encoder(), metadata, tmp_path / "resume", {}, chunk_size=6)
    reference = encode_frame_bank(Encoder(), metadata, tmp_path / "reference", {}, chunk_size=6)
    np.testing.assert_array_equal(resumed, reference)
    reused = Encoder(True)
    np.testing.assert_array_equal(
        encode_frame_bank(reused, metadata, tmp_path / "resume", {}, chunk_size=6), reference
    )
    assert reused.calls == 0
    with pytest.raises(ValueError, match="identity"):
        encode_frame_bank(
            Encoder(), metadata, tmp_path / "resume", {"different": True}, chunk_size=6
        )


def test_multiscale_uses_actual_frames_and_cache_retains_contained_windows(tmp_path):
    from ai_video_annotator.datasets.base import VideoMetadata

    metadata = VideoMetadata("v", tmp_path / "unused", 8, 16, 8, 8)
    bank = np.array([[1, i / 16] for i in range(16)], np.float32)
    windows = generate_multiscale_windows("v", 2, [0.25, 0.5, 1])
    pooled = pool_frame_bank(bank, metadata, windows)
    assert len(set(w.interval for w in windows)) == len(windows)
    assert any(
        a.interval.end_seconds > b.interval.end_seconds
        for a, b in zip(windows, windows[1:], strict=False)
    )
    for row, window in zip(pooled, windows, strict=True):
        indices = sample_frame_indices(window.interval, 8, 16, 16)
        np.testing.assert_array_equal(row, pool_frame_features(bank[indices]))
    save_cache(tmp_path / "cache", pooled, windows, {})
    assert load_cache(tmp_path / "cache").windows == windows


def test_sparse_indices_exactly_match_reference_nms_matching_and_metrics():
    from ai_video_annotator.evaluation.benchmark import budget_metrics
    from ai_video_annotator.evaluation.matching import match_predictions
    from ai_video_annotator.evaluation.metrics import retrieval_metrics
    from ai_video_annotator.evaluation.temporal_index import (
        EventIndex,
        SuppressionIndex,
        metrics_from_hits,
    )
    from ai_video_annotator.retrieval.postprocess import temporal_nms
    from ai_video_annotator.types import RetrievalResult

    rng = np.random.default_rng(42)
    windows = generate_multiscale_windows("v", 5, [0.125, 0.25, 0.5, 1])
    truth = [
        BehaviorEvent("v", "target", TimeInterval(float(a), float(a + b)))
        for a, b in zip(rng.uniform(0, 4, 20), rng.uniform(0.05, 1, 20), strict=True)
    ]
    truth += [truth[0], BehaviorEvent("other-video", "target", TimeInterval(0, 5))]
    for nms in (0, 0.3, 1):
        suppression = SuppressionIndex(windows, nms)
        index = EventIndex(windows, truth)
        for scores in (rng.normal(size=len(windows)), np.zeros(len(windows))):
            candidates = rng.random(len(windows)) > 0.2
            ranked = suppression.rank(scores, candidates)
            predictions = [RetrievalResult(windows[i], float(scores[i])) for i in ranked]
            reference = temporal_nms(
                [
                    RetrievalResult(w, float(s))
                    for w, s, c in zip(windows, scores, candidates, strict=True)
                    if c
                ],
                nms,
            )
            assert tuple(p.window.index for p in reference) == tuple(ranked)
            for threshold in (0.3, 0.5, 1):
                hits, matches = index.match(ranked, threshold)
                reference_hits, reference_matches = match_predictions(predictions, truth, threshold)
                np.testing.assert_array_equal(hits, reference_hits)
                assert tuple(matches) == reference_matches
                metrics = metrics_from_hits(hits, len(truth))
                for k, value in retrieval_metrics(predictions, truth, threshold).items():
                    assert metrics[k] == value
                for budget in (10, 20, 50):
                    assert metrics["budgets"][str(budget)] == budget_metrics(
                        predictions, truth, threshold, budget
                    )
