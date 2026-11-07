import json

import pytest

from ai_video_annotator.datasets import CalMS21Dataset, VideoMetadata
from ai_video_annotator.datasets.calms21 import TASK1_VOCAB, labels_to_events, probe_video

VIDEO_ID = "task1/train/mouse001_task1_annotator1"


def test_contiguous_events_include_first_and_last_frames():
    events = labels_to_events(VIDEO_ID, [2, 2, 3, 2], 30, TASK1_VOCAB)
    assert [event.behavior for event in events] == ["mount", "other", "mount"]
    assert events[0].interval.start_seconds == 0
    assert events[0].interval.end_seconds == pytest.approx(2 / 30)
    assert events[-1].interval.end_seconds == pytest.approx(4 / 30)
    assert sum(event.interval.duration_seconds for event in events) == pytest.approx(4 / 30)


def test_empty_and_single_frame():
    assert labels_to_events(VIDEO_ID, [], 30, TASK1_VOCAB) == ()
    assert labels_to_events(VIDEO_ID, [0], 30, TASK1_VOCAB)[0].interval.end_seconds == 1 / 30


@pytest.mark.parametrize("fps", [0, -1, float("nan"), float("inf")])
def test_invalid_fps(fps):
    with pytest.raises(ValueError, match="fps"):
        labels_to_events(VIDEO_ID, [2], fps, TASK1_VOCAB)


@pytest.mark.parametrize("labels", [[4], [2.0], [[2]], [True]])
def test_invalid_labels(labels):
    with pytest.raises(ValueError):
        labels_to_events(VIDEO_ID, labels, 30, TASK1_VOCAB)


@pytest.mark.parametrize("vocab", [{}, {"mount": 2, "other": 2}, {"": 2}])
def test_invalid_vocab(vocab):
    with pytest.raises(ValueError, match="vocab"):
        labels_to_events(VIDEO_ID, [2], 30, vocab)


def make_dataset(tmp_path, *, labels=None, vocab=None, frame_count=4):
    annotations = tmp_path / "annotations.json"
    record = {"annotations": [2, 2, 3, 2] if labels is None else labels}
    if vocab is not None:
        record["metadata"] = {"vocab": vocab}
    annotations.write_text(json.dumps({"annotator-id_0": {VIDEO_ID: record}}))
    videos = tmp_path / "videos"
    videos.mkdir()
    (videos / "mouse001_task1_annotator1.mp4").touch()
    return CalMS21Dataset(
        annotations,
        videos,
        metadata_reader=lambda path, video_id: VideoMetadata(
            video_id, path, 30, frame_count, 1024, 570
        ),
    )


def test_official_nested_format_and_behavior_filter(tmp_path):
    dataset = make_dataset(tmp_path)
    assert dataset.video_ids() == (VIDEO_ID,)
    assert dataset.video_metadata(VIDEO_ID).duration_seconds == pytest.approx(4 / 30)
    assert len(dataset.behavior_events(VIDEO_ID, "mount")) == 2
    with pytest.raises(ValueError, match="Unknown behavior"):
        dataset.behavior_events(VIDEO_ID, "grooming")
    with pytest.raises(KeyError):
        dataset.video_metadata("missing")


def test_metadata_vocab_overrides_default(tmp_path):
    dataset = make_dataset(tmp_path, labels=[0, 0, 1, 1], vocab={"mount": 0, "other": 1})
    assert dataset.behavior_events(VIDEO_ID)[0].behavior == "mount"


def test_frame_alignment_is_required(tmp_path):
    dataset = make_dataset(tmp_path, frame_count=5)
    with pytest.raises(ValueError, match="Frame mismatch"):
        dataset.behavior_events(VIDEO_ID)


def test_missing_video(tmp_path):
    dataset = make_dataset(tmp_path)
    (tmp_path / "videos" / "mouse001_task1_annotator1.mp4").unlink()
    dataset = CalMS21Dataset(dataset.annotations_path, dataset.videos_root)
    with pytest.raises(FileNotFoundError, match="No MP4"):
        dataset.video_metadata(VIDEO_ID)


def test_ambiguous_video(tmp_path):
    dataset = make_dataset(tmp_path)
    subdir = dataset.videos_root / "duplicate"
    subdir.mkdir()
    (subdir / "mouse001_task1_annotator1.mp4").touch()
    dataset = CalMS21Dataset(dataset.annotations_path, dataset.videos_root)
    with pytest.raises(ValueError, match="Ambiguous"):
        dataset.video_metadata(VIDEO_ID)


def test_probe_real_video_and_cli(tmp_path):
    import os
    import subprocess
    import sys

    import av
    import numpy as np

    dataset = make_dataset(tmp_path)
    path = dataset.videos_root / "mouse001_task1_annotator1.mp4"
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=30)
        stream.width, stream.height = 32, 32
        stream.pix_fmt = "yuv420p"
        for _ in range(4):
            frame = av.VideoFrame.from_ndarray(
                np.zeros((32, 32, 3), dtype=np.uint8), format="rgb24"
            )
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    metadata = probe_video(path, VIDEO_ID)
    assert metadata.frame_count == 4
    assert metadata.fps == 30
    result = subprocess.run(
        [
            sys.executable,
            "scripts/inspect_dataset.py",
            "--annotations",
            str(dataset.annotations_path),
            "--videos-root",
            str(dataset.videos_root),
            "--video-id",
            VIDEO_ID,
            "--behavior",
            "mount",
            "--json",
        ],
        env={**os.environ, "PYTHONPATH": "src"},
        capture_output=True,
        text=True,
        check=True,
    )
    output = json.loads(result.stdout)
    assert len(output[0]["events"]) == 2
    assert output[0]["video"]["frame_count"] == 4


@pytest.mark.parametrize(
    "raw",
    [
        [],
        {},
        {"annotator": {VIDEO_ID: {}}},
        {"annotator": {"task2/train/mouse001": {"annotations": [2]}}},
    ],
)
def test_rejects_unsupported_or_missing_annotations(tmp_path, raw):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        CalMS21Dataset(path, tmp_path)


def test_streaming_adapter_discards_pose_arrays(tmp_path):
    dataset = make_dataset(tmp_path)
    raw = json.loads(dataset.annotations_path.read_text())
    raw["annotator-id_0"][VIDEO_ID]["keypoints"] = [[[1.2, 3.4]]]
    dataset.annotations_path.write_text(json.dumps(raw))
    reloaded = CalMS21Dataset(dataset.annotations_path, dataset.videos_root)
    assert "keypoints" not in reloaded._records[VIDEO_ID]
    assert reloaded._records[VIDEO_ID]["annotations"] == [2, 2, 3, 2]


def test_annotation_counts_do_not_require_video_decoding(tmp_path):
    annotations = tmp_path / "counts.json"
    annotations.write_text(
        json.dumps(
            {"annotator-id_0": {"task1/train/missing": {"annotations": [2, 2, 3, 2, 0, 0, 1]}}}
        )
    )
    dataset = CalMS21Dataset(annotations, tmp_path)
    assert dataset.annotation_event_counts()["task1/train/missing"] == {
        "attack": 1,
        "investigation": 1,
        "mount": 2,
        "other": 1,
    }
