import pytest

from ai_video_annotator.datasets.event_csv import load_event_csv


def test_custom_species_overlapping_labels_and_quoted_names(tmp_path):
    path = tmp_path / "insects.csv"
    path.write_text(
        "\ufeffvideo_id,behavior,start_seconds,end_seconds,note\n"
        'fly01,"wing extension, male",1.2,2.5,independent labels\n'
        "fly01,circling,1.5,3.0,simultaneous behavior\n"
    )
    events = load_event_csv(path)
    assert events[0].video_id == "fly01"
    assert events[0].behavior == "wing extension, male"
    assert events[1].interval.start_seconds == 1.5
    assert len(events) == 2


@pytest.mark.parametrize(
    "rows", ["fly,mount,2,1", "fly,mount,nan,2", "fly,mount,1,2\nfly,mount,1,2"]
)
def test_invalid_or_duplicate_truth_rejected(tmp_path, rows):
    path = tmp_path / "labels.csv"
    path.write_text("video_id,behavior,start_seconds,end_seconds\n" + rows)
    with pytest.raises(ValueError, match="row"):
        load_event_csv(path)


def test_benchmark_cli_accepts_insect_csv_labels(tmp_path):
    import json
    import os
    import subprocess
    import sys

    import numpy as np

    from ai_video_annotator.embeddings.cache import save_cache, source_fingerprint
    from ai_video_annotator.types import TimeInterval, VideoWindow

    labels = tmp_path / "labels.csv"
    lines = ["video_id,behavior,start_seconds,end_seconds"]
    records = []
    for video_id in ("fly-source", "fly-target"):
        video = tmp_path / f"{video_id}.bin"
        video.write_bytes(video_id.encode())
        cache = tmp_path / video_id
        windows = tuple(VideoWindow(video_id, i, TimeInterval(i * 4, i * 4 + 2)) for i in range(8))
        vectors = np.array([[1, 0]] * 4 + [[0, 1]] * 4, dtype=np.float32)
        save_cache(cache, vectors, windows, {"source": source_fingerprint(video)})
        records.append(
            {"id": video_id, "video": str(video), "cache": str(cache), "events_csv": str(labels)}
        )
        lines.extend(f"{video_id},courtship,{i * 4},{i * 4 + 2}" for i in range(4))
    labels.write_text("\n".join(lines))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "recordings": records,
                "behaviors": ["courtship"],
                "pairs": [
                    {"source": "fly-source", "target": "fly-target", "protocol": "cross-video"}
                ],
            }
        )
    )
    output = tmp_path / "report.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/benchmark.py",
            "--manifest",
            str(manifest),
            "--output",
            str(output),
            "--shots",
            "1",
            "--trials",
            "1",
        ],
        env={**os.environ, "PYTHONPATH": "src"},
        capture_output=True,
        check=True,
    )
    report = json.loads(output.read_text())["reports"][0]
    assert report["source_event_count"] == report["target_event_count"] == 4
    assert all(r["average_precision"] == 1 for r in report["trials"] if r["method"] != "random")
