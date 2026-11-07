"""Lock one previously unused full official test recording before reading its labels."""

import json
from pathlib import Path

from ai_video_annotator.datasets.flyvsfly import official_split, recording
from ai_video_annotator.embeddings.cache import source_fingerprint


def main():
    directory = Path("data/raw/flyvsfly/Aggression")
    output = Path("outputs/benchmarks/fresh_fly_validation_manifest.json")
    selection = {
        "date": "2026-10-18",
        "test_recording": "movie7",
        "extent": "entire recording",
        "selected_before_reading_labels": True,
        "rule": "first numerically ordered official test movie not in earlier pilot",
    }
    lock = output.with_name("fresh_fly_selection.json")
    if lock.exists() and json.loads(lock.read_text()) != selection:
        raise ValueError("Fresh selection is already locked to a different protocol")
    lock.write_text(json.dumps(selection, indent=2))
    original = json.loads(Path("outputs/benchmarks/flyvsfly_manifest.json").read_text())
    sources = [r for r in original["recordings"] if r["split"] == "train"]
    assert official_split(directory)["movie7"] == "test"
    assert not any("movie7" in r["id"] for r in original["recordings"])
    metadata, events, _ = recording(directory, "movie7")
    ignores = [e.interval for e in events if e.behavior == "unknown"]
    target = {
        "id": metadata.video_id,
        "split": "test",
        "video": str(metadata.path),
        "cache": "",
        "events": [
            {
                "behavior": e.behavior,
                "start": e.interval.start_seconds,
                "end": e.interval.end_seconds,
            }
            for e in events
            if e.behavior != "unknown"
            and not any(e.interval.intersection_seconds(i) > 0 for i in ignores)
        ],
        "ignore_intervals": [{"start": i.start_seconds, "end": i.end_seconds} for i in ignores],
        "source_provenance": source_fingerprint(metadata.path),
        "annotation_provenance": source_fingerprint(directory / "movie7/movie7_actions.mat"),
        "duration_seconds": metadata.duration_seconds,
    }
    manifest = {
        "dataset": "Fly-vs-Fly full-recording validation",
        "selection": selection,
        "behaviors": ["lunge"],
        "recordings": [*sources, target],
        "pairs": [
            {"source": s["id"], "target": target["id"], "protocol": "cross-video"} for s in sources
        ],
    }
    output.write_text(json.dumps(manifest, indent=2))
    print(f"Locked movie7: full {metadata.duration_seconds:.1f}s; no scoring performed", flush=True)


if __name__ == "__main__":
    main()
