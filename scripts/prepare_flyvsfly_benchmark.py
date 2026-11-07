"""Prepare fixed Fly-vs-Fly excerpts without choosing regions by behavior labels."""

import argparse
import json
from pathlib import Path

import av

from ai_video_annotator.datasets.flyvsfly import official_split, recording
from ai_video_annotator.embeddings.cache import source_fingerprint


def extract_frames(source, target, start_frame, end_frame, fps):
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".partial.mp4")
    with av.open(str(source)) as incoming, av.open(str(temporary), "w") as outgoing:
        original = incoming.streams.video[0]
        stream = outgoing.add_stream("libx264", rate=original.average_rate)
        stream.width, stream.height = original.width, original.height
        stream.pix_fmt = "yuv420p"
        stream.options = {"preset": "veryfast", "crf": "0"}
        count = 0
        for index, frame in enumerate(incoming.decode(original)):
            if index >= end_frame:
                break
            if index < start_frame:
                continue
            frame.pts = count
            frame.time_base = 1 / original.average_rate
            for packet in stream.encode(frame):
                outgoing.mux(packet)
            count += 1
        for packet in stream.encode():
            outgoing.mux(packet)
    if count != end_frame - start_frame:
        temporary.unlink(missing_ok=True)
        raise ValueError("Excerpt frame count mismatch")
    temporary.replace(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("data/raw/flyvsfly/Aggression"))
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/benchmarks/flyvsfly_manifest.json")
    )
    args = parser.parse_args()
    split = official_split(args.directory)
    # Declared before inspecting labels: two sources and two targets, 600–720 seconds.
    names = ("movie2", "movie3", "movie6", "movie8")
    records = []
    for name in names:
        metadata, events, original_bouts = recording(args.directory, name)
        first, last = round(600 * metadata.fps), round(720 * metadata.fps)
        video_id = metadata.video_id + "/600-720"
        target = Path("data/processed/flyvsfly") / f"{name}_600_720.mp4"
        manifest_path = target.with_suffix(".json")
        provenance = {
            "original_source": source_fingerprint(metadata.path),
            "first_frame": first,
            "end_frame_exclusive": last,
            "source_fps": metadata.fps,
            "source_frame_count": metadata.frame_count,
            "annotations": source_fingerprint(args.directory / name / f"{name}_actions.mat"),
        }
        if target.exists() and manifest_path.exists():
            saved = json.loads(manifest_path.read_text())
            if any(saved.get(k) != v for k, v in provenance.items()) or saved.get(
                "excerpt_source"
            ) != source_fingerprint(target):
                raise ValueError("Existing excerpt/provenance mismatch")
        else:
            extract_frames(metadata.path, target, first, last, metadata.fps)
            manifest_path.write_text(
                json.dumps({**provenance, "excerpt_source": source_fingerprint(target)}, indent=2)
            )
        clipped, ignores = [], []
        for event in events:
            start, end = event.interval.start_seconds, event.interval.end_seconds
            if end <= 600 or start >= 720:
                continue
            interval = {"start": max(600, start) - 600, "end": min(720, end) - 600}
            if event.behavior == "unknown":
                ignores.append(interval)
            elif start >= 600 and end <= 720:
                clipped.append({"behavior": event.behavior, **interval})
            else:
                # Boundary-truncated bouts are neither complete truth nor valid negatives.
                ignores.append(interval)
        clipped = [
            e
            for e in clipped
            if not any(max(e["start"], i["start"]) < min(e["end"], i["end"]) for i in ignores)
        ]
        record = {
            "id": video_id,
            "split": split[name],
            "video": str(target),
            "events": clipped,
            "ignore_intervals": ignores,
            "original_provenance": provenance,
            "scene_policy": (
                "union overlapping actor bouts; raw third-column flags retained in source "
                "and treated as in authors visualizer"
            ),
            "cache": "",
            "source_annotation_bout_count": len(original_bouts),
        }
        records.append(record)
        print(
            name,
            split[name],
            {
                b: sum(e["behavior"] == b for e in clipped)
                for b in ["lunge", "wing_threat", "tussle"]
            },
            flush=True,
        )
    pairs = [{"source": r["id"], "target": r["id"], "protocol": "within-video"} for r in records]
    pairs += [
        {"source": s["id"], "target": t["id"], "protocol": "cross-video"}
        for s in records
        if s["split"] == "train"
        for t in records
        if t["split"] == "test"
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "dataset": "Fly-vs-Fly aggression",
                "selection": (
                    "fixed movie2/3 train and movie6/8 test, 600-720s; "
                    "chosen before event count inspection"
                ),
                "window_seconds": 0.25,
                "stride_seconds": 0.125,
                "behaviors": ["lunge", "wing_threat", "tussle"],
                "recordings": records,
                "pairs": pairs,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
