"""Reproducibly prepare a small stratified CalMS21 pixel-video benchmark."""

import argparse
import json
import os
import random
import subprocess
import sys
import zipfile
from pathlib import Path

from ai_video_annotator.datasets.calms21 import CalMS21Dataset

MODEL_REVISION = "dc740ceda42fce44faed2ea03c6d447db72f6af9"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("data/raw/calms21"))
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/benchmarks/calms21_pilot_manifest.json")
    )
    parser.add_argument("--selection-seed", type=int, default=20261013)
    parser.add_argument("--encode", action="store_true")
    parser.add_argument("--device", choices=["mps", "cpu", "cuda"])
    args = parser.parse_args()
    records = []
    with zipfile.ZipFile(args.raw / "task1_videos_mp4.zip") as archive:
        for split in ("train", "test"):
            members = sorted(
                n for n in archive.namelist() if f"/{split}/" in n and n.endswith(".mp4")
            )
            for name in random.Random(args.selection_seed).sample(members, 2):
                if not (args.raw / name).exists():
                    archive.extract(name, args.raw)
                records.append(
                    {
                        "id": f"task1/{split}/{Path(name).stem}",
                        "split": split,
                        "video": str(args.raw / name),
                        "cache": f"outputs/embeddings/{Path(name).stem}_letterbox",
                        "selection": "uniform archive sample before inspecting labels",
                    }
                )
    baseline = "task1_videos_mp4/train/mouse054_task1_annotator1.mp4"
    if not (args.raw / baseline).exists():
        with zipfile.ZipFile(args.raw / "task1_videos_mp4.zip") as archive:
            archive.extract(baseline, args.raw)
    records.insert(
        0,
        {
            "id": "task1/train/mouse054_task1_annotator1",
            "split": "train",
            "video": str(args.raw / "task1_videos_mp4/train/mouse054_task1_annotator1.mp4"),
            "cache": "outputs/embeddings/mouse054_videomae_letterbox",
            "selection": "previously observed development recording",
        },
    )
    with zipfile.ZipFile(args.raw / "task1_classic_classification.zip") as archive:
        for split in ("train", "test"):
            name = f"task1_classic_classification/calms21_task1_{split}.json"
            if not (args.raw / name).exists():
                archive.extract(name, args.raw)
    for split in ("train", "test"):
        dataset = CalMS21Dataset(
            args.raw / f"task1_classic_classification/calms21_task1_{split}.json",
            args.raw / "task1_videos_mp4",
        )
        if split == "train":
            counts = dataset.annotation_event_counts()
            eligible = sorted(v for v, c in counts.items() if c["attack"] >= 11)
            if not eligible:
                raise ValueError("No attack-positive training source with >=11 bouts")
            video_id = random.Random(args.selection_seed).choice(eligible)
            name = f"task1_videos_mp4/train/{Path(video_id).name}.mp4"
            with zipfile.ZipFile(args.raw / "task1_videos_mp4.zip") as archive:
                if not (args.raw / name).exists():
                    archive.extract(name, args.raw)
            if not any(r["id"] == video_id for r in records):
                records.append(
                    {
                        "id": video_id,
                        "split": "train",
                        "video": str(args.raw / name),
                        "cache": f"outputs/embeddings/{Path(video_id).name}_letterbox",
                        "selection": "stratified train source: >=11 attack bouts",
                    }
                )
            dataset = CalMS21Dataset(
                args.raw / f"task1_classic_classification/calms21_task1_{split}.json",
                args.raw / "task1_videos_mp4",
            )
        for record in records:
            if record["split"] != split:
                continue
            if args.encode:
                command = [
                    sys.executable,
                    "scripts/encode_video.py",
                    "--video",
                    record["video"],
                    "--video-id",
                    record["id"],
                    "--cache",
                    record["cache"],
                    "--batch-size",
                    "4",
                    "--preprocessing",
                    "letterbox",
                    "--revision",
                    MODEL_REVISION,
                    "--local-files-only",
                ]
                if args.device:
                    command += ["--device", args.device]
                subprocess.run(
                    command, check=True, env={**os.environ, "HF_HOME": "outputs/model-cache"}
                )
            events = dataset.behavior_events(record["id"])
            record["events"] = [
                {
                    "behavior": e.behavior,
                    "start": e.interval.start_seconds,
                    "end": e.interval.end_seconds,
                }
                for e in events
            ]
            print(
                record["id"],
                {
                    b: sum(e.behavior == b for e in events)
                    for b in ("mount", "attack", "investigation")
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
                "dataset": "CalMS21",
                "selection_seed": args.selection_seed,
                "behaviors": ["mount", "attack", "investigation"],
                "recordings": records,
                "pairs": pairs,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
