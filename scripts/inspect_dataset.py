"""Inspect one or all CalMS21 Task 1 recordings from local files."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from ai_video_annotator.datasets import CalMS21Dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--videos-root", required=True, type=Path)
    parser.add_argument("--video-id", help="Full sequence ID; omit to inspect all recordings")
    parser.add_argument("--behavior", help="Only show intervals for this behavior")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    args = parser.parse_args()
    try:
        dataset = CalMS21Dataset(args.annotations, args.videos_root)
        ids = (args.video_id,) if args.video_id else dataset.video_ids()
        results = []
        for video_id in ids:
            metadata = dataset.video_metadata(video_id)
            events = dataset.behavior_events(video_id, args.behavior)
            results.append(
                {
                    "video": {
                        **asdict(metadata),
                        "path": str(metadata.path),
                        "duration_seconds": metadata.duration_seconds,
                    },
                    "events": [asdict(event) for event in events],
                }
            )
            if not args.json:
                print(f"Video: {video_id}\nFPS: {metadata.fps:g}")
                print(f"Duration: {metadata.duration_seconds:.2f} s")
                for behavior in sorted({event.behavior for event in events}):
                    subset = [event for event in events if event.behavior == behavior]
                    print(f"\n{behavior}: {len(subset)} events")
                    for event in subset:
                        print(
                            f"  {event.interval.start_seconds:.2f}–"
                            f"{event.interval.end_seconds:.2f} s"
                        )
                print()
        if args.json:
            print(json.dumps(results, indent=2))
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
