"""Dataset-independent event labels for custom species and experimental setups."""

import csv
from pathlib import Path

from ai_video_annotator.types import BehaviorEvent, TimeInterval


def load_event_csv(path: str | Path) -> tuple[BehaviorEvent, ...]:
    """Read independently annotated events; support overlapping behavior labels.

    Required columns: video_id, behavior, start_seconds, end_seconds. Additional
    columns are ignored. Timestamps are seconds in the original recording timeline.
    Duplicate event labels are rejected rather than inflating evaluation truth.
    """
    events = []
    seen = set()
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"video_id", "behavior", "start_seconds", "end_seconds"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Event CSV requires columns: {', '.join(sorted(required))}")
        for number, row in enumerate(reader, 2):
            try:
                event = BehaviorEvent(
                    row["video_id"],
                    row["behavior"],
                    TimeInterval(float(row["start_seconds"]), float(row["end_seconds"])),
                )
                if event in seen:
                    raise ValueError("Duplicate event annotation")
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Invalid event CSV row {number}: {exc}") from exc
            seen.add(event)
            events.append(event)
    return tuple(events)
