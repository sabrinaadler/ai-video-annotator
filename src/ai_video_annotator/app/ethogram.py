"""Project ethogram and behavior-specific retrieval state."""

from copy import deepcopy
from math import isfinite

TRACK_KEYS = (
    "examples",
    "results",
    "method",
    "negative_examples",
    "negative_weight",
    "proposal_policy",
    "search_history",
)
RESERVED_KEYS = set("aiordxnpl ?") | {"i", "o"}


def empty_track():
    return dict(
        examples=[],
        results=[],
        method="nearest-example",
        negative_examples=[],
        negative_weight=0,
        proposal_policy="fixed",
        search_history=[],
    )


def migrate(state):
    """Preserve existing labels verbatim; species cannot be inferred from a filename."""
    state.setdefault(
        "ethogram", [{"name": state["behavior"], "type": "state", "key": "1", "description": ""}]
    )
    state.setdefault("behavior_tracks", {})
    state.setdefault("manual_events", [])
    state.setdefault("open_events", {})
    return state


def track(state):
    return {key: deepcopy(state[key]) for key in TRACK_KEYS}


def validate_project(state, duration):
    entries = state.get("ethogram")
    if not isinstance(entries, list) or not 1 <= len(entries) <= 50:
        raise ValueError("Define between 1 and 50 behaviors")
    names, keys, types = set(), set(), {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Invalid behavior definition")
        name, key = entry.get("name"), entry.get("key", "")
        if (
            not isinstance(name, str)
            or not name.strip()
            or name != name.strip()
            or len(name) > 80
            or name in names
        ):
            raise ValueError("Behavior names must be unique, nonempty, and at most 80 characters")
        if (
            not isinstance(key, str)
            or len(key) > 1
            or key != key.lower()
            or (
                key
                and (not key.isascii() or not key.isalnum() or key in RESERVED_KEYS or key in keys)
            )
        ):
            raise ValueError("Use unique letter or number keys; playback/review keys are reserved")
        if entry.get("type") not in {"state", "point"}:
            raise ValueError("Behavior type must be state or point")
        if (
            not isinstance(entry.get("description", ""), str)
            or len(entry.get("description", "")) > 2000
        ):
            raise ValueError("Behavior definition is too long")
        names.add(name)
        types[name] = entry["type"]
        if key:
            keys.add(key)
    if state["behavior"] not in names:
        raise ValueError("Select a behavior in the ethogram")
    tracks = state.get("behavior_tracks")
    if not isinstance(tracks, dict) or not set(tracks) <= names or state["behavior"] in tracks:
        raise ValueError("Invalid behavior workspaces")
    events = state.get("manual_events")
    if not isinstance(events, list) or len(events) > 10000:
        raise ValueError("Maximum 10,000 manually coded events per session")
    ids = set()
    for row in events:
        if (
            not isinstance(row, dict)
            or row.get("behavior") not in names
            or row.get("type") != types[row["behavior"]]
            or not isinstance(row.get("id"), str)
            or row["id"] in ids
        ):
            raise ValueError("Invalid coded event")
        ids.add(row["id"])
        start, end = row.get("start"), row.get("end")
        if (
            not isinstance(start, int | float)
            or not isinstance(end, int | float)
            or not isfinite(start)
            or not isfinite(end)
            or not 0 <= start <= end <= duration
            or (row["type"] == "state" and start == end)
            or (row["type"] == "point" and start != end)
        ):
            raise ValueError("Invalid event time or duration")
        if not isinstance(row.get("note", ""), str) or len(row.get("note", "")) > 2000:
            raise ValueError("Event note is too long")
    opened = state.get("open_events")
    if not isinstance(opened, dict) or not set(opened) <= names:
        raise ValueError("Invalid open events")
    for name, start in opened.items():
        if (
            types[name] != "state"
            or not isinstance(start, int | float)
            or not isfinite(start)
            or not 0 <= start <= duration
        ):
            raise ValueError("Invalid state event start")
