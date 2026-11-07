"""Loopback-only annotation server; no external services or database required."""

import csv
import io
import json
import os
import re
import tempfile
import threading
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from math import isfinite
from pathlib import Path
from urllib.parse import urlsplit

from ai_video_annotator.app.ethogram import empty_track, migrate, track, validate_project
from ai_video_annotator.embeddings.cache import load_cache, source_fingerprint
from ai_video_annotator.retrieval.search import retrieve
from ai_video_annotator.types import BehaviorEvent, TimeInterval


class Workspace:
    def __init__(self, video: Path, cache: Path, session: Path, reference_window_seconds=None):
        self.video = video.resolve(strict=True)
        self.cache = load_cache(cache)
        if self.cache.metadata.get("source") != source_fingerprint(self.video):
            raise ValueError("Video contents do not match the embedding cache")
        self.media = self.video
        self.session = session
        self.lock = threading.RLock()
        self.duration = float(self.cache.metadata["duration_seconds"])
        if reference_window_seconds is not None and (
            not isfinite(reference_window_seconds) or reference_window_seconds <= 0
        ):
            raise ValueError("Reference context duration must be positive")
        self.reference_window_seconds = reference_window_seconds
        if (
            len(self.cache.metadata.get("window_sizes", [])) > 1
            and reference_window_seconds is None
        ):
            raise ValueError(
                "Multi-duration caches require --reference-window-seconds for a fixed baseline"
            )
        self.identity = {
            "video_id": self.cache.windows[0].video_id,
            "source": self.cache.metadata["source"],
            "model": self.cache.metadata["model"],
            "revision": self.cache.metadata.get("resolved_revision"),
            "preprocessing": self.cache.metadata.get("preprocessing"),
            "window_seconds": self.cache.metadata.get("window_seconds"),
            "stride_seconds": self.cache.metadata.get("stride_seconds"),
            "num_frames": self.cache.metadata.get("num_frames"),
        }
        if self.cache.metadata.get("window_policy") is not None:
            for key in ("window_policy", "window_sizes", "pooling", "backend"):
                self.identity[key] = self.cache.metadata.get(key)
        if reference_window_seconds is not None:
            self.identity["reference_window_seconds"] = reference_window_seconds
        self.state = {
            "version": 1,
            "identity": self.identity,
            "revision": 0,
            "behavior": "New behavior",
            "examples": [],
            "results": [],
            "method": "nearest-example",
            "position": 0,
            "negative_examples": [],
            "negative_weight": 0,
            "proposal_policy": "fixed",
            "search_history": [],
        }
        migrate(self.state)
        if session.exists():
            saved = json.loads(session.read_text())
            if saved.get("identity") != self.identity or saved.get("version") != 1:
                raise ValueError("Session belongs to a different recording/cache")
            saved.setdefault("negative_examples", [])
            saved.setdefault("negative_weight", 0)
            saved.setdefault("proposal_policy", "fixed")
            saved.setdefault("search_history", [])
            if saved["results"] and not saved["search_history"]:
                saved["search_history"] = [
                    {
                        "id": 0,
                        "method": saved["method"],
                        "examples": deepcopy(saved["examples"]),
                        "negative_examples": [],
                        "score_type": "cosine",
                        "reference_window_seconds": reference_window_seconds,
                    }
                ]
                for row in saved["results"]:
                    row.setdefault("score_type", "cosine")
                    row.setdefault("search_run", 0)
            migrate(saved)
            self.validate(saved)
            self.state = saved

    def snapshot(self):
        with self.lock:
            return deepcopy(self.state)

    def validate(self, state):
        validate_project(state, self.duration)
        for name, saved_track in state["behavior_tracks"].items():
            nested = {**state, **saved_track, "behavior": name, "behavior_tracks": {}}
            self.validate(nested)
        active_type = next(e["type"] for e in state["ethogram"] if e["name"] == state["behavior"])
        if active_type == "point" and (
            state["examples"] or state["results"] or state["negative_examples"]
        ):
            raise ValueError("Point events use manual coding, not duration reference examples")
        behavior = state.get("behavior")
        if not isinstance(behavior, str) or not behavior.strip() or len(behavior) > 80:
            raise ValueError("Provide a behavior name (1–80 characters)")
        if state.get("method") not in ("nearest-example", "prototype"):
            raise ValueError("Unknown retrieval method")
        if state.get("proposal_policy", "fixed") not in {"fixed", "all", "examples"}:
            raise ValueError("Unknown proposal policy")
        weight = state.get("negative_weight", 0.5)
        if (
            not isinstance(weight, int | float)
            or isinstance(weight, bool)
            or not isfinite(weight)
            or not 0 <= weight <= 1
        ):
            raise ValueError("Counterexample strength must be between 0 and 1")
        position = state.get("position", 0)
        if not isinstance(position, float | int) or not 0 <= position <= self.duration:
            raise ValueError("Playback position is outside the recording")
        for key, limit in (("examples", 10), ("negative_examples", 10), ("results", 100)):
            rows = state.get(key)
            if not isinstance(rows, list) or len(rows) > limit:
                raise ValueError(f"Invalid {key}")
            identifiers = set()
            for row in rows:
                if not isinstance(row, dict):
                    raise ValueError("Invalid interval record")
                interval = TimeInterval(row["start"], row["end"])
                if interval.end_seconds > self.duration:
                    raise ValueError("Interval exceeds recording duration")
                identifier = row.get("id")
                if not isinstance(identifier, str) or identifier in identifiers:
                    raise ValueError("Interval identifiers must be unique")
                identifiers.add(identifier)
                if len(str(row.get("note", ""))) > 2000:
                    raise ValueError("Note is too long")
                if key == "results" and row.get("status") not in (
                    "pending",
                    "accepted",
                    "rejected",
                    "later",
                ):
                    raise ValueError("Unknown review decision")
        positives = [
            *state["examples"],
            *[r for r in state["results"] if r["status"] == "accepted"],
            *[
                r
                for r in state["manual_events"]
                if r["behavior"] == state["behavior"] and r["type"] == "state"
            ],
        ]
        if any(
            max(a["start"], b["start"]) < min(a["end"], b["end"])
            for a in positives
            for b in state["negative_examples"]
        ):
            raise ValueError(
                "A counterexample overlaps a positive annotation; resolve the conflict"
            )

    def commit(self, state):
        """Write atomically before reporting a successful save."""
        self.validate(state)
        self.session.parent.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(dir=self.session.parent, prefix=".session-")
        try:
            with os.fdopen(descriptor, "w") as handle:
                json.dump(state, handle, indent=2, allow_nan=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(name, self.session)
        finally:
            if os.path.exists(name):
                os.unlink(name)
        self.state = state
        return deepcopy(state)

    def update(self, incoming):
        with self.lock:
            if incoming.get("identity") != self.identity:
                raise RuntimeError("Recording or cache changed; reload before editing")
            if incoming.get("revision") != self.state["revision"]:
                raise RuntimeError("Session changed in another tab; reload before editing")
            state = deepcopy(self.state)
            for key in ("behavior", "examples", "results", "method", "position"):
                state[key] = incoming[key]
            for key in (
                "negative_examples",
                "negative_weight",
                "proposal_policy",
                "ethogram",
                "manual_events",
                "open_events",
            ):
                state[key] = incoming.get(key, state[key])
            if state["behavior"] != self.state["behavior"]:
                active = self.state["behavior"]
                used = (
                    self.state["examples"]
                    or self.state["results"]
                    or self.state["negative_examples"]
                    or active in self.state["open_events"]
                    or any(e["behavior"] == active for e in self.state["manual_events"])
                )
                if used or state["behavior"] in {e["name"] for e in self.state["ethogram"]}:
                    raise ValueError("Switch behavior using the behavior selector")
            # Existing annotations retain their behavior name and event type.
            new_types = {e["name"]: e["type"] for e in state["ethogram"]}
            for entry in self.state["ethogram"]:
                name = entry["name"]
                saved_track = (
                    track(self.state)
                    if name == self.state["behavior"]
                    else state["behavior_tracks"].get(name, empty_track())
                )
                used = (
                    saved_track["examples"]
                    or saved_track["results"]
                    or saved_track["negative_examples"]
                    or name in self.state["open_events"]
                    or any(e["behavior"] == name for e in self.state["manual_events"])
                )
                if used and new_types.get(name) != entry["type"]:
                    raise ValueError(
                        "A behavior with annotations cannot be removed, renamed, or retyped"
                    )
                if not used and name not in new_types:
                    state["behavior_tracks"].pop(name, None)
            # Cache scores and original proposal intervals are server-owned provenance.
            old = {row["id"]: row for row in self.state["results"]}
            if {row["id"] for row in state["results"]} != set(old):
                raise ValueError("Search results can only be created by retrieval")
            for row in state["results"]:
                previous = old[row["id"]]
                for key in ("score", "original_start", "original_end", "score_type", "search_run"):
                    if key in previous:
                        row[key] = previous[key]
                    else:
                        row.pop(key, None)
            if self.state["results"] and (
                state["behavior"] != self.state["behavior"]
                or state["examples"] != self.state["examples"]
                or state["method"] != self.state["method"]
            ):
                raise ValueError(
                    "Existing search settings are locked; start a new session to change"
                )
            state["revision"] += 1
            return self.commit(state)

    def switch_behavior(self, revision, name):
        with self.lock:
            if revision != self.state["revision"]:
                raise RuntimeError("Session changed; reload before switching behavior")
            if name not in {e["name"] for e in self.state["ethogram"]}:
                raise ValueError("Unknown behavior")
            if name == self.state["behavior"]:
                return self.snapshot()
            state = deepcopy(self.state)
            state["behavior_tracks"][state["behavior"]] = track(state)
            state.update(state["behavior_tracks"].pop(name, empty_track()))
            state["behavior"] = name
            state["revision"] += 1
            return self.commit(state)

    def search(self, revision):
        with self.lock:
            if revision != self.state["revision"]:
                raise RuntimeError("Session changed; reload before searching")
            if (
                next(e for e in self.state["ethogram"] if e["name"] == self.state["behavior"])[
                    "type"
                ]
                == "point"
            ):
                raise ValueError(
                    "Point events use manual coding; similarity search needs duration examples"
                )
            examples = [
                BehaviorEvent(
                    self.identity["video_id"],
                    self.state["behavior"],
                    TimeInterval(row["start"], row["end"]),
                )
                for row in self.state["examples"]
            ]
            negatives = [
                BehaviorEvent(
                    self.identity["video_id"],
                    self.state["behavior"],
                    TimeInterval(row["start"], row["end"]),
                )
                for row in self.state["negative_examples"]
            ]
            reference = self.reference_window_seconds
            proposal_duration = reference
            if self.state["proposal_policy"] == "all":
                proposal_duration = None
            elif self.state["proposal_policy"] == "examples":
                import numpy as np

                available = self.cache.metadata.get("window_sizes") or [
                    w.interval.duration_seconds for w in self.cache.windows
                ]
                if not examples:
                    raise ValueError("Mark a positive reference before searching")
                median = float(np.median([e.interval.duration_seconds for e in examples]))
                reference = proposal_duration = min(
                    set(available), key=lambda d: (abs(np.log(d / median)), d)
                )
            retained = [
                deepcopy(row)
                for row in self.state["results"]
                if row["status"] != "pending"
                or row.get("note")
                or row["start"] != row["original_start"]
                or row["end"] != row["original_end"]
            ]
            if len(retained) >= 100:
                raise ValueError(
                    "Session has 100 reviewed intervals; export and start another session"
                )
            ranked = retrieve(
                self.cache.embeddings,
                self.cache.windows,
                examples,
                method=self.state["method"],
                top_k=min(20, 100 - len(retained)),
                negative_examples=negatives,
                negative_weight=self.state["negative_weight"],
                excluded_intervals=[
                    *[
                        TimeInterval(row[a], row[b])
                        for row in retained
                        for a, b in (("start", "end"), ("original_start", "original_end"))
                    ],
                    *[
                        TimeInterval(row["start"], row["end"])
                        for row in self.state["manual_events"]
                        if row["behavior"] == self.state["behavior"] and row["type"] == "state"
                    ],
                ],
                reference_window_seconds=reference,
                candidate_window_seconds=proposal_duration,
            )
            state = deepcopy(self.state)
            run = max((r["id"] for r in state["search_history"]), default=-1) + 1
            score_type = "contrastive" if negatives and state["negative_weight"] else "cosine"
            state["search_history"].append(
                {
                    "id": run,
                    "method": state["method"],
                    "examples": deepcopy(state["examples"]),
                    "negative_examples": deepcopy(state["negative_examples"]),
                    "negative_weight": state["negative_weight"],
                    "score_type": score_type,
                    "reference_window_seconds": reference,
                    "proposal_policy": self.state["proposal_policy"],
                    "candidate_window_seconds": proposal_duration,
                }
            )
            state["results"] = retained + [
                {
                    "id": f"window-{item.window.index}",
                    "start": item.window.interval.start_seconds,
                    "end": item.window.interval.end_seconds,
                    "original_start": item.window.interval.start_seconds,
                    "original_end": item.window.interval.end_seconds,
                    "score": item.score,
                    "status": "pending",
                    "note": "",
                    "score_type": score_type,
                    "search_run": run,
                }
                for item in ranked
            ]
            used_runs = {row.get("search_run") for row in state["results"]}
            state["search_history"] = [r for r in state["search_history"] if r["id"] in used_runs]
            state["revision"] += 1
            return self.commit(state)

    def annotation_csv(self):
        state = self.snapshot()
        output = io.StringIO(newline="")
        writer = csv.writer(output)
        writer.writerow(
            ["video_id", "behavior", "start_seconds", "end_seconds", "source", "note", "event_type"]
        )
        records = []
        for behavior, data in {**state["behavior_tracks"], state["behavior"]: track(state)}.items():
            for source, rows in (
                ("researcher_example", data["examples"]),
                ("reviewed_suggestion", data["results"]),
            ):
                for row in rows:
                    if source == "reviewed_suggestion" and row["status"] != "accepted":
                        continue
                    records.append((behavior, row, source, "state"))
        records.extend(
            (r["behavior"], r, "manual_coding", r["type"]) for r in state["manual_events"]
        )
        for behavior, row, source, event_type in sorted(records, key=lambda r: r[1]["start"]):
            values = [
                self.identity["video_id"],
                behavior,
                row["start"],
                row["end"],
                source,
                row.get("note", ""),
                event_type,
            ]
            writer.writerow(
                [
                    "'" + value
                    if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@"))
                    else value
                    for value in values
                ]
            )
        return output.getvalue().encode()


def make_server(workspace, port=8765):
    static = Path(__file__).parent / "static"

    class Handler(BaseHTTPRequestHandler):
        def send_bytes(self, body, kind, status=200, headers=None):
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            for key, value in (headers or {}).items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(body)

        def json(self, payload, status=200):
            self.send_bytes(
                json.dumps(payload, allow_nan=False).encode(), "application/json", status
            )

        def do_GET(self):
            path = urlsplit(self.path).path
            if path == "/api/session":
                self.json(
                    {
                        "state": workspace.snapshot(),
                        "duration": workspace.duration,
                        "fps": workspace.cache.metadata["fps"],
                        "filename": workspace.video.name,
                        "session_name": workspace.session.name,
                    }
                )
            elif path == "/api/export.json":
                self.send_bytes(
                    json.dumps(workspace.snapshot(), indent=2).encode(),
                    "application/json",
                    headers={"Content-Disposition": 'attachment; filename="session.json"'},
                )
            elif path == "/api/export.csv":
                self.send_bytes(
                    workspace.annotation_csv(),
                    "text/csv; charset=utf-8",
                    headers={"Content-Disposition": 'attachment; filename="annotations.csv"'},
                )
            elif path == "/video":
                self.video()
            elif path in ("/", "/app.js", "/style.css"):
                filename = {"/": "index.html", "/app.js": "app.js", "/style.css": "style.css"}[path]
                kind = {"/": "text/html", "/app.js": "text/javascript", "/style.css": "text/css"}[
                    path
                ]
                self.send_bytes((static / filename).read_bytes(), kind + "; charset=utf-8")
            else:
                self.json({"error": "Not found"}, 404)

        def do_POST(self):
            if self.headers.get("Host") not in (
                f"127.0.0.1:{self.server.server_port}",
                f"localhost:{self.server.server_port}",
            ):
                self.json({"error": "Local host required"}, 403)
                return
            origin = self.headers.get("Origin")
            if origin and origin != f"http://{self.headers.get('Host')}":
                self.json({"error": "Cross-origin writes are disabled"}, 403)
                return
            if self.headers.get("Content-Type") != "application/json":
                self.json({"error": "JSON required"}, 415)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 4 * 1024 * 1024:
                    raise ValueError("Session request must be between 1 byte and 4 MiB")
                payload = json.loads(self.rfile.read(length))
                path = urlsplit(self.path).path
                if path == "/api/session":
                    self.json(workspace.update(payload))
                elif path == "/api/behavior":
                    if payload.get("identity") != workspace.identity:
                        raise RuntimeError("Recording changed; reload before switching behavior")
                    self.json(workspace.switch_behavior(payload["revision"], payload["behavior"]))
                elif path == "/api/search":
                    if payload.get("identity") != workspace.identity:
                        raise RuntimeError("Recording or cache changed; reload before searching")
                    self.json(workspace.search(payload["revision"]))
                else:
                    self.json({"error": "Not found"}, 404)
            except RuntimeError as exc:
                self.json({"error": str(exc)}, 409)
            except (ValueError, KeyError, TypeError) as exc:
                self.json({"error": str(exc)}, 400)

        def video(self):
            size = workspace.media.stat().st_size
            start, end = 0, size - 1
            raw = self.headers.get("Range")
            if raw:
                match = re.fullmatch(r"bytes=(\d*)-(\d*)", raw)
                if not match or not any(match.groups()):
                    self.send_bytes(b"", "video/mp4", 416, {"Content-Range": f"bytes */{size}"})
                    return
                first, last = match.groups()
                if first:
                    start = int(first)
                    end = min(int(last), end) if last else end
                else:
                    start = max(0, size - int(last))
                if start > end or start >= size:
                    self.send_bytes(b"", "video/mp4", 416, {"Content-Range": f"bytes */{size}"})
                    return
            self.send_response(206 if raw else 200)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(end - start + 1))
            if raw:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            try:
                with workspace.media.open("rb") as handle:
                    handle.seek(start)
                    remaining = end - start + 1
                    while remaining:
                        block = handle.read(min(1024 * 1024, remaining))
                        if not block:
                            break
                        self.wfile.write(block)
                        remaining -= len(block)
            except (BrokenPipeError, ConnectionResetError):
                pass  # Browsers cancel ranges when seeking.

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)
