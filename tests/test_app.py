"""Integration checks for real retrieval, disk recovery and HTTP video seeking."""

import csv
import io
import json
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pytest

from ai_video_annotator.app.server import Workspace, make_server
from ai_video_annotator.embeddings.cache import save_cache, source_fingerprint
from ai_video_annotator.types import TimeInterval, VideoWindow


@pytest.fixture
def workspace(tmp_path):
    video = tmp_path / "video.mp4"
    video.write_bytes(bytes(range(256)) * 4)
    windows = tuple(VideoWindow("recording", i, TimeInterval(i * 2, i * 2 + 2)) for i in range(5))
    cache = tmp_path / "cache"
    save_cache(
        cache,
        np.array([[1, 0], [0.9, 0.1], [0.8, 0.2], [0, 1], [0.7, 0.3]], dtype=np.float32),
        windows,
        {
            "source": source_fingerprint(video),
            "fps": 30,
            "duration_seconds": 10,
            "model": "fixture",
            "preprocessing": "letterbox-v1",
        },
    )
    return Workspace(video, cache, tmp_path / "session.json")


def test_review_search_restart_and_export(workspace):
    state = workspace.snapshot()
    state["examples"] = [{"id": "example", "start": 0, "end": 1, "note": "seed"}]
    state = workspace.update(state)
    state = workspace.search(state["revision"])
    assert state["results"][0]["start"] == 2
    assert all(row["start"] >= 2 for row in state["results"])
    state["results"][0].update(status="accepted", start=2.1, end=3.8, note='=formula,"note"\nline')
    state["results"][1]["status"] = "later"
    state = workspace.update(state)
    recovered = Workspace(workspace.video, workspace.video.parent / "cache", workspace.session)
    assert recovered.snapshot() == state
    rows = list(csv.DictReader(io.StringIO(recovered.annotation_csv().decode())))
    assert len(rows) == 2
    assert rows[1]["start_seconds"] == "2.1"
    assert rows[1]["note"] == '\'=formula,"note"\nline'
    assert rows[1]["source"] == "reviewed_suggestion"
    assert state["results"][0]["original_start"] == 2
    with pytest.raises(RuntimeError):
        workspace.update({**state, "revision": state["revision"] - 1})


def test_validation_and_source_identity(workspace):
    state = workspace.snapshot()
    state["examples"] = [{"id": "bad", "start": 4, "end": 3}]
    with pytest.raises(ValueError):
        workspace.update(state)
    assert not workspace.session.exists()
    workspace.video.write_bytes(b"changed")
    with pytest.raises(ValueError, match="do not match"):
        Workspace(workspace.video, workspace.video.parent / "cache", workspace.session)


def test_counterexample_rerank_preserves_review_edits_notes_and_provenance(workspace):
    state = workspace.snapshot()
    state["examples"] = [{"id": "seed", "start": 0, "end": 1, "note": ""}]
    state = workspace.update(state)
    state = workspace.search(state["revision"])
    state["results"][0].update(status="accepted", start=2.1, end=3.8, note="keep boundary")
    state["results"][1]["status"] = "rejected"
    state["results"][2]["note"] = "unfinished observation"
    negative = state["results"][1]
    state["negative_examples"] = [
        {
            "id": "negative",
            "start": negative["start"],
            "end": negative["end"],
            "note": "not the target",
        }
    ]
    state["negative_weight"] = 0.5  # Explicit opt-in; pilot findings keep this off by default.
    state = workspace.update(state)
    retained = state["results"][:3]
    reranked = workspace.search(state["revision"])
    assert reranked["results"][:3] == retained
    assert len(reranked["results"]) == 4
    assert reranked["results"][3]["score_type"] == "contrastive"
    assert reranked["results"][3]["search_run"] != retained[0]["search_run"]
    assert len(reranked["search_history"]) == 2
    assert len(list(csv.DictReader(io.StringIO(workspace.annotation_csv().decode())))) == 2
    recovered = Workspace(workspace.video, workspace.video.parent / "cache", workspace.session)
    assert recovered.snapshot() == reranked
    reranked["results"][3]["score_type"] = "invented"
    assert workspace.update(reranked)["results"][3]["score_type"] == "contrastive"


def test_rerank_excludes_original_proposal_after_boundary_moves(workspace):
    state = workspace.snapshot()
    state["examples"] = [{"id": "seed", "start": 0, "end": 1}]
    state = workspace.update(state)
    state = workspace.search(state["revision"])
    original_id = state["results"][0]["id"]
    state["results"][0].update(status="accepted", start=6.1, end=7.9)
    state = workspace.update(state)
    state = workspace.search(state["revision"])
    assert len({r["id"] for r in state["results"]}) == len(state["results"])
    assert sum(r["id"] == original_id for r in state["results"]) == 1
    assert state["negative_examples"] == []  # Plain decisions are not inferred negative labels.
    state["negative_examples"] = [{"id": "bad", "start": 6, "end": 8}]
    with pytest.raises(ValueError, match="overlaps a positive"):
        workspace.update(state)


def test_duration_policy_is_opt_in_and_records_actual_reference_context(workspace, tmp_path):
    from ai_video_annotator.video.windows import generate_multiscale_windows

    windows = generate_multiscale_windows("recording", 10, [0.5, 1, 2])
    cache = tmp_path / "multiple"
    save_cache(
        cache,
        np.array([[1, w.interval.start_seconds / 10 + 0.1] for w in windows], np.float32),
        windows,
        {**workspace.cache.metadata, "window_policy": "multiscale", "window_sizes": [0.5, 1, 2]},
    )
    with pytest.raises(ValueError, match="reference-window"):
        Workspace(workspace.video, cache, tmp_path / "new.json")
    app = Workspace(workspace.video, cache, tmp_path / "new.json", 1)
    state = app.snapshot()
    assert state["negative_weight"] == 0
    state["examples"] = [{"id": "seed", "start": 0, "end": 0.4}]
    state = app.update(state)
    state = app.search(state["revision"])
    assert all(r["end"] - r["start"] == 1 for r in state["results"])
    state["proposal_policy"] = "examples"
    state = app.update(state)
    state = app.search(state["revision"])
    assert all(r["end"] - r["start"] == 0.5 for r in state["results"])
    assert state["search_history"][-1]["reference_window_seconds"] == 0.5
    assert state["search_history"][-1]["candidate_window_seconds"] == 0.5


def test_http_range_and_write_boundaries(workspace):
    server = make_server(workspace, 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urlopen(base + "/") as response:
            assert b"Fieldnote" in response.read()
        for header, expected in (
            ("bytes=10-19", bytes(range(10, 20))),
            ("bytes=-3", bytes([253, 254, 255])),
        ):
            with urlopen(Request(base + "/video", headers={"Range": header})) as response:
                assert response.status == 206
                assert response.read() == expected
        with pytest.raises(HTTPError) as failure:
            urlopen(Request(base + "/video", headers={"Range": "bytes=9999-"}))
        assert failure.value.code == 416
        for path in ("/../metadata.json", "/api/unknown"):
            with pytest.raises(HTTPError) as failure:
                urlopen(base + path)
            assert failure.value.code == 404
        request = Request(
            base + "/api/session",
            data=json.dumps(workspace.snapshot()).encode(),
            headers={"Content-Type": "application/json", "Origin": "https://elsewhere"},
        )
        with pytest.raises(HTTPError) as failure:
            urlopen(request)
        assert failure.value.code == 403
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_browser_copy_preserves_frames_and_timestamps(tmp_path):
    from fractions import Fraction

    import av

    from ai_video_annotator.video.browser import prepare_browser_video

    source = tmp_path / "legacy.mp4"
    with av.open(str(source), "w") as output:
        stream = output.add_stream("mpeg4", rate=30)
        stream.width = stream.height = 32
        stream.pix_fmt = "yuv420p"
        for index in range(12):
            frame = av.VideoFrame.from_ndarray(
                np.full((32, 32, 3), index * 15, dtype=np.uint8), format="rgb24"
            )
            frame.pts = index
            frame.time_base = Fraction(1, 30)
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    target = prepare_browser_video(source, tmp_path / "viewing")
    with av.open(str(source)) as original, av.open(str(target)) as viewing:
        assert viewing.streams.video[0].codec_context.name == "h264"
        times = [float(frame.pts * frame.time_base) for frame in original.decode(video=0)]
        copied_times = [float(frame.pts * frame.time_base) for frame in viewing.decode(video=0)]
        assert copied_times == times
    modified = target.stat().st_mtime_ns
    assert prepare_browser_video(source, tmp_path / "viewing") == target
    assert target.stat().st_mtime_ns == modified


def test_ethogram_switching_preserves_review_provenance_and_exports_all_behaviors(workspace):
    state = workspace.snapshot()
    assert state["behavior"] == "New behavior"
    state["ethogram"] = [
        {"name": "lunge", "type": "state", "key": "1", "description": "Forward thrust"},
        {"name": "groom", "type": "state", "key": "g", "description": "Cleaning"},
        {"name": "jump", "type": "point", "key": "2", "description": "Takeoff"},
    ]
    state["behavior"] = "lunge"
    state["examples"] = [{"id": "seed", "start": 0, "end": 1}]
    state = workspace.update(state)
    state = workspace.search(state["revision"])
    state["results"][0].update(status="accepted", note="verified")
    state = workspace.update(state)
    reviewed = json.loads(json.dumps(state))
    state = workspace.switch_behavior(state["revision"], "groom")
    assert state["results"] == state["examples"] == []
    state["examples"] = [{"id": "groom-seed", "start": 7, "end": 8}]
    state["manual_events"] = [
        {"id": "point", "behavior": "jump", "type": "point", "start": 5, "end": 5},
        {"id": "state", "behavior": "groom", "type": "state", "start": 8, "end": 9},
    ]
    state["open_events"] = {"lunge": 9.5}
    state = workspace.update(state)
    recovered = Workspace(workspace.video, workspace.video.parent / "cache", workspace.session)
    assert recovered.snapshot() == state
    state = recovered.switch_behavior(state["revision"], "lunge")
    assert state["results"] == reviewed["results"]
    assert state["search_history"] == reviewed["search_history"]
    assert state["examples"] == reviewed["examples"]
    assert state["open_events"] == {"lunge": 9.5}
    rows = list(csv.DictReader(io.StringIO(recovered.annotation_csv().decode())))
    assert len(rows) == 5
    assert {r["behavior"] for r in rows} == {"lunge", "groom", "jump"}
    assert next(r for r in rows if r["behavior"] == "jump")["event_type"] == "point"
    assert all(r["start_seconds"] != "9.5" for r in rows)  # Unclosed states are not exported.
    with pytest.raises(RuntimeError):
        recovered.switch_behavior(state["revision"] - 1, "groom")
    with pytest.raises(ValueError, match="cannot be removed"):
        state["ethogram"][0]["name"] = "renamed"
        recovered.update(state)


@pytest.mark.parametrize("key", ["a", "i", "r", "x", " ", "F1", "é"])
def test_ethogram_rejects_reserved_or_unsupported_shortcuts(workspace, key):
    state = workspace.snapshot()
    state["ethogram"][0]["key"] = key
    with pytest.raises(ValueError, match="keys"):
        workspace.update(state)


def test_point_event_validation_and_legacy_label_preservation(workspace):
    state = workspace.snapshot()
    for key in ("ethogram", "behavior_tracks", "manual_events", "open_events"):
        state.pop(key)
    state["behavior"] = "mount"
    workspace.session.write_text(json.dumps(state))
    recovered = Workspace(workspace.video, workspace.video.parent / "cache", workspace.session)
    state = recovered.snapshot()
    assert state["ethogram"][0]["name"] == "mount"  # Never silently relabel legacy annotations.
    state["ethogram"][0]["type"] = "point"
    state = recovered.update(state)
    with pytest.raises(ValueError, match="Point events"):
        recovered.search(state["revision"])
    state["manual_events"] = [
        {"id": "bad", "behavior": "mount", "type": "point", "start": 2, "end": 3}
    ]
    with pytest.raises(ValueError, match="event time"):
        recovered.update(state)
    state["manual_events"][0]["end"] = 2
    state = recovered.update(state)
    state["ethogram"][0]["type"] = "state"
    with pytest.raises(ValueError, match="cannot be removed"):
        recovered.update(state)


def test_manual_coding_is_not_resuggested_and_cannot_be_a_counterexample(workspace):
    state = workspace.snapshot()
    state["examples"] = [{"id": "seed", "start": 0, "end": 1}]
    state["manual_events"] = [
        {
            "id": "coded",
            "behavior": state["behavior"],
            "type": "state",
            "start": 2,
            "end": 4,
        }
    ]
    state = workspace.update(state)
    state = workspace.search(state["revision"])
    assert all(r["start"] >= 4 for r in state["results"])
    state["negative_examples"] = [{"id": "conflict", "start": 2, "end": 3}]
    with pytest.raises(ValueError, match="overlaps a positive"):
        workspace.update(state)


def test_http_behavior_switch_identity_and_duplicate_key_guards(workspace):
    state = workspace.snapshot()
    state["ethogram"].append({"name": "groom", "type": "state", "key": "g", "description": ""})
    state = workspace.update(state)
    server = make_server(workspace, 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"

    def post(path, payload):
        return urlopen(
            Request(
                base + path,
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
            )
        )

    try:
        with pytest.raises(HTTPError) as failure:
            post("/api/behavior", {"revision": state["revision"], "behavior": "groom"})
        assert failure.value.code == 409
        with post(
            "/api/behavior",
            {"revision": state["revision"], "behavior": "groom", "identity": state["identity"]},
        ) as response:
            switched = json.load(response)
        assert switched["behavior"] == "groom"
        switched["ethogram"][1]["key"] = "1"
        with pytest.raises(HTTPError) as failure:
            post("/api/session", switched)
        assert failure.value.code == 400
        switched["identity"] = {}
        with pytest.raises(HTTPError) as failure:
            post("/api/session", switched)
        assert failure.value.code == 409
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
