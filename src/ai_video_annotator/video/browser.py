"""Prepare a browser-decodable viewing copy without changing research inputs."""

import json
from pathlib import Path

import av

from ai_video_annotator.embeddings.cache import source_fingerprint


def prepare_browser_video(source: Path, directory: Path) -> Path:
    """Preserve frame timestamps; cache H.264 derivatives by complete source hash."""
    fingerprint = source_fingerprint(source)
    with av.open(str(source)) as container:
        if container.streams.video[0].codec_context.name == "h264":
            return source
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{fingerprint['sha256']}.mp4"
    manifest = target.with_suffix(".json")
    if target.exists() and manifest.exists():
        saved = json.loads(manifest.read_text())
        if saved.get("source") == fingerprint and saved.get("viewing_copy") == source_fingerprint(
            target
        ):
            return target
    temporary = target.with_suffix(".partial.mp4")
    print(
        "Preparing H.264 viewing copy (original recording and embeddings stay intact)…", flush=True
    )
    try:
        with av.open(str(source)) as incoming, av.open(str(temporary), "w") as outgoing:
            original = incoming.streams.video[0]
            stream = outgoing.add_stream("libx264", rate=original.average_rate)
            stream.width, stream.height = original.width, original.height
            stream.pix_fmt = "yuv420p"
            stream.options = {"preset": "veryfast", "crf": "20"}
            stream.time_base = original.time_base
            count = 0
            for frame in incoming.decode(original):
                # Keep the original timeline rather than constructing new frame timestamps.
                for packet in stream.encode(frame):
                    outgoing.mux(packet)
                count += 1
            for packet in stream.encode():
                outgoing.mux(packet)
        temporary.replace(target)
        manifest.write_text(
            json.dumps(
                {
                    "source": fingerprint,
                    "viewing_copy": source_fingerprint(target),
                    "frame_count": count,
                },
                indent=2,
            )
        )
    finally:
        temporary.unlink(missing_ok=True)
    return target
