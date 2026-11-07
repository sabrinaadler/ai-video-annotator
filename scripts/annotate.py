"""Launch Fieldnote on a verified local recording and embedding cache."""

import argparse
from pathlib import Path

from ai_video_annotator.app.server import Workspace, make_server
from ai_video_annotator.video.browser import prepare_browser_video


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--viewing-cache", type=Path, default=Path("outputs/viewing"))
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--reference-window-seconds", type=float)
    args = parser.parse_args()
    try:
        workspace = Workspace(args.video, args.cache, args.session, args.reference_window_seconds)
        workspace.media = prepare_browser_video(workspace.video, args.viewing_cache)
        server = make_server(workspace, args.port)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(f"Fieldnote: http://127.0.0.1:{server.server_port}", flush=True)
    print(f"Session: {args.session.resolve()}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
