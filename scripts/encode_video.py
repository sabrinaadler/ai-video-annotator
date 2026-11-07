"""Encode a local constant-frame-rate video once, then reuse its embeddings."""

import argparse
import importlib.metadata
import time
from pathlib import Path

from ai_video_annotator.datasets.calms21 import probe_video
from ai_video_annotator.embeddings import load_cache, save_cache, source_fingerprint
from ai_video_annotator.embeddings.encoding import encode_sampled_windows
from ai_video_annotator.encoders import VideoMAEEncoder
from ai_video_annotator.video import generate_windows, iter_sampled_windows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--model", default="MCG-NJU/videomae-base")
    parser.add_argument("--revision", default="main", help="Use a commit hash for reproducibility")
    parser.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    parser.add_argument("--window-seconds", type=float, default=2.0)
    parser.add_argument("--stride-seconds", type=float, default=0.5)
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--preprocessing", choices=["letterbox", "center-crop"], default="letterbox"
    )
    args = parser.parse_args()
    try:
        if args.batch_size <= 0 or args.num_frames <= 0:
            raise ValueError("batch-size and num-frames must be positive")
        # Validate window settings even when reusing a cache.
        generate_windows(args.video_id, 1.0, args.window_seconds, args.stride_seconds)
        print("Checking source video fingerprint...", flush=True)
        identity = {
            "source": source_fingerprint(args.video),
            "video_id": args.video_id,
            "model": args.model,
            "requested_revision": args.revision,
            "window_seconds": args.window_seconds,
            "stride_seconds": args.stride_seconds,
            "num_frames": args.num_frames,
            "pooling": "mean_all_tokens",
            "preprocessing": args.preprocessing + "-v1",
            "sampling": "uniform_inclusive_frame_indices_v1",
            "transformers_version": importlib.metadata.version("transformers"),
        }
        if args.cache.exists():
            cached = load_cache(args.cache, identity)
            print(
                f"Reused cache: {len(cached.windows)} windows, "
                f"{cached.embeddings.shape[1]} dimensions. Model was not loaded."
            )
            return
        print("Inspecting video and loading frozen VideoMAE...", flush=True)
        metadata = probe_video(args.video, args.video_id)
        windows = generate_windows(
            args.video_id, metadata.duration_seconds, args.window_seconds, args.stride_seconds
        )
        encoder = VideoMAEEncoder(
            args.model,
            args.device,
            args.revision,
            args.local_files_only,
            preprocessing=args.preprocessing,
        )
        if encoder.num_frames != args.num_frames:
            raise ValueError(f"Model requires {encoder.num_frames} frames per window")
        print(f"Encoding {len(windows)} windows on {encoder.device}...", flush=True)
        started = time.monotonic()

        def progress(count: int) -> None:
            if count % 100 == 0 or count == len(windows):
                print(
                    f"Encoded {count}/{len(windows)} windows in {time.monotonic() - started:.1f}s",
                    flush=True,
                )

        embeddings, encoded_windows = encode_sampled_windows(
            encoder,
            iter_sampled_windows(metadata, windows, args.num_frames),
            args.batch_size,
            progress=progress,
        )
        save_cache(
            args.cache,
            embeddings,
            encoded_windows,
            {
                **identity,
                "resolved_revision": encoder.revision,
                "fps": metadata.fps,
                "frame_count": metadata.frame_count,
                "duration_seconds": metadata.duration_seconds,
                "device": str(encoder.device),
            },
        )
        print(f"Saved {embeddings.shape} embeddings to {args.cache}")
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
