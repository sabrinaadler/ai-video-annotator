"""Cache frozen OpenCLIP or DINO frame features with mean and ordered pooling."""

import argparse
import importlib.metadata
import time
from pathlib import Path

from ai_video_annotator.datasets.calms21 import probe_video
from ai_video_annotator.embeddings.cache import load_cache, save_cache, source_fingerprint
from ai_video_annotator.embeddings.image_encoding import encode_image_windows
from ai_video_annotator.encoders.image import ImageClipEncoder
from ai_video_annotator.video import generate_windows, iter_sampled_windows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--video-id", required=True)
    parser.add_argument(
        "--backend", choices=["bioclip", "dinov2", "openclip", "resnet50"], required=True
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--cache-prefix", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--window-seconds", type=float, default=2)
    parser.add_argument("--stride-seconds", type=float, default=0.5)
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--frame-batch-size", type=int, default=16)
    args = parser.parse_args()
    identity = {
        "source": source_fingerprint(args.video),
        "video_id": args.video_id,
        "model": args.model,
        "requested_revision": args.revision,
        "backend": args.backend,
        "window_seconds": args.window_seconds,
        "stride_seconds": args.stride_seconds,
        "num_frames": args.num_frames,
        "preprocessing": "letterbox-v1",
        "sampling": "uniform_inclusive_frame_indices_v1",
        "image_size": 224,
        "transformers_version": importlib.metadata.version("transformers"),
        "open_clip_version": importlib.metadata.version("open_clip_torch"),
    }
    paths = {name: Path(str(args.cache_prefix) + "_" + name) for name in ("mean", "ordered4")}
    if args.backend == "resnet50":
        identity["torchvision_version"] = importlib.metadata.version("torchvision")
    pooling = {name: f"frame_l2_temporal_{name}_clip_l2_v1" for name in paths}
    for name, path in paths.items():
        if path.exists():
            load_cache(path, {**identity, "pooling": pooling[name]})
    if all(path.exists() for path in paths.values()):
        print("Reused both image caches; model not loaded.", flush=True)
        return
    metadata = probe_video(args.video, args.video_id)
    windows = generate_windows(
        args.video_id, metadata.duration_seconds, args.window_seconds, args.stride_seconds
    )
    encoder = ImageClipEncoder(
        args.backend,
        args.model,
        args.revision,
        args.device,
        args.local_files_only,
        frame_batch_size=args.frame_batch_size,
    )
    if encoder.image_size != identity["image_size"]:
        raise ValueError("This matched comparison requires native 224-pixel image models")
    started = time.monotonic()

    def progress(count, frames):
        if count % 100 == 0 or count == len(windows):
            memory = ""
            if encoder.device.type == "mps":
                import torch

                memory = f"; MPS driver {torch.mps.driver_allocated_memory() / 2**30:.2f} GiB"
            print(
                f"{args.backend}: {count}/{len(windows)} windows; {frames} unique frame "
                f"encodings; {time.monotonic() - started:.1f}s{memory}",
                flush=True,
            )

    vectors, encoded, frame_count = encode_image_windows(
        encoder,
        metadata,
        iter_sampled_windows(metadata, windows, args.num_frames),
        num_frames=args.num_frames,
        progress=progress,
    )
    for name, path in paths.items():
        if not path.exists():
            save_cache(
                path,
                vectors[name],
                encoded,
                {
                    **identity,
                    "pooling": pooling[name],
                    "resolved_revision": encoder.revision,
                    "fps": metadata.fps,
                    "frame_count": metadata.frame_count,
                    "duration_seconds": metadata.duration_seconds,
                    "unique_frame_encodings": frame_count,
                    "frame_batch_size": encoder.frame_batch_size,
                    "weight_loading": encoder.weight_loading,
                    "encoding_seconds": time.monotonic() - started,
                    "device": str(encoder.device),
                },
            )
    print(f"Saved {args.backend} mean and ordered4 caches", flush=True)


if __name__ == "__main__":
    main()
