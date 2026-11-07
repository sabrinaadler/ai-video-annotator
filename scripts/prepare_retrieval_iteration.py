"""Build genuine single/multiple-duration CLIP proposals from resumable frame banks."""

import argparse
import importlib.metadata
import json
import time
from pathlib import Path

from ai_video_annotator.datasets.calms21 import probe_video
from ai_video_annotator.embeddings.cache import load_cache, save_cache, source_fingerprint
from ai_video_annotator.embeddings.frame_bank import encode_frame_bank, pool_frame_bank
from ai_video_annotator.encoders.image import ImageClipEncoder
from ai_video_annotator.video.windows import generate_multiscale_windows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=Path("outputs/benchmarks/flyvsfly_manifest.json")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/benchmarks/retrieval_iteration_manifest.json")
    )
    parser.add_argument("--durations", nargs="+", type=float, default=[0.125, 0.25, 0.5, 1])
    parser.add_argument("--baseline-duration", type=float, default=0.25)
    parser.add_argument("--stride-fraction", type=float, default=0.5)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    encoder = None
    for record in manifest["recordings"]:
        source = source_fingerprint(record["video"])
        prefix = Path(record["video"]).stem + "_" + source["sha256"][:12]
        paths = {
            policy: Path("outputs/embeddings/retrieval_iteration") / f"{prefix}_{policy}"
            for policy in ("single", "multiscale")
        }
        identity = {
            "source": source,
            "video_id": record["id"],
            "model": "laion/CLIP-ViT-B-16-laion2B-s34B-b88K",
            "resolved_revision": "7288da5a0d6f0b51c4a2b27c624837a9236d0112",
            "preprocessing": "letterbox-v1",
            "backend": "openclip",
            "num_frames": 16,
            "sampling": "uniform_inclusive_frame_indices_v1",
            "pooling": "frame_l2_temporal_mean_clip_l2_v1",
            "stride_fraction": args.stride_fraction,
            "open_clip_version": importlib.metadata.version("open_clip_torch"),
        }
        policies = {"single": [args.baseline_duration], "multiscale": sorted(set(args.durations))}
        for policy, path in paths.items():
            if path.exists():
                load_cache(path, {**identity, "window_sizes": policies[policy]})
        if not all(p.exists() for p in paths.values()):
            metadata = probe_video(record["video"], record["id"])
            if encoder is None:
                encoder = ImageClipEncoder(
                    "openclip",
                    identity["model"],
                    identity["resolved_revision"],
                    args.device,
                    local_files_only=True,
                    frame_batch_size=8,
                )
            started = time.monotonic()
            bank = encode_frame_bank(
                encoder,
                metadata,
                Path("outputs/frame_features/clip") / prefix,
                {
                    k: identity[k]
                    for k in (
                        "source",
                        "video_id",
                        "model",
                        "resolved_revision",
                        "preprocessing",
                        "backend",
                        "open_clip_version",
                    )
                }
                | {"bank_version": 1},
                progress=lambda count, total, record=record, started=started: print(
                    f"{record['id']}: {count}/{total} frame features; "
                    f"{time.monotonic() - started:.1f}s",
                    flush=True,
                ),
            )
            for policy, path in paths.items():
                if path.exists():
                    continue
                sizes = policies[policy]
                windows = generate_multiscale_windows(
                    record["id"], metadata.duration_seconds, sizes, args.stride_fraction
                )
                save_cache(
                    path,
                    pool_frame_bank(bank, metadata, windows),
                    windows,
                    {
                        **identity,
                        "window_sizes": sizes,
                        "window_policy": policy,
                        "window_seconds": sizes[0] if len(sizes) == 1 else None,
                        "stride_seconds": sizes[0] * args.stride_fraction
                        if len(sizes) == 1
                        else None,
                        "fps": metadata.fps,
                        "frame_count": metadata.frame_count,
                        "duration_seconds": metadata.duration_seconds,
                    },
                )
                print(f"Saved {policy}: {len(windows)} actual windows", flush=True)
        record["cache"] = str(paths["single"])
        record["multiscale_cache"] = str(paths["multiscale"])
    manifest["iteration_protocol"] = "source-negative-multiscale-v1"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
