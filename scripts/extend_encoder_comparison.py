"""Encode and score additional frozen models against the existing paired pilot.

Defaults to four insect excerpts; --datasets all includes the longer mouse pilot.
One encoder process at a time bounds memory. Existing valid caches are reused.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

MODELS = {
    "clip": (
        "laion/CLIP-ViT-B-16-laion2B-s34B-b88K",
        "7288da5a0d6f0b51c4a2b27c624837a9236d0112",
    ),
    "bioclip2": ("imageomics/bioclip-2", "2957b322090f9cb17ae72c71981c7218a28d81e0"),
    "resnet50": ("torchvision/resnet50", "IMAGENET1K_V2"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", choices=["flyvsfly", "calms21", "all"], default="flyvsfly")
    parser.add_argument("--models", nargs="+", choices=MODELS, default=["clip", "resnet50"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--frame-batch-size", type=int, default=8)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    env = {
        **os.environ,
        "HF_HOME": "outputs/model-cache",
        "PYTHONPATH": "src",
        "TORCH_HOME": "outputs/model-cache/torch",
    }

    def run(arguments):
        subprocess.run([sys.executable, *arguments], check=True, env=env)

    for dataset in ("flyvsfly", "calms21"):
        if args.datasets not in {"all", dataset}:
            continue
        fly = dataset == "flyvsfly"
        manifest_file = (
            "outputs/benchmarks/flyvsfly_manifest.json"
            if fly
            else "outputs/benchmarks/calms21_pilot_manifest.json"
        )
        manifest = json.loads(Path(manifest_file).read_text())
        for model in dict.fromkeys(args.models):
            repository, revision = MODELS[model]
            for record in manifest["recordings"]:
                name = record["id"].split("/")[-2] if fly else Path(record["id"]).name
                prefix = "fly_" + name if fly else name
                print(f"Encoding {dataset} / {name} / {model}", flush=True)
                arguments = [
                    "scripts/encode_image_video.py",
                    "--video",
                    record["video"],
                    "--video-id",
                    record["id"],
                    "--backend",
                    "resnet50" if model == "resnet50" else "openclip",
                    "--model",
                    repository,
                    "--revision",
                    revision,
                    "--cache-prefix",
                    f"outputs/embeddings/comparison/{prefix}_{model}",
                    "--device",
                    args.device,
                    "--frame-batch-size",
                    str(args.frame_batch_size),
                    "--window-seconds",
                    ".25" if fly else "2",
                    "--stride-seconds",
                    ".125" if fly else ".5",
                ]
                if args.local_files_only:
                    arguments.append("--local-files-only")
                run(arguments)
            run(
                [
                    "scripts/compare_encoders.py",
                    "--datasets",
                    dataset,
                    "--variants",
                    model + "_mean",
                    model + "_ordered4",
                ]
            )
    run(["scripts/report_encoder_comparison.py"])
    if args.datasets in {"flyvsfly", "all"} and {"clip", "resnet50"}.issubset(args.models):
        run(["scripts/report_extended_encoders.py"])


if __name__ == "__main__":
    main()
