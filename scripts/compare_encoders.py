"""Run paired mouse/insect encoder comparisons with audited cache geometry."""

import argparse
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

from ai_video_annotator.embeddings.cache import load_cache

VARIANTS = ("videomae", "dinov2_mean", "dinov2_ordered4", "bioclip_mean", "bioclip_ordered4")
EXTENDED_VARIANTS = (
    "bioclip2_mean",
    "bioclip2_ordered4",
    "clip_mean",
    "clip_ordered4",
    "resnet50_mean",
    "resnet50_ordered4",
)


def verify_geometry(caches):
    baseline = next(iter(caches.values()))
    keys = (
        "source",
        "video_id",
        "window_seconds",
        "stride_seconds",
        "num_frames",
        "sampling",
        "fps",
        "frame_count",
        "duration_seconds",
        "preprocessing",
    )
    for name, cache in caches.items():
        if cache.windows != baseline.windows:
            raise ValueError(f"Window geometry mismatch for {name}")
        for key in keys:
            if cache.metadata.get(key) != baseline.metadata.get(key):
                raise ValueError(f"Sampling/source mismatch for {name}: {key}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mouse-manifest",
        type=Path,
        default=Path("outputs/benchmarks/calms21_pilot_manifest.json"),
    )
    parser.add_argument(
        "--fly-manifest", type=Path, default=Path("outputs/benchmarks/flyvsfly_manifest.json")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/benchmarks/encoder_comparison")
    )
    parser.add_argument("--datasets", choices=["all", "calms21", "flyvsfly"], default="all")
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument(
        "--variants", nargs="+", choices=VARIANTS + EXTENDED_VARIANTS, default=VARIANTS
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    existing = args.output / "inventory.json"
    inventory = json.loads(existing.read_text()) if existing.exists() else []
    selected_names = (
        {"CalMS21", "Fly-vs-Fly"}
        if args.datasets == "all"
        else ({"CalMS21"} if args.datasets == "calms21" else {"Fly-vs-Fly"})
    )
    inventory = [
        row
        for row in inventory
        if row["dataset"] not in selected_names or row["encoder"] not in args.variants
    ]
    for dataset, path in (("CalMS21", args.mouse_manifest), ("Fly-vs-Fly", args.fly_manifest)):
        if dataset not in selected_names:
            continue
        manifest = json.loads(path.read_text())
        paths = {}
        metadata = {}
        for record in manifest["recordings"]:
            fly = dataset == "Fly-vs-Fly"
            name = record["id"].split("/")[-2] if fly else Path(record["id"]).name
            prefix = "fly_" + name if fly else name
            selected = {
                variant: Path("outputs/embeddings/comparison") / f"{prefix}_{variant}"
                for variant in dict.fromkeys(("videomae", *args.variants))
            }
            if not fly:
                selected["videomae"] = Path(record["cache"])
            caches = {variant: load_cache(directory) for variant, directory in selected.items()}
            verify_geometry(caches)
            paths[record["id"]] = selected
            metadata[record["id"]] = {variant: cache.metadata for variant, cache in caches.items()}
        # All encoders use the same manifests, seed, nested examples and target truth.
        for variant in dict.fromkeys(args.variants):
            specific = deepcopy(manifest)
            for record in specific["recordings"]:
                record["cache"] = str(paths[record["id"]][variant])
            name = dataset.lower().replace("-", "").replace(" ", "_")
            manifest_file = args.output / f"{name}_{variant}_manifest.json"
            result_file = args.output / f"{name}_{variant}.json"
            manifest_file.write_text(json.dumps(specific, indent=2))
            if not result_file.exists():
                subprocess.run(
                    [
                        sys.executable,
                        "scripts/benchmark.py",
                        "--manifest",
                        str(manifest_file),
                        "--output",
                        str(result_file),
                        "--trials",
                        str(args.trials),
                        "--seed",
                        "42",
                    ],
                    check=True,
                )
            else:
                saved = json.loads(result_file.read_text())
                if saved["manifest"] != specific or saved.get("seed_trials") != args.trials:
                    raise ValueError(f"Existing comparison report mismatch: {result_file}")
            inventory.append(
                {
                    "dataset": dataset,
                    "encoder": variant,
                    "report": str(result_file),
                    "cache_metadata": metadata,
                }
            )
    (args.output / "inventory.json").write_text(json.dumps(inventory, indent=2))
    print("Paired encoder comparison complete.", flush=True)


if __name__ == "__main__":
    main()
