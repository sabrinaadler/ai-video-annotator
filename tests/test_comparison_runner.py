import json
import runpy
import sys
from types import SimpleNamespace


def test_subset_rerun_preserves_other_model_and_dataset_reports(tmp_path, monkeypatch):
    runner = runpy.run_path("scripts/compare_encoders.py")
    manifest = {"recordings": [{"id": "fly/movie2/clip", "cache": "original"}]}
    manifest_path = tmp_path / "fly.json"
    manifest_path.write_text(json.dumps(manifest))
    output = tmp_path / "results"
    output.mkdir()
    retained = [
        {"dataset": "CalMS21", "encoder": "clip_mean", "report": "mouse-result"},
        {"dataset": "Fly-vs-Fly", "encoder": "videomae", "report": "baseline"},
        {"dataset": "Fly-vs-Fly", "encoder": "bioclip2_mean", "report": "other-model"},
    ]
    (output / "inventory.json").write_text(
        json.dumps([*retained, {"dataset": "Fly-vs-Fly", "encoder": "clip_mean", "report": "old"}])
    )
    specific = {
        "recordings": [
            {
                "id": "fly/movie2/clip",
                "cache": "outputs/embeddings/comparison/fly_movie2_clip_mean",
            }
        ]
    }
    (output / "flyvsfly_clip_mean.json").write_text(
        json.dumps({"manifest": specific, "seed_trials": 30})
    )
    monkeypatch.setitem(
        runner["main"].__globals__, "load_cache", lambda _: SimpleNamespace(windows=(), metadata={})
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_encoders.py",
            "--datasets",
            "flyvsfly",
            "--fly-manifest",
            str(manifest_path),
            "--output",
            str(output),
            "--variants",
            "clip_mean",
        ],
    )
    runner["main"]()
    inventory = json.loads((output / "inventory.json").read_text())
    assert inventory[:3] == retained
    assert len(inventory) == 4
    assert inventory[3]["report"] == str(output / "flyvsfly_clip_mean.json")
