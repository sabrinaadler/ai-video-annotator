"""Frozen image features with mean or temporal-bin pooling for video retrieval."""

import json
import zipfile
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from ai_video_annotator.device import select_device
from ai_video_annotator.encoders.preprocessing import letterbox_frame


def pool_frame_features(features: np.ndarray, bins: int = 1) -> np.ndarray:
    """Normalize frames, average within temporal bins, concatenate, normalize clip."""
    if features.ndim != 2 or len(features) < bins or bins < 1 or not np.isfinite(features).all():
        raise ValueError("Provide finite frame features and a valid temporal bin count")
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("Frame features must be nonzero")
    normalized = features / norms
    pooled = np.concatenate([part.mean(axis=0) for part in np.array_split(normalized, bins)])
    norm = np.linalg.norm(pooled)
    if norm <= 0:
        raise ValueError("Clip features cancel to zero")
    return (pooled / norm).astype(np.float32)


def load_openclip_visual(model_config: dict, checkpoint: Path):
    """Load a native CLIP visual tower without materializing unused text weights.

    Meta initialization avoids a second random copy of the large visual tower.
    Memory-mapped CPU storage can be reclaimed rather than swapped after GPU copy.
    Strict visual-key matching preserves the checkpoint's exact representation.
    """
    from open_clip.model import CLIP

    with torch.device("meta"):
        visual = CLIP(**model_config).visual
    state = torch.load(
        checkpoint, map_location="cpu", weights_only=True, mmap=zipfile.is_zipfile(checkpoint)
    )
    state = state.get("state_dict", state)
    visual_state = {
        key.removeprefix("visual."): value
        for key, value in state.items()
        if key.startswith("visual.")
    }
    visual.load_state_dict(visual_state, strict=True, assign=True)
    return visual


class ImageClipEncoder:
    def __init__(
        self,
        backend: str,
        model_name: str,
        revision: str,
        device="auto",
        local_files_only=False,
        frame_batch_size=16,
    ):
        self.device = select_device(device)
        self.backend = backend
        self.frame_batch_size = frame_batch_size
        self.weight_loading = "transformers"
        if backend in {"openclip", "resnet50"} and self.device.type == "mps":
            # The default MPS allocator can exceed physical RAM on a 16GB Mac.
            torch.mps.set_per_process_memory_fraction(0.5)
        if frame_batch_size < 1:
            raise ValueError("Frame batch size must be positive")
        if backend == "dinov2":
            from transformers import AutoImageProcessor, AutoModel

            self.processor = AutoImageProcessor.from_pretrained(
                model_name, revision=revision, local_files_only=local_files_only, use_fast=False
            )
            self.model = AutoModel.from_pretrained(
                model_name, revision=revision, local_files_only=local_files_only
            )
            self.revision = getattr(self.model.config, "_commit_hash", None) or revision
            self.embedding_dim = self.model.config.hidden_size
            self.image_size = 224
            self.mean = self.processor.image_mean
        elif backend == "resnet50":
            from torchvision.models import ResNet50_Weights, resnet50

            if model_name != "torchvision/resnet50" or revision != "IMAGENET1K_V2":
                raise ValueError("ResNet baseline requires pinned IMAGENET1K_V2 weights")
            weights = ResNet50_Weights.IMAGENET1K_V2
            filename = weights.url.rsplit("/", 1)[-1]
            if (
                local_files_only
                and not (Path(torch.hub.get_dir()) / "checkpoints" / filename).exists()
            ):
                raise FileNotFoundError("ResNet checkpoint is not cached locally")
            model = resnet50(weights=weights)
            model.fc = torch.nn.Identity()
            self.model = model
            self.image_size = 224
            self.embedding_dim = 2048
            self.mean, self.std = weights.transforms().mean, weights.transforms().std
            self.revision = revision + ":" + filename
            self.weight_loading = "torchvision_hash_checked_v1"
        elif backend in {"bioclip", "openclip"}:
            import open_clip
            from huggingface_hub import snapshot_download

            path = snapshot_download(
                model_name,
                revision=revision,
                local_files_only=local_files_only,
                allow_patterns=["open_clip_config.json", "open_clip_pytorch_model.bin"],
            )
            config = json.loads((Path(path) / "open_clip_config.json").read_text())
            if backend == "openclip":
                self.weight_loading = "visual_mmap_v1"
                self.model = load_openclip_visual(
                    config["model_cfg"], Path(path) / "open_clip_pytorch_model.bin"
                )
            else:
                self.weight_loading = "full_clip_v1"
                model, _, _ = open_clip.create_model_and_transforms(
                    f"local-dir:{path}", device="cpu", weights_only=True
                )
                self.model = model.visual
            self.image_size = config["model_cfg"]["vision_cfg"]["image_size"]
            self.mean = config["preprocess_cfg"]["mean"]
            self.std = config["preprocess_cfg"]["std"]
            self.embedding_dim = config["model_cfg"]["embed_dim"]
            self.revision = Path(path).name
        else:
            raise ValueError("Unknown image encoder")
        self.model.to(self.device).eval().requires_grad_(False)

    def encode(self, clips: Sequence[np.ndarray]) -> np.ndarray:
        """Generic VideoEncoder-compatible mean pooling (frame reuse is an optional fast path)."""
        if not clips:
            return np.empty((0, self.embedding_dim), dtype=np.float32)
        vectors = []
        for clip in clips:
            if clip.ndim != 4 or not len(clip):
                raise ValueError("Expected nonempty clips shaped frames × height × width × RGB")
            vectors.append(pool_frame_features(self.encode_frames(list(clip))))
        return np.stack(vectors).astype(np.float32)

    def encode_frames(self, frames: Sequence[np.ndarray]) -> np.ndarray:
        result = []
        fill = tuple(round(float(value) * 255) for value in self.mean)
        with torch.inference_mode():
            for start in range(0, len(frames), self.frame_batch_size):
                selected = frames[start : start + self.frame_batch_size]
                if any(f.ndim != 3 or f.shape[-1] != 3 or f.dtype != np.uint8 for f in selected):
                    raise ValueError("Expected uint8 RGB frames")
                fitted = [letterbox_frame(f, self.image_size, fill) for f in selected]
                if self.backend == "dinov2":
                    pixels = self.processor(
                        fitted, do_resize=False, do_center_crop=False, return_tensors="pt"
                    )["pixel_values"].to(self.device)
                    features = self.model(pixel_values=pixels).last_hidden_state[:, 0]
                else:
                    pixels = torch.from_numpy(np.stack(fitted)).permute(0, 3, 1, 2).float() / 255
                    mean = torch.tensor(self.mean).view(1, 3, 1, 1)
                    std = torch.tensor(self.std).view(1, 3, 1, 1)
                    features = self.model(((pixels - mean) / std).to(self.device))
                result.append(features.cpu().float().numpy())
        vectors = (
            np.concatenate(result) if result else np.empty((0, self.embedding_dim), np.float32)
        )
        if not np.isfinite(vectors).all():
            raise ValueError("Image encoder returned nonfinite features")
        if result and self.device.type == "mps":
            del features, pixels
            torch.mps.empty_cache()
        return vectors
