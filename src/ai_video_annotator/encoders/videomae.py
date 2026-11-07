"""Frozen VideoMAE representations via mean pooling of all temporal/spatial tokens."""

from collections.abc import Sequence

import numpy as np
import torch

from ai_video_annotator.device import select_device
from ai_video_annotator.encoders.preprocessing import letterbox_frame


class VideoMAEEncoder:
    def __init__(
        self,
        model_name: str = "MCG-NJU/videomae-base",
        device: str = "auto",
        revision: str = "main",
        local_files_only: bool = False,
        preprocessing: str = "letterbox",
    ) -> None:
        from transformers import VideoMAEImageProcessor, VideoMAEModel

        if preprocessing not in {"letterbox", "center-crop"}:
            raise ValueError("preprocessing must be letterbox or center-crop")
        self.preprocessing = preprocessing
        self.device = select_device(device)
        self.processor = VideoMAEImageProcessor.from_pretrained(
            model_name, revision=revision, local_files_only=local_files_only
        )
        self.model = VideoMAEModel.from_pretrained(
            model_name, revision=revision, local_files_only=local_files_only
        ).to(self.device)
        self.model.requires_grad_(False)
        self.model.eval()
        self.model_name = model_name
        self.revision = getattr(self.model.config, "_commit_hash", None) or revision

    @property
    def embedding_dim(self) -> int:
        return self.model.config.hidden_size

    @property
    def num_frames(self) -> int:
        return self.model.config.num_frames

    def encode(self, clips: Sequence[np.ndarray]) -> np.ndarray:
        if not clips:
            return np.empty((0, self.embedding_dim), dtype=np.float32)
        for clip in clips:
            if (
                clip.ndim != 4
                or clip.shape[0] != self.num_frames
                or clip.shape[-1] != 3
                or clip.dtype != np.uint8
                or clip.shape[1] == 0
                or clip.shape[2] == 0
            ):
                raise ValueError(
                    f"Expected uint8 RGB clips shaped ({self.num_frames}, height, width, 3)"
                )
        pixels = self.prepare_pixels(clips).to(self.device)
        with torch.inference_mode():
            tokens = self.model(pixel_values=pixels).last_hidden_state
            vectors = tokens.mean(dim=1)
        result = vectors.cpu().float().numpy()
        if not np.isfinite(result).all():
            raise ValueError("Encoder returned nonfinite embeddings")
        return result

    def prepare_pixels(self, clips: Sequence[np.ndarray]) -> torch.Tensor:
        """Apply full-frame fitting before checkpoint rescaling and normalization."""
        if self.preprocessing == "center-crop":
            return self.processor([list(clip) for clip in clips], return_tensors="pt")[
                "pixel_values"
            ]
        size = self.model.config.image_size
        if not isinstance(size, int):
            raise ValueError("Letterboxing currently requires a square model image size")
        fill = tuple(round(float(channel) * 255) for channel in self.processor.image_mean)
        videos = [[letterbox_frame(frame, size, fill) for frame in clip] for clip in clips]
        return self.processor(videos, do_resize=False, do_center_crop=False, return_tensors="pt")[
            "pixel_values"
        ]
