from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from vision.vendor.autogaze.models.autogaze import AutoGaze, AutoGazeConfig, AutoGazeImageProcessor
from vision.visual_gaze._utils import extract_size, load_local_pretrained_model, resolve_model_path


class AutoGazePolicy(nn.Module):
    """Video selection policy backed by AutoGaze."""

    def __init__(self, config: Any):
        super().__init__()
        self.config = config
        self.history_len = int(config.history_len)
        self.gazing_ratio = float(config.gazing_ratio)
        self.task_loss_requirement = config.task_loss_requirement

        if config.init_mode == "pretrained":
            model_dir = resolve_model_path(config.model_path)
            if model_dir is None:
                raise FileNotFoundError(f"Cannot resolve AutoGaze model path: {config.model_path}")
            self.model = load_local_pretrained_model(AutoGaze, model_dir)
            processor = AutoGazeImageProcessor.from_pretrained(str(model_dir))
            image_mean = tuple(float(v) for v in processor.image_mean)
            image_std = tuple(float(v) for v in processor.image_std)
            image_size = extract_size(processor.size)
        elif config.init_mode == "random":
            model_config = AutoGazeConfig.from_dict(dict(config.model_config))
            self.model = AutoGaze(model_config)
            image_mean = tuple(float(v) for v in config.image_mean)
            image_std = tuple(float(v) for v in config.image_std)
            image_size = int(config.image_size)
        else:
            raise ValueError(f"Unsupported AutoGaze init_mode: {config.init_mode}")

        self.scales = tuple(int(scale) for scale in str(self.model.config.scales).split("+"))
        self.register_buffer("image_mean", torch.tensor(image_mean, dtype=torch.float32).view(1, 1, 3, 1, 1), persistent=False)
        self.register_buffer("image_std", torch.tensor(image_std, dtype=torch.float32).view(1, 1, 3, 1, 1), persistent=False)
        self.image_size = int(image_size)

    def _prepare_pixels(self, unit_video: torch.Tensor) -> torch.Tensor:
        batch_size, num_frames, channels, _, _ = unit_video.shape
        video = unit_video.flatten(0, 1)
        video = F.interpolate(
            video,
            size=(self.image_size, self.image_size),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        video = video.view(batch_size, num_frames, channels, self.image_size, self.image_size)
        video = video * 2.0 - 1.0
        return (video - self.image_mean.to(video.device, video.dtype)) / self.image_std.to(video.device, video.dtype)

    def forward(
        self,
        unit_video: torch.Tensor,
        *,
        target_scales: tuple[int, ...],
        target_patch_size: int,
    ) -> dict[str, torch.Tensor]:
        if unit_video.dim() != 5:
            raise ValueError("unit_video must have shape [B, T, C, H, W]")
        pixels = self._prepare_pixels(unit_video)
        autogaze_kwargs: dict[str, Any] = {
            "gazing_ratio": self.gazing_ratio,
            "target_scales": list(target_scales),
            "target_patch_size": int(target_patch_size),
        }
        if self.task_loss_requirement is not None:
            autogaze_kwargs["task_loss_requirement"] = self.task_loss_requirement
        return self.model({"video": pixels}, generate_only=True, **autogaze_kwargs)
