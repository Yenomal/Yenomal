from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
from typing import Mapping

import torch
from torch import nn

from common.vision_stacks.autogaze_siglip import (
    AutoGazeSiglipConfig,
    AutoGazeSiglipVisionStack,
    load_config,
)


class CommonVisualPrefixAdapter(nn.Module):
    """Bridge `pi05` observations into the reusable AutoGaze+SigLIP visual stack."""

    CURRENT_VIEW_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")

    def __init__(self, out_dim: int, history_len: int):
        super().__init__()
        config_path = Path(
            os.environ.get(
                "MAEVLA_AUTOGAZE_STACK_CONFIG",
                Path(__file__).resolve().parents[1] / "simvla_autogaze" / "config" / "autogaze_siglip_simvla.yaml",
            )
        )
        base_config = load_config(config_path)
        stack_config = replace(base_config.stack, history_len=int(history_len))
        adapter_config = replace(base_config.adapter, output_dim=int(out_dim))

        self.stack = AutoGazeSiglipVisionStack(
            AutoGazeSiglipConfig(
                stack=stack_config,
                adapter=adapter_config,
            )
        )
        self.history_len = int(history_len)
        self.register_buffer(
            "input_mean",
            torch.tensor(stack_config.input_mean, dtype=torch.float32).view(1, 1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "input_std",
            torch.tensor(stack_config.input_std, dtype=torch.float32).view(1, 1, 3, 1, 1),
            persistent=False,
        )

    def _ensure_channels_first_5d(self, tensor: torch.Tensor, *, name: str) -> torch.Tensor:
        if tensor.ndim != 5:
            raise ValueError(f"{name} must have shape [B, T, C, H, W] or [B, T, H, W, C], got {tuple(tensor.shape)}")
        if tensor.shape[2] == 3:
            return tensor
        if tensor.shape[-1] == 3:
            return tensor.permute(0, 1, 4, 2, 3)
        raise ValueError(f"{name} must contain an RGB channel dimension, got {tuple(tensor.shape)}")

    def _mask_current_views(
        self,
        current_views: torch.Tensor,
        image_masks: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        masked_views = []
        for view_idx, key in enumerate(self.CURRENT_VIEW_KEYS):
            view = current_views[:, view_idx]
            mask = image_masks.get(key)
            if mask is None:
                masked_views.append(view)
                continue
            mask = mask.to(device=view.device, dtype=torch.bool).view(-1, 1, 1, 1)
            masked_views.append(torch.where(mask, view, torch.full_like(view, -1.0)))
        return torch.stack(masked_views, dim=1)

    def _prepare_history(self, images: Mapping[str, torch.Tensor], head_history: torch.Tensor | None) -> torch.Tensor:
        if head_history is None:
            base_view = images["base_0_rgb"]
            return base_view[:, None].expand(-1, self.history_len, -1, -1, -1)

        history = self._ensure_channels_first_5d(head_history, name="head_history")
        if history.shape[1] < self.history_len:
            pad = history[:, :1].expand(-1, self.history_len - history.shape[1], -1, -1, -1)
            history = torch.cat([pad, history], dim=1)
        elif history.shape[1] > self.history_len:
            history = history[:, -self.history_len :]
        return history

    def _convert_pi_pixels_for_common(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor.to(dtype=torch.float32)
        tensor = (tensor.clamp(-1.0, 1.0) + 1.0) * 0.5
        mean = self.input_mean.to(device=tensor.device, dtype=tensor.dtype)
        std = self.input_std.to(device=tensor.device, dtype=tensor.dtype)
        return (tensor - mean) / std

    def forward(
        self,
        images: Mapping[str, torch.Tensor],
        image_masks: Mapping[str, torch.Tensor],
        head_history: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        missing_keys = [key for key in self.CURRENT_VIEW_KEYS if key not in images]
        if missing_keys:
            raise KeyError(f"Missing required current views for common visual prefix: {missing_keys}")

        current_views = torch.stack([images[key] for key in self.CURRENT_VIEW_KEYS], dim=1)
        current_views = self._mask_current_views(current_views, image_masks)
        history_frames = self._prepare_history(images, head_history)

        history_frames = self._convert_pi_pixels_for_common(history_frames)
        current_views = self._convert_pi_pixels_for_common(current_views)

        outputs = self.stack(history_frames=history_frames, current_views=current_views)
        visual_tokens = outputs["tokens"]
        visual_pad_mask = outputs["attention_mask"].to(dtype=torch.bool)
        return visual_tokens, visual_pad_mask
