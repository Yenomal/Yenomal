from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class SiglipPatchTokenizer(nn.Module):
    """Convert images plus selection metadata into SigLIP patch tokens."""

    def __init__(
        self,
        embeddings: nn.Module,
        *,
        image_size: int,
        image_mean: tuple[float, float, float],
        image_std: tuple[float, float, float],
    ):
        super().__init__()
        self.embeddings = embeddings
        self.image_size = int(image_size)
        self.patch_size = int(embeddings.patch_size)
        self.scales = tuple(int(scale) for scale in str(embeddings.config.scales).split("+"))
        self.hidden_size = int(embeddings.config.hidden_size)
        self.num_tokens_each_group = sum((scale // self.patch_size) ** 2 for scale in self.scales)

        self.register_buffer(
            "image_mean",
            torch.tensor(image_mean, dtype=torch.float32).view(1, 1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.tensor(image_std, dtype=torch.float32).view(1, 1, 3, 1, 1),
            persistent=False,
        )

    def _prepare_pixels(self, unit_pixels: torch.Tensor) -> torch.Tensor:
        batch_size, num_frames, channels, _, _ = unit_pixels.shape
        pixels = unit_pixels.flatten(0, 1)
        pixels = F.interpolate(
            pixels,
            size=(self.image_size, self.image_size),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        pixels = pixels.view(batch_size, num_frames, channels, self.image_size, self.image_size)
        return (pixels - self.image_mean.to(pixels.device, pixels.dtype)) / self.image_std.to(
            pixels.device, pixels.dtype
        )

    def forward(self, unit_pixels: torch.Tensor, selection: dict[str, Any]) -> torch.Tensor:
        if unit_pixels.dim() != 5:
            raise ValueError("unit_pixels must have shape [B, T, C, H, W]")
        pixels = self._prepare_pixels(unit_pixels)
        return self.embeddings(pixels, gazing_info=selection)
