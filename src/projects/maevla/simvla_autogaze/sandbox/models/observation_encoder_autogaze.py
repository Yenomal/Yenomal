from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn

from common.vision_stacks.autogaze_siglip import (
    AdapterConfig,
    AutoGazeSiglipConfig,
    AutoGazeSiglipVisionStack,
    load_config,
)


class AutoGazeObservationEncoder(nn.Module):
    """Thin SimVLA-compatible wrapper around the reusable common vision stack."""

    def __init__(
        self,
        out_dim: int,
        autogaze_model_path: str = "nvidia/AutoGaze",
        siglip_model_path: str = "google/siglip2-base-patch16-224",
        history_len: int = 8,
        projector_hidden_dim: int = 1536,
        gazing_ratio: float = 0.10,
        task_loss_requirement: float | None = None,
    ):
        super().__init__()
        config_path = Path(
            os.environ.get(
                "MAEVLA_AUTOGAZE_STACK_CONFIG",
                Path(__file__).resolve().parents[1] / "config" / "autogaze_siglip_simvla.yaml",
            )
        )
        base_config = load_config(config_path)
        stack_config = replace(
            base_config.stack,
            autogaze_model_path=autogaze_model_path,
            siglip_model_path=siglip_model_path,
            history_len=int(history_len),
            gazing_ratio=float(gazing_ratio),
            task_loss_requirement=task_loss_requirement,
        )
        adapter_config = replace(
            base_config.adapter,
            output_dim=int(out_dim),
            projector_hidden_dim=int(projector_hidden_dim),
        )
        self.stack = AutoGazeSiglipVisionStack(
            AutoGazeSiglipConfig(
                stack=stack_config,
                adapter=adapter_config,
            )
        )

    def forward(
        self,
        image_input: torch.Tensor,
        image_mask: torch.Tensor,
        head_history: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        del image_mask  # Current SimVLA RMBench path always uses the first 3 current views.
        current_views = image_input[:, :3]
        outputs = self.stack(
            history_frames=head_history,
            current_views=current_views,
        )
        return {
            "obs_tokens": outputs["tokens"],
            "obs_attention_mask": outputs["attention_mask"],
            "gazing_info": outputs["metadata"]["gazing_info"],
        }
