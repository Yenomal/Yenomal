from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoImageProcessor

from .config import AutoGazeSiglipConfig


class ProjectionAdapter(nn.Module):
    """Project vision features to the consumer model dimension."""

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AutoGazeSiglipVisionStack(nn.Module):
    """Reusable AutoGaze + SigLIP stack with a configurable projection adapter."""

    def __init__(self, config: AutoGazeSiglipConfig):
        super().__init__()
        self.config = config
        self.repo_root = Path(__file__).resolve().parents[4]
        self.autogaze_repo_root = self._resolve_autogaze_repo_root(config.stack.autogaze_repo_root)

        AutoGaze, AutoGazeImageProcessor, SiglipVisionModel = self._load_third_party_modules()

        self.history_len = int(config.stack.history_len)
        self.gazing_ratio = float(config.stack.gazing_ratio)
        self.task_loss_requirement = config.stack.task_loss_requirement
        self.current_view_names = tuple(config.stack.current_view_names)
        self.output_dim = int(config.adapter.output_dim)
        self.add_source_embeddings = bool(config.adapter.add_source_embeddings)
        self.add_age_embeddings = bool(config.adapter.add_age_embeddings)

        self.autogaze = AutoGaze.from_pretrained(config.stack.autogaze_model_path)
        self.autogaze_processor = AutoGazeImageProcessor.from_pretrained(config.stack.autogaze_model_path)

        self.siglip_processor = AutoImageProcessor.from_pretrained(config.stack.siglip_model_path)
        self.siglip = SiglipVisionModel.from_pretrained(
            config.stack.siglip_model_path,
            scales=self.autogaze.config.scales,
            attn_implementation=config.stack.attn_implementation,
        )
        self.siglip.vision_model.embeddings.register_buffer(
            "position_ids",
            torch.arange(self.siglip.vision_model.embeddings.num_positions).expand((1, -1)),
            persistent=False,
        )

        self.patch_size = int(self.siglip.config.patch_size)
        self.scales = sorted(int(scale) for scale in str(self.siglip.config.scales).split("+"))
        self.num_tokens_each_group = sum((scale // self.patch_size) ** 2 for scale in self.scales)

        self.projector = ProjectionAdapter(
            in_dim=int(self.siglip.config.hidden_size),
            out_dim=self.output_dim,
            hidden_dim=int(config.adapter.projector_hidden_dim),
        )

        self.role_to_id = {"history": 0}
        for idx, view_name in enumerate(self.current_view_names, start=1):
            self.role_to_id[view_name] = idx

        if self.add_source_embeddings:
            self.source_embed = nn.Embedding(len(self.role_to_id), self.output_dim)
        else:
            self.source_embed = None

        if self.add_age_embeddings:
            self.age_embed = nn.Embedding(self.history_len + 1, self.output_dim)
        else:
            self.age_embed = None

        self.register_buffer(
            "input_mean",
            torch.tensor(config.stack.input_mean, dtype=torch.float32).view(1, 1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "input_std",
            torch.tensor(config.stack.input_std, dtype=torch.float32).view(1, 1, 3, 1, 1),
            persistent=False,
        )

        autogaze_mean = torch.tensor(self.autogaze_processor.image_mean, dtype=torch.float32).view(1, 1, 3, 1, 1)
        autogaze_std = torch.tensor(self.autogaze_processor.image_std, dtype=torch.float32).view(1, 1, 3, 1, 1)
        self.register_buffer("autogaze_mean", autogaze_mean, persistent=False)
        self.register_buffer("autogaze_std", autogaze_std, persistent=False)
        self.autogaze_size = self._extract_size(self.autogaze_processor.size)

        siglip_mean = torch.tensor(self.siglip_processor.image_mean, dtype=torch.float32).view(1, 1, 3, 1, 1)
        siglip_std = torch.tensor(self.siglip_processor.image_std, dtype=torch.float32).view(1, 1, 3, 1, 1)
        self.register_buffer("siglip_mean", siglip_mean, persistent=False)
        self.register_buffer("siglip_std", siglip_std, persistent=False)
        self.siglip_size = self._extract_size(self.siglip_processor.size)

    def _resolve_autogaze_repo_root(self, repo_root: str) -> Path:
        path = Path(repo_root)
        if path.is_absolute():
            return path
        return self.repo_root / path

    def _load_third_party_modules(self):
        if str(self.autogaze_repo_root) not in sys.path:
            sys.path.insert(0, str(self.autogaze_repo_root))

        from autogaze.models.autogaze import AutoGaze, AutoGazeImageProcessor
        from autogaze.vision_encoders.siglip import SiglipVisionModel

        return AutoGaze, AutoGazeImageProcessor, SiglipVisionModel

    @staticmethod
    def _extract_size(size_cfg) -> int:
        if isinstance(size_cfg, dict):
            if "height" in size_cfg:
                return int(size_cfg["height"])
            if "shortest_edge" in size_cfg:
                return int(size_cfg["shortest_edge"])
            if "longest_edge" in size_cfg:
                return int(size_cfg["longest_edge"])
        if isinstance(size_cfg, int):
            return int(size_cfg)
        raise ValueError(f"Unsupported size config: {size_cfg}")

    def _input_to_unit(self, x: torch.Tensor) -> torch.Tensor:
        mean = self.input_mean.to(x.device, x.dtype)
        std = self.input_std.to(x.device, x.dtype)
        return (x * std + mean).clamp(0.0, 1.0)

    def _prepare_autogaze_pixels(self, x: torch.Tensor) -> torch.Tensor:
        x = self._input_to_unit(x)
        batch_size, num_frames, channels, _, _ = x.shape
        x = x.flatten(0, 1)
        x = F.interpolate(
            x,
            size=(self.autogaze_size, self.autogaze_size),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        x = x.view(batch_size, num_frames, channels, self.autogaze_size, self.autogaze_size)
        x = x * 2.0 - 1.0
        x = (x - self.autogaze_mean.to(x.device, x.dtype)) / self.autogaze_std.to(x.device, x.dtype)
        return x

    def _prepare_siglip_pixels(self, x: torch.Tensor) -> torch.Tensor:
        x = self._input_to_unit(x)
        batch_size, num_frames, channels, _, _ = x.shape
        x = x.flatten(0, 1)
        x = F.interpolate(
            x,
            size=(self.siglip_size, self.siglip_size),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        x = x.view(batch_size, num_frames, channels, self.siglip_size, self.siglip_size)
        x = (x - self.siglip_mean.to(x.device, x.dtype)) / self.siglip_std.to(x.device, x.dtype)
        return x

    def _build_dense_group(self, batch_size: int, offset: int, device: torch.device) -> Dict[str, torch.Tensor]:
        positions = torch.arange(self.num_tokens_each_group, device=device, dtype=torch.long)
        positions = positions.unsqueeze(0).expand(batch_size, -1)
        return {
            "gazing_pos": positions + offset,
            "if_padded_gazing": torch.zeros(batch_size, self.num_tokens_each_group, device=device, dtype=torch.bool),
            "num_gazing_each_frame": torch.tensor([self.num_tokens_each_group], device=device, dtype=torch.long),
        }

    def _build_unified_gazing_info(
        self,
        history_info: Dict[str, torch.Tensor],
        batch_size: int,
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        parts_pos = []
        parts_pad = []
        group_lengths = []
        group_role_ids = []

        history_lengths = history_info["num_gazing_each_frame"].tolist()
        start = 0
        for length in history_lengths:
            end = start + int(length)
            parts_pos.append(history_info["gazing_pos"][:, start:end])
            parts_pad.append(history_info["if_padded_gazing"][:, start:end])
            group_lengths.append(int(length))
            group_role_ids.append(self.role_to_id["history"])
            start = end

        history_offset = len(history_lengths) * self.num_tokens_each_group
        for view_index, view_name in enumerate(self.current_view_names):
            info = self._build_dense_group(
                batch_size=batch_size,
                offset=history_offset + view_index * self.num_tokens_each_group,
                device=device,
            )
            parts_pos.append(info["gazing_pos"])
            parts_pad.append(info["if_padded_gazing"])
            group_lengths.append(int(info["num_gazing_each_frame"][0]))
            group_role_ids.append(self.role_to_id[view_name])

        unified = {
            "gazing_pos": torch.cat(parts_pos, dim=1),
            "if_padded_gazing": torch.cat(parts_pad, dim=1),
            "num_gazing_each_frame": torch.tensor(group_lengths, device=device, dtype=torch.long),
            "group_role_ids": torch.tensor(group_role_ids, device=device, dtype=torch.long),
        }

        max_allowed = self.num_tokens_each_group * (len(history_lengths) + len(self.current_view_names))
        if unified["gazing_pos"].numel() > 0:
            max_pos = int(unified["gazing_pos"].max().item())
            if max_pos >= max_allowed:
                raise ValueError(
                    "Unified gazing positions exceed range: "
                    f"max_pos={max_pos}, max_allowed={max_allowed - 1}, "
                    f"history_groups={len(history_lengths)}, "
                    f"current_views={len(self.current_view_names)}, "
                    f"num_tokens_each_group={self.num_tokens_each_group}"
                )
        return unified

    def _build_token_metadata(
        self,
        unified_gazing_info: Dict[str, torch.Tensor],
        batch_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        group_lengths = unified_gazing_info["num_gazing_each_frame"].tolist()
        group_role_ids = unified_gazing_info["group_role_ids"].tolist()

        source_ids = []
        age_ids = []
        history_group_count = sum(1 for role_id in group_role_ids if role_id == self.role_to_id["history"])
        history_seen = 0

        for role_id, length in zip(group_role_ids, group_lengths):
            source_ids.append(
                torch.full((batch_size, int(length)), int(role_id), device=device, dtype=torch.long)
            )
            if role_id == self.role_to_id["history"]:
                age_value = history_group_count - history_seen
                age_ids.append(
                    torch.full((batch_size, int(length)), age_value, device=device, dtype=torch.long)
                )
                history_seen += 1
            else:
                age_ids.append(torch.zeros(batch_size, int(length), device=device, dtype=torch.long))

        return torch.cat(source_ids, dim=1), torch.cat(age_ids, dim=1)

    def forward(
        self,
        history_frames: torch.Tensor,
        current_views: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if history_frames.dim() != 5:
            raise ValueError("history_frames must have shape [B, T, C, H, W]")
        if current_views.dim() != 5:
            raise ValueError("current_views must have shape [B, V, C, H, W]")
        if current_views.shape[1] != len(self.current_view_names):
            raise ValueError(
                f"Expected {len(self.current_view_names)} current views, got {current_views.shape[1]}"
            )

        batch_size = current_views.shape[0]
        device = current_views.device
        dtype = current_views.dtype

        history_pixels_for_gaze = self._prepare_autogaze_pixels(history_frames)
        autogaze_kwargs = {
            "gazing_ratio": self.gazing_ratio,
            "target_scales": self.scales,
            "target_patch_size": self.patch_size,
        }
        if self.task_loss_requirement is not None:
            autogaze_kwargs["task_loss_requirement"] = self.task_loss_requirement

        history_info = self.autogaze({"video": history_pixels_for_gaze}, generate_only=True, **autogaze_kwargs)

        siglip_pixels = torch.cat([history_frames, current_views], dim=1)
        siglip_pixels = self._prepare_siglip_pixels(siglip_pixels)

        unified_gazing_info = self._build_unified_gazing_info(history_info, batch_size, device)
        siglip_outputs = self.siglip(
            siglip_pixels,
            gazing_info=unified_gazing_info,
            output_hidden_states=False,
        )

        tokens = self.projector(siglip_outputs.last_hidden_state.to(dtype))
        source_ids, age_ids = self._build_token_metadata(unified_gazing_info, batch_size, device)

        if self.source_embed is not None:
            tokens = tokens + self.source_embed(source_ids)
        if self.age_embed is not None:
            tokens = tokens + self.age_embed(age_ids)

        attention_mask = (~unified_gazing_info["if_padded_gazing"]).long()
        return {
            "tokens": tokens,
            "attention_mask": attention_mask,
            "metadata": {
                "source_ids": source_ids,
                "age_ids": age_ids,
                "gazing_info": unified_gazing_info,
                "current_view_names": self.current_view_names,
            },
        }
