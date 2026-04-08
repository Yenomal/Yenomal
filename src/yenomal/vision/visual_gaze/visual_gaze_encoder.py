from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn as nn

from ...nn import TokenProjector
from ..encoder import build_siglip_modules
from ..gaze_policy import AutoGazePolicy

from .config import VisualGazeConfig, load_config, load_default_config


class VisualGazeEncoder(nn.Module):
    """Composable sparse+dense visual encoder with a one-call public API."""

    def __init__(
        self,
        config: VisualGazeConfig,
        *,
        gaze_policy: AutoGazePolicy,
        tokenizer: nn.Module,
        encoder: nn.Module,
        projector: nn.Module,
    ):
        super().__init__()
        self.config = config
        self.gaze_policy = gaze_policy
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.projector = projector

        self.current_view_names = tuple(config.input.current_view_names)
        self.history_len = int(config.gaze_policy.history_len)
        self.add_source_embeddings = bool(config.projector.add_source_embeddings)
        self.add_age_embeddings = bool(config.projector.add_age_embeddings)
        self.attn_implementation = str(config.encoder.attn_implementation)

        self.role_to_id = {"history": 0}
        for index, view_name in enumerate(self.current_view_names, start=1):
            self.role_to_id[view_name] = index

        if self.add_source_embeddings:
            self.source_embed = nn.Embedding(len(self.role_to_id), self.projector.out_dim)
        else:
            self.source_embed = None

        if self.add_age_embeddings:
            self.age_embed = nn.Embedding(self.history_len + 1, self.projector.out_dim)
        else:
            self.age_embed = None

        self.register_buffer(
            "input_mean",
            torch.tensor(config.input.input_mean, dtype=torch.float32).view(1, 1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "input_std",
            torch.tensor(config.input.input_std, dtype=torch.float32).view(1, 1, 3, 1, 1),
            persistent=False,
        )

    @classmethod
    def from_default(cls, overrides: Mapping[str, Any] | None = None) -> "VisualGazeEncoder":
        return cls.from_config(load_default_config() if overrides is None else load_config(overrides=overrides))

    @classmethod
    def from_yaml(cls, path: str, overrides: Mapping[str, Any] | None = None) -> "VisualGazeEncoder":
        return cls.from_config(load_config(path, overrides=overrides))

    @classmethod
    def from_config(cls, config: VisualGazeConfig) -> "VisualGazeEncoder":
        if config.gaze_policy.name != "autogaze":
            raise ValueError(f"Unsupported gaze policy: {config.gaze_policy.name}")
        if config.encoder.name != "siglip":
            raise ValueError(f"Unsupported encoder: {config.encoder.name}")
        if config.projector.name != "mlp":
            raise ValueError(f"Unsupported projector: {config.projector.name}")

        gaze_policy = AutoGazePolicy(config.gaze_policy)
        tokenizer, encoder = build_siglip_modules(config.encoder, scales=gaze_policy.scales)
        projector = TokenProjector(
            in_dim=tokenizer.hidden_size,
            out_dim=int(config.projector.output_dim),
            hidden_dim=int(config.projector.hidden_dim),
        )
        return cls(
            config,
            gaze_policy=gaze_policy,
            tokenizer=tokenizer,
            encoder=encoder,
            projector=projector,
        )

    def _to_unit(self, x: torch.Tensor) -> torch.Tensor:
        return (x * self.input_std.to(x.device, x.dtype) + self.input_mean.to(x.device, x.dtype)).clamp(0.0, 1.0)

    def _build_dense_group(self, batch_size: int, offset: int, device: torch.device) -> dict[str, torch.Tensor]:
        positions = torch.arange(self.tokenizer.num_tokens_each_group, device=device, dtype=torch.long)
        positions = positions.unsqueeze(0).expand(batch_size, -1)
        return {
            "gazing_pos": positions + offset,
            "if_padded_gazing": torch.zeros(batch_size, self.tokenizer.num_tokens_each_group, device=device, dtype=torch.bool),
            "num_gazing_each_frame": torch.tensor([self.tokenizer.num_tokens_each_group], device=device, dtype=torch.long),
        }

    def _build_unified_selection(
        self,
        history_selection: dict[str, torch.Tensor] | None,
        *,
        batch_size: int,
        device: torch.device,
        dense_view_count: int,
    ) -> dict[str, Any]:
        parts_pos: list[torch.Tensor] = []
        parts_pad: list[torch.Tensor] = []
        group_lengths: list[int] = []
        group_roles: list[str] = []

        history_group_count = 0
        if history_selection is not None:
            history_lengths = history_selection["num_gazing_each_frame"].tolist()
            history_group_count = len(history_lengths)
            start = 0
            for length in history_lengths:
                end = start + int(length)
                parts_pos.append(history_selection["gazing_pos"][:, start:end])
                parts_pad.append(history_selection["if_padded_gazing"][:, start:end])
                group_lengths.append(int(length))
                group_roles.append("history")
                start = end

        offset = history_group_count * self.tokenizer.num_tokens_each_group
        for view_index in range(dense_view_count):
            view_name = self.current_view_names[view_index]
            info = self._build_dense_group(batch_size=batch_size, offset=offset, device=device)
            parts_pos.append(info["gazing_pos"])
            parts_pad.append(info["if_padded_gazing"])
            group_lengths.append(int(info["num_gazing_each_frame"][0]))
            group_roles.append(view_name)
            offset += self.tokenizer.num_tokens_each_group

        if not parts_pos:
            raise ValueError("At least one sparse or dense branch must be enabled.")

        unified = {
            "gazing_pos": torch.cat(parts_pos, dim=1),
            "if_padded_gazing": torch.cat(parts_pad, dim=1),
            "num_gazing_each_frame": torch.tensor(group_lengths, device=device, dtype=torch.long),
            "group_roles": tuple(group_roles),
        }
        total_groups = history_group_count + dense_view_count
        max_allowed = self.tokenizer.num_tokens_each_group * total_groups
        if unified["gazing_pos"].numel() > 0:
            max_pos = int(unified["gazing_pos"].max().item())
            if max_pos >= max_allowed:
                raise ValueError(
                    "Unified gazing positions exceed range: "
                    f"max_pos={max_pos}, max_allowed={max_allowed - 1}, "
                    f"total_groups={total_groups}, num_tokens_each_group={self.tokenizer.num_tokens_each_group}"
                )
        return unified

    def _build_token_metadata(
        self,
        selection: dict[str, Any],
        *,
        batch_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        group_lengths = selection["num_gazing_each_frame"].tolist()
        group_roles = selection["group_roles"]

        source_ids: list[torch.Tensor] = []
        age_ids: list[torch.Tensor] = []
        history_group_count = sum(1 for role in group_roles if role == "history")
        if history_group_count > self.history_len:
            raise ValueError(
                f"History groups exceed configured history_len: {history_group_count} > {self.history_len}"
            )
        history_seen = 0
        for role, length in zip(group_roles, group_lengths):
            if role not in self.role_to_id:
                raise ValueError(f"Unknown role '{role}' in group_roles.")
            source_ids.append(torch.full((batch_size, int(length)), self.role_to_id[role], device=device, dtype=torch.long))
            if role == "history":
                age_value = history_group_count - history_seen
                age_ids.append(torch.full((batch_size, int(length)), age_value, device=device, dtype=torch.long))
                history_seen += 1
            else:
                age_ids.append(torch.zeros(batch_size, int(length), device=device, dtype=torch.long))
        return torch.cat(source_ids, dim=1), torch.cat(age_ids, dim=1)

    def _build_encoder_attention_mask(
        self,
        selection: dict[str, Any],
        *,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        if self.attn_implementation != "sdpa":
            raise ValueError(
                f"Only sdpa attention is supported in the first VisualGaze version, got {self.attn_implementation}."
            )

        lengths = [int(length) for length in selection["num_gazing_each_frame"].tolist()]
        roles = list(selection["group_roles"])
        total_len = sum(lengths)
        offsets = [0]
        for length in lengths:
            offsets.append(offsets[-1] + length)

        allowed = torch.zeros(batch_size, total_len, total_len, device=device, dtype=torch.float32)

        history_indices = [idx for idx, role in enumerate(roles) if role == "history"]
        current_indices = [idx for idx, role in enumerate(roles) if role != "history"]
        head_indices = [idx for idx, role in enumerate(roles) if role == "head"]

        for history_pos, group_idx in enumerate(history_indices):
            q0, q1 = offsets[group_idx], offsets[group_idx + 1]
            for key_idx in history_indices[: history_pos + 1]:
                k0, k1 = offsets[key_idx], offsets[key_idx + 1]
                allowed[:, q0:q1, k0:k1] = 1.0

        for group_idx in current_indices:
            q0, q1 = offsets[group_idx], offsets[group_idx + 1]
            for key_idx in current_indices:
                k0, k1 = offsets[key_idx], offsets[key_idx + 1]
                allowed[:, q0:q1, k0:k1] = 1.0

        for head_idx in head_indices:
            q0, q1 = offsets[head_idx], offsets[head_idx + 1]
            for key_idx in history_indices:
                k0, k1 = offsets[key_idx], offsets[key_idx + 1]
                allowed[:, q0:q1, k0:k1] = 1.0

        diag = torch.arange(total_len, device=device)
        allowed[:, diag, diag] = 1.0

        valid_mask = ~selection["if_padded_gazing"]
        allowed = allowed * valid_mask.unsqueeze(1).to(allowed.dtype) * valid_mask.unsqueeze(2).to(allowed.dtype)
        additive = torch.where(allowed > 0, torch.zeros_like(allowed), torch.full_like(allowed, float("-inf")))
        return additive.unsqueeze(1).expand(-1, self.encoder.num_heads, -1, -1).to(dtype)

    def forward(
        self,
        *,
        sparse_video: torch.Tensor | None = None,
        dense_images: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        if sparse_video is None and dense_images is None:
            raise ValueError("At least one input branch must be provided.")

        if sparse_video is not None and sparse_video.dim() != 5:
            raise ValueError("sparse_video must have shape [B, T, C, H, W]")
        if dense_images is not None and dense_images.dim() != 5:
            raise ValueError("dense_images must have shape [B, V, C, H, W]")

        batch_size = sparse_video.shape[0] if sparse_video is not None else dense_images.shape[0]
        device = sparse_video.device if sparse_video is not None else dense_images.device
        dtype = sparse_video.dtype if sparse_video is not None else dense_images.dtype

        if sparse_video is not None and sparse_video.shape[0] != batch_size:
            raise ValueError("sparse_video batch size mismatch.")
        if dense_images is not None and dense_images.shape[0] != batch_size:
            raise ValueError("dense_images batch size mismatch.")
        if dense_images is not None and dense_images.shape[1] != len(self.current_view_names):
            raise ValueError(
                f"Expected {len(self.current_view_names)} dense views, got {dense_images.shape[1]}."
            )

        sparse_unit = self._to_unit(sparse_video) if sparse_video is not None else None
        dense_unit = self._to_unit(dense_images) if dense_images is not None else None

        history_selection = None
        if sparse_unit is not None:
            history_selection = self.gaze_policy(
                sparse_unit,
                target_scales=self.tokenizer.scales,
                target_patch_size=self.tokenizer.patch_size,
            )

        pixel_parts = []
        if sparse_unit is not None:
            pixel_parts.append(sparse_unit)
        if dense_unit is not None:
            pixel_parts.append(dense_unit)
        combined_pixels = torch.cat(pixel_parts, dim=1)

        unified_selection = self._build_unified_selection(
            history_selection,
            batch_size=batch_size,
            device=device,
            dense_view_count=0 if dense_unit is None else dense_unit.shape[1],
        )
        token_embeddings = self.tokenizer(combined_pixels, unified_selection)
        encoder_attention_mask = self._build_encoder_attention_mask(
            unified_selection,
            batch_size=batch_size,
            dtype=token_embeddings.dtype,
            device=device,
        )
        tokens = self.encoder(token_embeddings, encoder_attention_mask).to(dtype)
        tokens = self.projector(tokens)

        source_ids, age_ids = self._build_token_metadata(
            unified_selection,
            batch_size=batch_size,
            device=device,
        )
        if self.source_embed is not None:
            tokens = tokens + self.source_embed(source_ids)
        if self.age_embed is not None:
            tokens = tokens + self.age_embed(age_ids)

        attention_mask = (~unified_selection["if_padded_gazing"]).long()
        return {
            "tokens": tokens,
            "attention_mask": attention_mask,
            "metadata": {
                "has_sparse": sparse_video is not None,
                "has_dense": dense_images is not None,
                "source_ids": source_ids,
                "age_ids": age_ids,
                "selection": unified_selection,
                "encoder_attention_mask": encoder_attention_mask,
            },
        }
