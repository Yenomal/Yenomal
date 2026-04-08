from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from transformers import AutoImageProcessor

from ...nn.tokenizer import SiglipPatchTokenizer
from ..vendor.autogaze.vision_encoders.siglip.configuration_siglip import SiglipVisionConfig
from ..vendor.autogaze.vision_encoders.siglip.modeling_siglip import SiglipVisionModel
from ..visual_gaze._utils import load_local_pretrained_model, resolve_model_path


class SiglipTokenEncoder(nn.Module):
    """Run only the transformer encoder layers on precomputed token embeddings."""

    def __init__(self, encoder: nn.Module, post_layernorm: nn.Module, *, num_heads: int, hidden_size: int):
        super().__init__()
        self.encoder = encoder
        self.post_layernorm = post_layernorm
        self.num_heads = int(num_heads)
        self.hidden_size = int(hidden_size)

    def forward(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        encoder_outputs = self.encoder(
            inputs_embeds=token_embeddings,
            attention_mask=attention_mask,
            output_attentions=False,
            output_hidden_states=False,
        )
        return self.post_layernorm(encoder_outputs.last_hidden_state)


def build_siglip_modules(config: Any, *, scales: tuple[int, ...] | None = None) -> tuple[SiglipPatchTokenizer, SiglipTokenEncoder]:
    config_overrides: dict[str, Any] = {
        "_attn_implementation": config.attn_implementation,
        "attn_type": config.attn_type,
    }
    if scales is not None:
        config_overrides["scales"] = "+".join(str(scale) for scale in scales)

    if config.init_mode == "pretrained":
        model_dir = resolve_model_path(config.model_path)
        if model_dir is None:
            raise FileNotFoundError(f"Cannot resolve SigLIP model path: {config.model_path}")
        model = load_local_pretrained_model(SiglipVisionModel, model_dir, config_overrides=config_overrides)
        processor = AutoImageProcessor.from_pretrained(str(model_dir))
        image_mean = tuple(float(v) for v in processor.image_mean)
        image_std = tuple(float(v) for v in processor.image_std)
        image_size = int(config.image_size if config.image_size is not None else model.config.image_size)
    elif config.init_mode == "random":
        payload = dict(config.model_config)
        if scales is not None:
            payload["scales"] = "+".join(str(scale) for scale in scales)
        payload["_attn_implementation"] = config.attn_implementation
        payload["attn_type"] = config.attn_type
        vision_config = SiglipVisionConfig.from_dict(payload)
        model = SiglipVisionModel(vision_config)
        image_mean = tuple(float(v) for v in config.image_mean)
        image_std = tuple(float(v) for v in config.image_std)
        image_size = int(config.image_size)
    else:
        raise ValueError(f"Unsupported SigLIP init_mode: {config.init_mode}")

    tokenizer = SiglipPatchTokenizer(
        model.vision_model.embeddings,
        image_size=image_size,
        image_mean=image_mean,
        image_std=image_std,
    )
    encoder = SiglipTokenEncoder(
        model.vision_model.encoder,
        model.vision_model.post_layernorm,
        num_heads=model.config.num_attention_heads,
        hidden_size=model.config.hidden_size,
    )
    return tokenizer, encoder
