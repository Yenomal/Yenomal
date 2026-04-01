from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


def _tuple_of_floats(values: Any, expected_len: int) -> tuple[float, ...]:
    if len(values) != expected_len:
        raise ValueError(f"Expected {expected_len} values, got {len(values)}")
    return tuple(float(v) for v in values)


@dataclass(frozen=True)
class StackConfig:
    name: str = "autogaze_siglip"
    autogaze_repo_root: str = "third_party/AutoGaze"
    autogaze_model_path: str = "nvidia/AutoGaze"
    siglip_model_path: str = "google/siglip2-base-patch16-224"
    history_len: int = 8
    gazing_ratio: float = 0.10
    task_loss_requirement: float | None = None
    current_view_names: tuple[str, ...] = ("head", "left", "right")
    input_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    input_std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    attn_implementation: str = "sdpa"
    attn_type: str = "bidirectional"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "StackConfig":
        payload = dict(data)
        if "current_view_names" in payload:
            payload["current_view_names"] = tuple(str(v) for v in payload["current_view_names"])
        if "input_mean" in payload:
            payload["input_mean"] = _tuple_of_floats(payload["input_mean"], 3)
        if "input_std" in payload:
            payload["input_std"] = _tuple_of_floats(payload["input_std"], 3)
        return cls(**payload)


@dataclass(frozen=True)
class AdapterConfig:
    output_dim: int = 960
    projector_hidden_dim: int = 1536
    add_source_embeddings: bool = True
    add_age_embeddings: bool = True

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AdapterConfig":
        return cls(**dict(data))


@dataclass(frozen=True)
class AutoGazeSiglipConfig:
    stack: StackConfig = field(default_factory=StackConfig)
    adapter: AdapterConfig = field(default_factory=AdapterConfig)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AutoGazeSiglipConfig":
        payload = dict(data)
        stack_cfg = StackConfig.from_dict(payload.get("stack", {}))
        adapter_cfg = AdapterConfig.from_dict(payload.get("adapter", {}))
        return cls(stack=stack_cfg, adapter=adapter_cfg)


def load_config(path: str | Path) -> AutoGazeSiglipConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        import yaml

        data = yaml.safe_load(f) or {}
    return AutoGazeSiglipConfig.from_dict(data)
