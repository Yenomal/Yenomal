from __future__ import annotations

from dataclasses import asdict, dataclass, field
from importlib.resources import files
from pathlib import Path
from typing import Any, Mapping

import yaml


def _tuple_of_floats(values: Any, expected_len: int) -> tuple[float, ...]:
    if len(values) != expected_len:
        raise ValueError(f"Expected {expected_len} values, got {len(values)}")
    return tuple(float(v) for v in values)


def _merge_dict(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


@dataclass(frozen=True)
class InputConfig:
    input_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    input_std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    current_view_names: tuple[str, ...] = ("head", "left", "right")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "InputConfig":
        payload = dict(data)
        if "input_mean" in payload:
            payload["input_mean"] = _tuple_of_floats(payload["input_mean"], 3)
        if "input_std" in payload:
            payload["input_std"] = _tuple_of_floats(payload["input_std"], 3)
        if "current_view_names" in payload:
            payload["current_view_names"] = tuple(str(v) for v in payload["current_view_names"])
        return cls(**payload)


@dataclass(frozen=True)
class GazePolicyConfig:
    name: str = "autogaze"
    init_mode: str = "pretrained"
    model_path: str = "nvidia/AutoGaze"
    history_len: int = 8
    gazing_ratio: float = 0.10
    task_loss_requirement: float | None = None
    image_size: int = 224
    image_mean: tuple[float, float, float] = (0.5, 0.5, 0.5)
    image_std: tuple[float, float, float] = (0.5, 0.5, 0.5)
    model_config: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GazePolicyConfig":
        payload = dict(data)
        if "image_mean" in payload:
            payload["image_mean"] = _tuple_of_floats(payload["image_mean"], 3)
        if "image_std" in payload:
            payload["image_std"] = _tuple_of_floats(payload["image_std"], 3)
        payload["model_config"] = dict(payload.get("model_config", {}))
        return cls(**payload)


@dataclass(frozen=True)
class EncoderConfig:
    name: str = "siglip"
    init_mode: str = "pretrained"
    model_path: str = "google/siglip2-base-patch16-224"
    image_size: int = 224
    image_mean: tuple[float, float, float] = (0.5, 0.5, 0.5)
    image_std: tuple[float, float, float] = (0.5, 0.5, 0.5)
    attn_implementation: str = "sdpa"
    attn_type: str = "bidirectional"
    model_config: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EncoderConfig":
        payload = dict(data)
        if "image_mean" in payload:
            payload["image_mean"] = _tuple_of_floats(payload["image_mean"], 3)
        if "image_std" in payload:
            payload["image_std"] = _tuple_of_floats(payload["image_std"], 3)
        payload["model_config"] = dict(payload.get("model_config", {}))
        return cls(**payload)


@dataclass(frozen=True)
class ProjectorConfig:
    name: str = "mlp"
    output_dim: int = 960
    hidden_dim: int = 1536
    add_source_embeddings: bool = True
    add_age_embeddings: bool = True

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ProjectorConfig":
        return cls(**dict(data))


@dataclass(frozen=True)
class VisualGazeConfig:
    input: InputConfig = field(default_factory=InputConfig)
    gaze_policy: GazePolicyConfig = field(default_factory=GazePolicyConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    projector: ProjectorConfig = field(default_factory=ProjectorConfig)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "VisualGazeConfig":
        payload = dict(data)
        return cls(
            input=InputConfig.from_dict(payload.get("input", {})),
            gaze_policy=GazePolicyConfig.from_dict(payload.get("gaze_policy", {})),
            encoder=EncoderConfig.from_dict(payload.get("encoder", {})),
            projector=ProjectorConfig.from_dict(payload.get("projector", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_default_config() -> VisualGazeConfig:
    default_path = files(__package__).joinpath("config.yaml")
    with default_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return VisualGazeConfig.from_dict(data)


def load_config(path: str | Path | None = None, overrides: Mapping[str, Any] | None = None) -> VisualGazeConfig:
    if path is None:
        config = load_default_config()
    else:
        with Path(path).expanduser().open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        config = VisualGazeConfig.from_dict(data)
    if not overrides:
        return config
    merged = _merge_dict(config.to_dict(), overrides)
    return VisualGazeConfig.from_dict(merged)
