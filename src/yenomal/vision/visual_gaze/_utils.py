from __future__ import annotations

import os
from pathlib import Path

import torch
from safetensors.torch import load_file as load_safetensors


def resolve_existing_path(path_str: str | None) -> Path | None:
    if not path_str:
        return None
    path = Path(path_str).expanduser()
    if path.is_absolute() and path.exists():
        return path.resolve()
    candidate = Path.cwd() / path
    if candidate.exists():
        return candidate.resolve()
    if path.exists():
        return path.resolve()
    return None


def resolve_model_path(model_path: str | None) -> Path | None:
    direct = resolve_existing_path(model_path)
    if direct is not None:
        return direct
    if not model_path or "/" not in model_path:
        return None

    cache_key = "models--" + model_path.replace("/", "--")
    candidate_roots: list[Path] = []
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        candidate_roots.append(Path(hf_home) / "hub")
    hub_cache = os.environ.get("HUGGINGFACE_HUB_CACHE")
    if hub_cache:
        candidate_roots.append(Path(hub_cache))
    candidate_roots.extend(
        [
            Path.cwd() / ".hf" / "hub",
            Path.cwd() / ".cache" / ".hf" / "hub",
            Path.cwd() / ".cache" / "huggingface" / "hub",
        ]
    )

    for root in candidate_roots:
        repo_cache = root / cache_key
        snapshots_dir = repo_cache / "snapshots"
        if not snapshots_dir.exists():
            continue
        ref_path = repo_cache / "refs" / "main"
        if ref_path.exists():
            snapshot = snapshots_dir / ref_path.read_text(encoding="utf-8").strip()
            if snapshot.exists():
                return snapshot.resolve()
        snapshots = sorted(snapshots_dir.iterdir(), reverse=True)
        if snapshots:
            return snapshots[0].resolve()
    return None


def load_state_dict(model_dir: Path) -> dict[str, torch.Tensor]:
    safetensors_path = model_dir / "model.safetensors"
    pytorch_bin_path = model_dir / "pytorch_model.bin"
    if safetensors_path.exists():
        return load_safetensors(str(safetensors_path), device="cpu")
    if pytorch_bin_path.exists():
        return torch.load(str(pytorch_bin_path), map_location="cpu")
    raise FileNotFoundError(f"No model weights found under {model_dir}")


def load_local_pretrained_model(model_cls, model_dir: Path, config_overrides: dict | None = None):
    raw_config = model_cls.config_class.get_config_dict(str(model_dir))[0]
    if model_cls.__name__ == "SiglipVisionModel" and "vision_config" in raw_config:
        config = model_cls.config_class.from_dict(raw_config["vision_config"])
    else:
        config = model_cls.config_class.from_dict(raw_config)
    if config_overrides:
        for key, value in config_overrides.items():
            setattr(config, key, value)

    model = model_cls(config)
    state_dict = load_state_dict(model_dir)
    if model_cls.__name__ == "SiglipVisionModel":
        state_dict = {key: value for key, value in state_dict.items() if key.startswith("vision_model.")}
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    allowed_missing = {"vision_model.embeddings.position_ids"}
    invalid_missing = [key for key in missing_keys if key not in allowed_missing]
    if invalid_missing:
        raise RuntimeError(f"Missing keys when loading {model_dir}: {invalid_missing[:20]}")
    if unexpected_keys:
        raise RuntimeError(f"Unexpected keys when loading {model_dir}: {unexpected_keys[:20]}")
    return model


def extract_size(size_cfg) -> int:
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
