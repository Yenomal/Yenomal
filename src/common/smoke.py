from __future__ import annotations

import torch

from vision.visual_gaze import VisualGazeEncoder


def build_smoke_overrides() -> dict:
    return {
        "gaze_policy": {
            "init_mode": "random",
            "history_len": 4,
            "gazing_ratio": 0.25,
            "image_size": 64,
            "model_config": {
                "scales": "64",
                "num_vision_tokens_each_frame": 16,
                "gaze_model_config": {
                    "input_img_size": 64,
                    "vision_model_config": {
                        "hidden_dim": 32,
                        "out_dim": 32,
                        "depth": 1,
                        "kernel_size": 16,
                        "temporal_patch_size": 1,
                    },
                    "connector_config": {
                        "hidden_dim": 32,
                    },
                    "gaze_decoder_config": {
                        "hidden_size": 32,
                        "intermediate_size": 64,
                        "num_hidden_layers": 2,
                        "num_attention_heads": 4,
                        "num_key_value_heads": 4,
                        "num_multi_token_pred": 1,
                    },
                },
            },
        },
        "encoder": {
            "init_mode": "random",
            "image_size": 64,
            "attn_implementation": "sdpa",
            "attn_type": "bidirectional",
            "model_config": {
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "image_size": 64,
                "patch_size": 16,
            },
        },
        "projector": {
            "output_dim": 48,
            "hidden_dim": 96,
        },
    }


def run_case(model: VisualGazeEncoder, name: str, *, sparse_video=None, dense_images=None) -> None:
    outputs = model(sparse_video=sparse_video, dense_images=dense_images)
    print(
        f"{name}: tokens={tuple(outputs['tokens'].shape)} "
        f"attention_mask={tuple(outputs['attention_mask'].shape)} "
        f"has_sparse={outputs['metadata']['has_sparse']} has_dense={outputs['metadata']['has_dense']}"
    )


def main() -> None:
    torch.manual_seed(0)
    model = VisualGazeEncoder.from_default(overrides=build_smoke_overrides())
    model.eval()

    batch_size = 2
    sparse_video = torch.randn(batch_size, 4, 3, 64, 64)
    dense_images = torch.randn(batch_size, 3, 3, 64, 64)

    with torch.no_grad():
        run_case(model, "sparse+dense", sparse_video=sparse_video, dense_images=dense_images)
        run_case(model, "sparse_only", sparse_video=sparse_video, dense_images=None)
        run_case(model, "dense_only", sparse_video=None, dense_images=dense_images)


if __name__ == "__main__":
    main()
