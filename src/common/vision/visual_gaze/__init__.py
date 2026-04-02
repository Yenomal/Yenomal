"""Composable visual gaze encoder."""

from .config import VisualGazeConfig, load_config, load_default_config
from .visual_gaze_encoder import VisualGazeEncoder

__all__ = [
    "VisualGazeConfig",
    "VisualGazeEncoder",
    "load_config",
    "load_default_config",
]
