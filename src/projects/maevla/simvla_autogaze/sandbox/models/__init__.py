"""Experiment-local overrides for SimVLA models.

This package shadows the upstream ``models`` package only where needed.
Everything not overridden here still resolves from ``third_party/SimVLA/models``.
"""

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[7]
_UPSTREAM_MODELS = _REPO_ROOT / "third_party" / "SimVLA" / "models"

if str(_UPSTREAM_MODELS) not in __path__:
    __path__.append(str(_UPSTREAM_MODELS))
