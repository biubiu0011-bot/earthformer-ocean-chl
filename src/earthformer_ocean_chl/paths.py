"""Portable paths for data and generated experiment outputs."""

from __future__ import annotations

import os
from pathlib import Path


def data_root() -> Path:
    """Return the local data root selected by the user or CI environment."""
    return Path(os.environ.get("EARTHFORMER_OCEAN_CHL_DATA_ROOT", "data")).expanduser()


def results_root() -> Path:
    """Return the directory used for generated metrics, predictions, and figures."""
    return Path(os.environ.get("EARTHFORMER_OCEAN_CHL_RESULTS_ROOT", "results")).expanduser()


def pretrain_path(*parts: str) -> Path:
    """Return a location underneath the NEMO pre-training data directory."""
    return data_root().joinpath("pre_train_data", *parts)

