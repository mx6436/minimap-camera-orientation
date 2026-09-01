"""训练配置：TOML 加载、键校验与代码内默认值。"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from endfield.data_utils import SEED
from endfield.model import (
    DEFAULT_DROPOUT,
    DEFAULT_HEAD_CHANNELS,
    DEFAULT_HEAD_GRID,
    DEFAULT_NORM,
    DEFAULT_RADIUS_POOL,
)

CONFIG_DEFAULTS: dict[str, Any] = {
    "batch_size": 32,
    "epochs": 200,
    "seed": SEED,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "scheduler_patience": 8,
    "early_stop_patience": 25,
    "dropout": DEFAULT_DROPOUT,
    "norm_lambda": 0.0,
    "head_grid": list(DEFAULT_HEAD_GRID),
    "head_channels": DEFAULT_HEAD_CHANNELS,
    "radius_pool": DEFAULT_RADIUS_POOL,
    "norm": DEFAULT_NORM,
    "rotation": True,
}


def load_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        values = tomllib.load(handle)
    unknown = sorted(set(values) - set(CONFIG_DEFAULTS))
    if unknown:
        raise SystemExit(f"unknown config keys in {path}: {', '.join(unknown)}")
    config = {**CONFIG_DEFAULTS, **values}
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    if config["epochs"] < 1 or config["batch_size"] < 1:
        raise SystemExit("epochs/batch_size must be positive")
    if not 0.0 <= config["dropout"] < 1.0:
        raise SystemExit("dropout must be in [0, 1)")
    if config["norm_lambda"] < 0.0:
        raise SystemExit("norm_lambda must be non-negative")
    if len(config["head_grid"]) != 2 or min(config["head_grid"]) < 1:
        raise SystemExit("head_grid must be two positive integers")
    if config["head_channels"] < 0:
        raise SystemExit("head_channels must be non-negative (0 keeps the 256 trunk channels)")
    if config["radius_pool"] not in ("max", "avg"):
        raise SystemExit("radius_pool must be 'max' or 'avg'")
    if config["norm"] not in ("batch", "group"):
        raise SystemExit("norm must be 'batch' or 'group'")
    if config["lr"] <= 0.0 or config["weight_decay"] < 0.0:
        raise SystemExit("lr must be positive and weight_decay non-negative")
    if config["scheduler_patience"] < 1 or config["early_stop_patience"] < 1:
        raise SystemExit("scheduler_patience/early_stop_patience must be positive")
