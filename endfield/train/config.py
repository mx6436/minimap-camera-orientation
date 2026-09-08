"""训练配置：TOML 加载、键校验与代码内默认值。"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from endfield.data_utils import SEED
from endfield.model import TARGET_SIGMA

CONFIG_DEFAULTS: dict[str, Any] = {
    "batch_size": 128,
    "epochs": 200,
    "seed": SEED,
    "target_sigma": TARGET_SIGMA,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "scheduler_patience": 8,
    "early_stop_patience": 25,
    "noise_augment": False,
    "roll_augment": True,
    "precision": "bf16",
    "compile": True,
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
    if config["lr"] <= 0.0 or config["weight_decay"] < 0.0:
        raise SystemExit("lr must be positive and weight_decay non-negative")
    if config["scheduler_patience"] < 1 or config["early_stop_patience"] < 1:
        raise SystemExit("scheduler_patience/early_stop_patience must be positive")
    if config["target_sigma"] <= 0.0 or config["target_sigma"] >= 90.0:
        raise SystemExit("target_sigma must be in degrees, 0 < target_sigma < 90")
    if not isinstance(config["noise_augment"], bool):
        raise SystemExit("noise_augment must be a boolean")
    if not isinstance(config["roll_augment"], bool):
        raise SystemExit("roll_augment must be a boolean")
    if config["precision"] not in ("fp32", "bf16"):
        raise SystemExit('precision must be "fp32" or "bf16"')
    if not isinstance(config["compile"], bool):
        raise SystemExit("compile must be a boolean")
