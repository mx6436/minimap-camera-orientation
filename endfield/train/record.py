"""run 档案（record.json）：复现一次训练所需的配置、环境与数据指纹快照。"""

from __future__ import annotations

from typing import Any

import torch

from endfield.model import ARCHITECTURE_CONE


def build_record(
    config: dict[str, Any],
    threads: int,
    device: torch.device,
    train_count: int,
    val_count: int,
    train_sha256: str,
    val_sha256: str,
) -> dict[str, Any]:
    cone = config["architecture"] == ARCHITECTURE_CONE
    loss_description = (
        "KL(q||p) over 360 bins, q = circular gaussian sigma=2 deg"
        " (= cross entropy minus constant target entropy H(q))"
        if cone
        else "MSE([raw_sin, raw_cos], [target_sin, target_cos])"
    )
    if not cone and config["norm_lambda"] > 0.0:
        loss_description += f" + {config['norm_lambda']:g}*(||v||-1)^2"
    track_metric = "expected_rmse" if cone else "circular_rmse"
    return {
        "version": 20,
        "architecture": config["architecture"],
        "head_grid": None if cone else list(config["head_grid"]),
        "head_channels": None if cone else config["head_channels"] or None,
        "radius_pool": None if cone else config["radius_pool"],
        "norm": None if cone else config["norm"],
        "dropout": 0.0 if cone else config["dropout"],
        "norm_lambda": 0.0 if cone else config["norm_lambda"],
        "loss": loss_description,
        "input_shape": [3, 44, 360],
        "input_scaling": "RGB uint8 / 255",
        "input_representation": (
            "polar_unwrap_rgb_360x44 (angle->x, 1 deg/column, clockwise, north at column 0; "
            "radius->y, inner at top)"
        ),
        "conv_padding_mode": "azimuth-circular; radius-zero (radius boundaries are ring-outside)",
        "seed": config["seed"],
        "threads": threads,
        "device": str(device),
        "model": "ConeCNN" if cone else "AngleCNN",
        "trainable_parameters": None,
        "batch_size": config["batch_size"],
        "max_epochs": config["epochs"],
        "optimizer": "AdamW",
        "learning_rate": config["lr"],
        "weight_decay": config["weight_decay"],
        "scheduler": {
            "name": "ReduceLROnPlateau",
            "metric": track_metric,
            "patience": config["scheduler_patience"],
            "factor": 0.5,
            "min_lr": 1e-6,
        },
        "early_stopping_metric": track_metric,
        "early_stopping_patience": config["early_stop_patience"],
        "augmentation": {
            "rgb_gaussian_noise": {
                "probability": 0.5,
                "sigma": 0.02,
                "masked_to_ring_alpha": False,
            },
            "clockwise_rotation": {
                "probability": 0.5 if config["rotation"] else 0.0,
                "degrees": (
                    "15-multiples 15..345 (24 directions); "
                    "lossless np.roll on the 1 deg/column angular axis"
                ),
            },
        },
        "train_count": train_count,
        "val_count": val_count,
        "train_files_sha256": train_sha256,
        "val_files_sha256": val_sha256,
    }
