"""run 档案（record.json）：复现一次训练所需的配置、环境与数据指纹快照。"""

from __future__ import annotations

from typing import Any

import torch


def build_record(
    config: dict[str, Any],
    threads: int,
    device: torch.device,
    train_count: int,
    val_count: int,
    train_sha256: str,
    val_sha256: str,
) -> dict[str, Any]:
    return {
        "version": 23,
        "target_sigma": config["target_sigma"],
        "loss": (
            f"KL(q||p) between circular categorical distributions on Z/360Z, "
            f"q = wrapped gaussian pmf with sigma={config['target_sigma']:g} deg "
            "(= cross entropy minus constant target entropy H(q))"
        ),
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
        "model": "AzimuthNet",
        "trainable_parameters": None,
        "batch_size": config["batch_size"],
        "max_epochs": config["epochs"],
        "optimizer": "AdamW",
        "learning_rate": config["lr"],
        "weight_decay": config["weight_decay"],
        "scheduler": {
            "name": "ReduceLROnPlateau",
            "metric": "expected_rmse",
            "patience": config["scheduler_patience"],
            "factor": 0.5,
            "min_lr": 1e-6,
        },
        "early_stopping_metric": "expected_rmse",
        "early_stopping_patience": config["early_stop_patience"],
        "augmentation": (
            {
                "rgb_gaussian_noise": {
                    "probability": 0.5,
                    "sigma": 0.02,
                    "masked_to_ring_alpha": False,
                },
            }
            if config["noise_augment"]
            else {}
        ),
        "train_count": train_count,
        "val_count": val_count,
        "train_files_sha256": train_sha256,
        "val_files_sha256": val_sha256,
    }
