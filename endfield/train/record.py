"""run 档案（record.json）：复现一次训练所需的配置、环境与数据指纹快照。"""

from __future__ import annotations

from typing import Any

import torch

from endfield.polar import IMG_H, IMG_W


def augmentation(config: dict[str, Any]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    if config["noise_augment"]:
        values["rgb_gaussian_noise"] = {
            "probability": 0.5,
            "sigma": 0.02,
            "masked_to_ring_alpha": False,
        }
    if config["roll_augment"]:
        values["azimuth_roll"] = {
            "axis": "azimuth",
            "delta_sample": "uniform integer [0, 360)",
            "label_shift": "same delta mod 360",
        }
    return values


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
        "version": 25,
        "target_sigma": config["target_sigma"],
        "loss": (
            f"KL(q||p) between circular categorical distributions on Z/360Z, "
            f"q = wrapped gaussian pmf with sigma={config['target_sigma']:g} deg "
            "(= cross entropy minus constant target entropy H(q))"
        ),
        "input_shape": [3, IMG_H, IMG_W],
        "input_scaling": "RGB uint8 / 255",
        "input_representation": (
            f"polar_unwrap_rgb_{IMG_W}x{IMG_H} (angle->x, 1 deg/column, clockwise, north at column 0; "
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
            "metric": "rms_error",
            "patience": config["scheduler_patience"],
            "factor": 0.5,
            "min_lr": 1e-6,
        },
        "early_stopping_metric": "rms_error",
        "early_stopping_patience": config["early_stop_patience"],
        "augmentation": augmentation(config),
        "train_count": train_count,
        "val_count": val_count,
        "train_files_sha256": train_sha256,
        "val_files_sha256": val_sha256,
    }
