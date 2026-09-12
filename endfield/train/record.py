"""run 档案（record.json）：复现一次训练所需的配置、环境与数据指纹快照。"""

from __future__ import annotations

from typing import Any

import torch

from endfield.polar import IMG_H, IMG_W
from endfield.train.data import input_channels


def augmentation(config: dict[str, Any]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    if config["noise_augment"]:
        values["bgr_gaussian_noise"] = {
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


def input_representation(config: dict[str, Any]) -> str:
    if config["input_mode"] == "ref":
        return (
            f"ref_polar_unwrap_obs_bgr_ref_bgra_{IMG_W}x{IMG_H} "
            "(channels = [obs.BGR, ref.BGR, ref.A]; reference = MapLocator zone asset "
            "cropped at (x,y) with the zone's MapLocator ZoneTemplateScale "
            "(ValleyIV_Base 15/16, otherwise 1:1), resized to 118x120; "
            "ref.BGR = observed-backdrop composite black_ref + obs_roi*(1 - alpha/255) "
            "in the 118x120 ROI before unwrapping (alpha==0 -> observed pixels, "
            "alpha==255 -> black-composited reference; rounded to uint8), "
            "ref.A = raw continuous alpha (0 = reference gap); both streams unwrapped "
            "at the ROI center; angle->x, radius->y)"
        )
    return (
        f"polar_unwrap_bgr_{IMG_W}x{IMG_H} (angle->x, 1 deg/column, clockwise, "
        "north at column 0; "
        "radius->y, inner at top)"
    )


def build_record(
    config: dict[str, Any],
    threads: int,
    device: torch.device,
    train_count: int,
    val_count: int,
    train_sha256: str,
    val_sha256: str,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "version": 31,
        "target_sigma": config["target_sigma"],
        "loss": (
            f"KL(q||p) between circular categorical distributions on Z/360Z, "
            f"q = wrapped gaussian pmf with sigma={config['target_sigma']:g} deg "
            "(= cross entropy minus constant target entropy H(q))"
        ),
        "input_shape": [input_channels(config["input_mode"]), IMG_H, IMG_W],
        "input_scaling": "BGR uint8 / 255",
        "input_mode": config["input_mode"],
        "input_representation": input_representation(config),
        "conv_padding_mode": "azimuth-circular; radius-zero (radius boundaries are ring-outside)",
        "seed": config["seed"],
        "threads": threads,
        "device": str(device),
        "precision": config["precision"],
        "compile": config["compile"],
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
    if config["input_mode"] == "ref":
        record["ref_reference_assets_root"] = config["map_assets_root"]
    return record
