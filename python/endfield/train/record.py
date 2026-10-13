"""运行档案的写侧 adapter：训练配置与环境事实 -> RunRecord（核心字段 + 复现 metadata）。

envelope 与核心字段由 `endfield/run_record.py` 持有并落盘；本模块只把训练侧的配置、
环境与数据指纹翻译成 metadata。训练固定为 ref 单模式，预处理定义的权威在 MaaEnd C++
（`CameraOrientationPreprocess.cpp`）。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from endfield.input_encoding import REFERENCE_STRIP_SEMANTICS
from endfield.run_record import RunRecord
from endfield.strip import INPUT_CHANNELS, STRIP_H, STRIP_W
from endfield.train.data import names_fingerprint


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
    gap_filter = ""
    if config["max_ref_missing"] is not None:
        gap_filter = (
            f"; train samples filtered to reference-gap fraction "
            f"(ref.A<255) <= {config['max_ref_missing']:g}"
        )
    return (
        f"ref_polar_unwrap_obs_bgr_ref_bgra_{STRIP_W}x{STRIP_H} "
        f"(channels = [obs.BGR, ref.BGR, ref.A]; {REFERENCE_STRIP_SEMANTICS}; "
        f"angle->x, radius->y{gap_filter})"
    )


def build_record(
    config: dict[str, Any],
    threads: int,
    device: torch.device,
    train_count: int,
    val_count: int,
    train_sha256: str,
    val_sha256: str,
    trainable_parameters: int,
    hard_names: Sequence[str] = (),
) -> RunRecord:
    """配置、环境与数据指纹 -> 运行档案。

    `hard_names` 是**落在训练划分里**的困难样本（调用方已与训练名单求交）：它们是
    `hard_weight` 实际作用到的样本，因此档案只记这一集合的数量与摘要。
    """
    metadata: dict[str, Any] = {
        "loss": (
            f"KL(q||p) between circular categorical distributions on Z/360Z, "
            f"q = wrapped gaussian pmf with sigma={config['target_sigma']:g} deg "
            "(= cross entropy minus constant target entropy H(q)); "
            "train loss is sample-weighted by data/hard_raw membership "
            "(w = hard_weight for those samples, 1 otherwise; sum(w*KL)/sum(w)); "
            "val loss and metrics are unweighted"
        ),
        "input_shape": [INPUT_CHANNELS, STRIP_H, STRIP_W],
        "input_scaling": "BGR uint8 / 255",
        "input_representation": input_representation(config),
        "max_ref_missing": config["max_ref_missing"],
        "conv_padding_mode": "azimuth-circular; radius-zero (radius boundaries are ring-outside)",
        "seed": config["seed"],
        "threads": threads,
        "device": str(device),
        "precision": config["precision"],
        "compile": config["compile"],
        "model": "AzimuthNet",
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
        "hard_weight": float(config["hard_weight"]),
        "hard_count": len(hard_names),
        "hard_names_sha256": names_fingerprint(list(hard_names)),
    }
    return RunRecord(
        target_sigma=float(config["target_sigma"]),
        trainable_parameters=trainable_parameters,
        metadata=metadata,
    )
