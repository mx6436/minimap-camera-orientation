"""运行档案的写侧 adapter：训练配置与环境事实 -> RunRecord（核心字段 + 复现 metadata）。

envelope 与核心字段由 `endfield/run_record.py` 持有并落盘；本模块只把训练侧的配置、
环境与数据指纹翻译成 metadata。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from endfield.polar import IMG_H, IMG_W
from endfield.run_record import InputMode, RunRecord, input_channels


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
    if InputMode.parse(config["input_mode"]) is InputMode.REF:
        gap_filter = ""
        if config["max_ref_missing"] is not None:
            gap_filter = (
                f"; train samples filtered to reference-gap fraction "
                f"(ref.A<255) <= {config['max_ref_missing']:g}"
            )
        return (
            f"ref_polar_unwrap_obs_bgr_ref_bgra_{IMG_W}x{IMG_H} "
            "(channels = [obs.BGR, ref.BGR, ref.A]; reference = MapLocator zone asset "
            "sampled once on the strip grid at (x,y)+(q_roi-pole)*scale with the zone's "
            "MapLocator ZoneTemplateScale (ValleyIV_Base 15/16, otherwise 1:1); "
            "out-of-bounds reads 0 = reference gap; "
            "ref.BGR = rgb*(a/255) + obs*(1 - a/255) composited once in the strip domain "
            "(alpha==0 -> observed pixels), ref.A = raw continuous alpha; "
            "both streams defined by endfield/preprocess.py; "
            f"angle->x, radius->y{gap_filter})"
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
    trainable_parameters: int,
    assets_root: str | None = None,
) -> RunRecord:
    """配置、环境与数据指纹 -> 运行档案；ref 必须给出数据集所用的资产根。"""
    mode = InputMode.parse(config["input_mode"])
    metadata: dict[str, Any] = {
        "loss": (
            f"KL(q||p) between circular categorical distributions on Z/360Z, "
            f"q = wrapped gaussian pmf with sigma={config['target_sigma']:g} deg "
            "(= cross entropy minus constant target entropy H(q))"
        ),
        "input_shape": [input_channels(mode), IMG_H, IMG_W],
        "input_scaling": "BGR uint8 / 255",
        "input_representation": input_representation(config),
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
    }
    if mode is InputMode.REF:
        if assets_root is None:
            raise SystemExit(
                "ref record requires the dataset's reference assets root; "
                "run prepare_data.py --mode ref to write the stamp"
            )
        metadata["max_ref_missing"] = config["max_ref_missing"]
    return RunRecord(
        input_mode=mode,
        assets_root=Path(assets_root) if assets_root is not None else None,
        target_sigma=float(config["target_sigma"]),
        trainable_parameters=trainable_parameters,
        metadata=metadata,
    )
