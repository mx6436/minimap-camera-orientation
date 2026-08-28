#!/usr/bin/env python3
"""Shared image, angle, split, and JSON helpers for the training tools."""
from __future__ import annotations

import json
import math
import os
import random
import re
import tempfile
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

ANGLE_RE = re.compile(r"_r(\d+(?:\.\d+)?)\.png$")
SEED = 42
IMAGE_SIZE = (112, 112)
POLAR_IMAGE_SIZE = (360, 44)  # PIL (width, height)：1°/列，1px/行


def parse_angle(path: Path) -> float:
    """从文件名的 `_r<角度>.png` 后缀解析角度；标注可带一位小数。"""
    match = ANGLE_RE.search(path.name)
    if match is None:
        raise ValueError(f"cannot parse angle from filename: {path.name}")
    angle = float(match.group(1))
    if not 0 <= angle < 360:
        raise ValueError(f"angle out of range in filename: {path.name}")
    return angle


def angle_target(angle: float) -> np.ndarray:
    radians = math.radians(angle)
    return np.asarray([math.sin(radians), math.cos(radians)], dtype=np.float32)


def circular_error(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.abs((predicted - target + 180.0) % 360.0 - 180.0)


def decode_angle(outputs: np.ndarray) -> np.ndarray:
    outputs = np.asarray(outputs, dtype=np.float64)
    if outputs.shape[-1] != 2:
        raise ValueError(f"expected final dimension of 2, got {outputs.shape}")
    norms = np.linalg.norm(outputs, axis=-1, keepdims=True)
    normalized = outputs / np.maximum(norms, 1e-8)
    angles = np.degrees(np.arctan2(normalized[..., 0], normalized[..., 1])) % 360.0
    return angles


def round_angle(angles: np.ndarray | float) -> np.ndarray | int:
    rounded = np.floor(np.asarray(angles, dtype=np.float64) + 0.5).astype(np.int64) % 360
    return int(rounded) if rounded.ndim == 0 else rounded


def load_rgba(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        if image.format != "PNG":
            raise ValueError(f"{path}: expected PNG, got {image.format}")
        if image.size != IMAGE_SIZE:
            raise ValueError(f"{path}: expected 112x112, got {image.size}")
        if image.mode != "RGBA":
            raise ValueError(f"{path}: expected RGBA, got {image.mode}")
        array = np.asarray(image, dtype=np.uint8).copy()
    array[array[..., 3] == 0, :3] = 0
    return array


def load_rgb(path: Path) -> np.ndarray:
    """加载极坐标展开 RGB 样本（360x44，无透明区域）。"""
    with Image.open(path) as image:
        if image.format != "PNG":
            raise ValueError(f"{path}: expected PNG, got {image.format}")
        if image.size != POLAR_IMAGE_SIZE:
            raise ValueError(f"{path}: expected 360x44, got {image.size}")
        if image.mode != "RGB":
            raise ValueError(f"{path}: expected RGB, got {image.mode}")
        return np.asarray(image, dtype=np.uint8).copy()


def atomic_json_dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=True)
            stream.write("\n")
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def png_names(directory: Path) -> list[str]:
    return sorted(path.name for path in directory.glob("*.png"))


def validate_manifest_names(names: Iterable[str], available: set[str], label: str) -> list[str]:
    values = list(names)
    if len(values) != len(set(values)):
        raise ValueError(f"{label} manifest contains duplicate filenames")
    if not set(values).issubset(available):
        missing = sorted(set(values) - available)
        raise ValueError(f"{label} manifest references missing files: {missing[:5]}")
    return values


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
