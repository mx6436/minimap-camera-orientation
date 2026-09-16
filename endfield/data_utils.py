from __future__ import annotations

import random
import re
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from endfield.polar import imread_png
from endfield.preprocess import IMG_H, IMG_W

ANGLE_RE = re.compile(r"_r(\d+(?:\.\d+)?)\.png$")
SEED = 42


def parse_angle(path: Path) -> float:
    """文件名标注的角度可带一位小数。"""
    match = ANGLE_RE.search(path.name)
    if match is None:
        raise ValueError(f"cannot parse angle from filename: {path.name}")
    angle = float(match.group(1))
    if not 0 <= angle < 360:
        raise ValueError(f"angle out of range in filename: {path.name}")
    return angle


def circular_error(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.abs((predicted - target + 180.0) % 360.0 - 180.0)


def load_bgr(path: Path) -> np.ndarray:
    image = imread_png(path)
    if image.shape != (IMG_H, IMG_W, 3):
        raise ValueError(f"{path}: expected {IMG_W}x{IMG_H} BGR, got shape {image.shape}")
    return image


def load_bgra(path: Path) -> np.ndarray:
    """BGRA PNG（ref 参考流布局：BGR = 白底合成参考，A = 原始连续 alpha）。"""
    image = imread_png(path)
    if image.shape != (IMG_H, IMG_W, 4):
        raise ValueError(f"{path}: expected {IMG_W}x{IMG_H} BGRA, got shape {image.shape}")
    return image


def png_names(directory: Path) -> list[str]:
    return sorted(path.name for path in directory.glob("*.png"))


def union_png_samples(directories: Iterable[Path]) -> dict[str, Path]:
    """原始目录并集 -> {样本名: 源文件}。"""
    samples: dict[str, Path] = {}
    for directory in directories:
        for path in sorted(directory.glob("*.png")):
            if path.name in samples:
                raise SystemExit(f"sample name in both raw dirs: {path.name}")
            samples[path.name] = path
    return samples


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
