"""训练数据的通用小工具：PNG 解码、角度标注解析与样本名列举。"""

from __future__ import annotations

import random
import re
from pathlib import Path

import cv2
import numpy as np

from endfield.strip import REFERENCE_CHANNELS, STRIP_H, STRIP_W

ANGLE_RE = re.compile(r"_r(\d+(?:\.\d+)?)\.png$")
SEED = 42
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def parse_angle(path: Path) -> float:
    """文件名标注的角度可带一位小数。"""
    match = ANGLE_RE.search(path.name)
    if match is None:
        raise ValueError(f"cannot parse angle from filename: {path.name}")
    angle = float(match.group(1))
    if not 0 <= angle < 360:
        raise ValueError(f"angle out of range in filename: {path.name}")
    return angle


def imread_png(path: Path) -> np.ndarray:
    """PNG -> cv2 原生布局 BGR/BGRA uint8 HWC；按 magic bytes 拒绝非 PNG 输入。"""
    with path.open("rb") as stream:
        if stream.read(len(PNG_MAGIC)) != PNG_MAGIC:
            raise ValueError(f"{path}: expected PNG data")
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"{path}: cannot decode PNG image")
    return image


def load_bgr(path: Path) -> np.ndarray:
    image = imread_png(path)
    if image.shape != (STRIP_H, STRIP_W, 3):
        raise ValueError(f"{path}: expected {STRIP_W}x{STRIP_H} BGR, got shape {image.shape}")
    return image


def load_bgra(path: Path) -> np.ndarray:
    """BGRA PNG（ref 参考流布局：BGR = 白底合成参考，A = 原始连续 alpha）。"""
    image = imread_png(path)
    if image.shape != (STRIP_H, STRIP_W, REFERENCE_CHANNELS):
        raise ValueError(f"{path}: expected {STRIP_W}x{STRIP_H} BGRA, got shape {image.shape}")
    return image


def png_names(directory: Path) -> list[str]:
    return sorted(path.name for path in directory.glob("*.png"))


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
