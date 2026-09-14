"""小地图采集与显示几何：图像 I/O、帧基准缩放与 ROI 参数。

`IMG_H` / `IMG_W` / `INNER_R` / `OUTER_R` 在此转发，供显示与数据校验使用。
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from endfield.preprocess import IMG_H, IMG_W, INNER_R, OUTER_R
from endfield.preprocess import ROI_CENTER as _ROI_CENTER

__all__ = [
    "BASE_SIZE",
    "IMG_H",
    "IMG_W",
    "INNER_R",
    "OUTER_R",
    "PNG_MAGIC",
    "imread_png",
    "load_source_frame",
    "scaled_roi",
]

# 采集几何（训练数据采集的 720p 基准）；ROI 中心由前处理定义持有
BASE_SIZE = (1280, 720)

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def imread_png(path: Path) -> np.ndarray:
    """PNG -> cv2 原生布局 BGR uint8 HWC；按 magic bytes 拒绝非 PNG 输入。"""
    with path.open("rb") as stream:
        if stream.read(8) != PNG_MAGIC:
            raise ValueError(f"{path}: expected PNG data")
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"{path}: cannot decode PNG image")
    return image


def load_source_frame(path: Path) -> np.ndarray:
    """原始截图 PNG -> 原样通道的 uint8 HWC frame（BGR 或 BGRA）。

    透明像素清零与 ROI 裁剪属前处理定义（`preprocess.observed_roi`）；本函数只解码并保留原通道。
    """
    image = imread_png(path)
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError(f"{path}: expected 3/4-channel PNG, got shape {image.shape}")
    return image


def scaled_roi(frame_shape: tuple[int, int]) -> tuple[float, float, float, float]:
    """非基准帧的显示/采集几何：把 ROI 中心与内外径按帧尺寸等比缩放。"""
    height, width = frame_shape[:2]
    sx, sy = width / BASE_SIZE[0], height / BASE_SIZE[1]
    # 非等比缩放会破坏环形状
    if abs(sx - sy) / max(sx, sy) > 0.01:
        print(f"WARNING: non-uniform scale sx={sx:.4f} sy={sy:.4f}; ring will be distorted")
    cx, cy = _ROI_CENTER[0] * sx, _ROI_CENTER[1] * sy
    return cx, cy, INNER_R * sx, OUTER_R * sx
