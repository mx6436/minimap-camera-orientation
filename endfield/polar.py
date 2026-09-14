"""小地图采集与显示几何：图像 I/O、帧基准缩放与 ROI 参数。

ROI 与条带几何（尺寸、内外径、中心）由前处理定义模块持有；本模块只 import 自用，
不转发。需要这些常量的调用方直接 `from endfield.preprocess import ...`。
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from endfield import preprocess

__all__ = [
    "BASE_SIZE",
    "PNG_MAGIC",
    "imread_png",
    "load_source_frame",
    "scaled_roi",
    "to_base_frame",
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


def to_base_frame(frame: np.ndarray) -> np.ndarray:
    """任意分辨率帧 -> 1280x720 训练基准；已是基准则原样返回（不复制）。"""
    width, height = BASE_SIZE
    if frame.shape[1] == width and frame.shape[0] == height:
        return frame
    interpolation = cv2.INTER_AREA if frame.shape[1] > width else cv2.INTER_LINEAR
    return cv2.resize(frame, BASE_SIZE, interpolation=interpolation)


def scaled_roi(frame_shape: tuple[int, int]) -> tuple[float, float, float, float]:
    """非基准帧的显示/采集几何：把 ROI 中心与内外径按帧尺寸等比缩放。"""
    height, width = frame_shape[:2]
    sx, sy = width / BASE_SIZE[0], height / BASE_SIZE[1]
    # 非等比缩放会破坏环形状
    if abs(sx - sy) / max(sx, sy) > 0.01:
        print(f"WARNING: non-uniform scale sx={sx:.4f} sy={sy:.4f}; ring will be distorted")
    cx, cy = preprocess.ROI_CENTER[0] * sx, preprocess.ROI_CENTER[1] * sy
    return cx, cy, preprocess.INNER_R * sx, preprocess.OUTER_R * sx
