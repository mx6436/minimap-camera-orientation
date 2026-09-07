"""极坐标展开几何：环形小地图 -> 360x42 BGR 模型输入的唯一共享实现。

约定（见 CONTEXT.md「极坐标展开」词条）：
- 极点为 ROI 中心，角度零点为正北，顺时针为正；
- 角度 -> x 轴：第 j 列的像素中心对应方位角 j 度（1°/列，x=0 是正北，
  0/360 接缝位于第 359 列与第 0 列之间）；
- 半径 -> y 轴：第 i 行的像素中心对应半径 r_in + (i + 0.5)·step
  （step = (r_out - r_in) / IMG_H；基准分辨率下 step=1，即内径 12、
  外径 54 之间 1:1 采样，内径在上）；
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

# ROI 几何（训练数据采集的 720p 基准）
BASE_SIZE = (1280, 720)
ROI_CENTER = (108.0, 111.0)
INNER_R = 12.0
OUTER_R = 54.0

IMG_W = 360
IMG_H = 42

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


def load_source_bgr(path: Path) -> np.ndarray:
    """原始截图 PNG -> BGR uint8 HWC；全透明像素置 0，保留原 RGBA 约定。"""
    image = imread_png(path)
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError(f"{path}: expected 3/4-channel PNG, got shape {image.shape}")
    if image.shape[2] == 4:
        image[image[..., 3] == 0, :3] = 0
        image = image[..., :3]
    return image


def unwrap(bgr: np.ndarray, cx: float, cy: float, r_in: float, r_out: float) -> np.ndarray:
    height, width = bgr.shape[:2]
    step = (r_out - r_in) / IMG_H
    # BORDER_REPLICATE 只是兜底；圆盘加一圈邻居必须在图内，否则说明 ROI
    # 几何本身错了，直接失败。
    if not (r_out + 1 <= cx <= width - r_out - 1 and r_out + 1 <= cy <= height - r_out - 1):
        raise ValueError(
            f"image {width}x{height} too small for ROI center "
            f"({cx}, {cy}) with outer radius {r_out}"
        )

    radii = r_in + step * (np.arange(IMG_H, dtype=np.float32) + 0.5)
    theta = np.deg2rad(np.arange(IMG_W, dtype=np.float32))
    map_x = cx + radii[:, None] * np.sin(theta)[None, :]
    map_y = cy - radii[:, None] * np.cos(theta)[None, :]
    return cv2.remap(
        bgr,
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def scaled_roi(frame_shape: tuple[int, int]) -> tuple[float, float, float, float]:
    height, width = frame_shape[:2]
    sx, sy = width / BASE_SIZE[0], height / BASE_SIZE[1]
    # 非等比缩放会破坏环形状
    if abs(sx - sy) / max(sx, sy) > 0.01:
        print(f"WARNING: non-uniform scale sx={sx:.4f} sy={sy:.4f}; ring will be distorted")
    cx, cy = ROI_CENTER[0] * sx, ROI_CENTER[1] * sy
    return cx, cy, INNER_R * sx, OUTER_R * sx
