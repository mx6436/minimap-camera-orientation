"""极坐标展开几何：环形小地图 -> 360x44 RGB 模型输入的唯一共享实现。

约定（见 CONTEXT.md「极坐标展开」词条）：
- 极点为 ROI 中心，角度零点为正北，顺时针为正；
- 角度 -> x 轴：第 j 列的像素中心对应方位角 j 度（1°/列，x=0 是正北，
  0/360 接缝位于第 359 列与第 0 列之间）；
- 半径 -> y 轴：第 i 行的像素中心对应半径 r_in + (i + 0.5)·step
  （step = (r_out - r_in) / IMG_H；基准分辨率下 step=1，即内径 12、
  外径 56 之间 1:1 采样，内径在上）；
- 输出像素由源图 (cx + r·sin θ, cy − r·cos θ) 处双线性采样得到（反向
  映射）。输出无透明区域，天然全有效；全部行半径落在环内，位于中心
  圆内的箭头被排除在输入之外。

prepare_data.py 在原始分辨率上调用；live.py / predict.py 在按基准分辨率
等比缩放后的浮点 ROI 上调用。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

# ROI 几何（训练数据采集的 720p 基准）
BASE_SIZE = (1280, 720)
ROI_CENTER = (108.0, 111.0)
INNER_R = 12.0
OUTER_R = 56.0

IMG_W = 360
IMG_H = 44


def load_source_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        if im.format != "PNG":
            raise ValueError(f"{path}: expected PNG data, got {im.format}")
        rgba = np.asarray(im.convert("RGBA"), dtype=np.uint8).copy()
    rgba[rgba[..., 3] == 0, :3] = 0
    return rgba[..., :3]


def unwrap(rgb: np.ndarray, cx: float, cy: float, r_in: float, r_out: float) -> np.ndarray:
    height, width = rgb.shape[:2]
    step = (r_out - r_in) / IMG_H
    # 双线性插值需要在源图内取到邻居像素：要求整个圆盘加一圈邻居都在图内。
    if not (r_out + 1 <= cx <= width - r_out - 1 and r_out + 1 <= cy <= height - r_out - 1):
        raise ValueError(
            f"image {width}x{height} too small for ROI center "
            f"({cx}, {cy}) with outer radius {r_out}"
        )

    radii = r_in + step * (np.arange(IMG_H, dtype=np.float64) + 0.5)
    theta = np.deg2rad(np.arange(IMG_W, dtype=np.float64))
    src_x = cx + radii[:, None] * np.sin(theta)[None, :]
    src_y = cy - radii[:, None] * np.cos(theta)[None, :]

    x0 = np.floor(src_x).astype(np.int64)
    y0 = np.floor(src_y).astype(np.int64)
    fx = (src_x - x0)[..., None]
    fy = (src_y - y0)[..., None]
    x0c = np.clip(x0, 0, width - 1)
    x1c = np.clip(x0 + 1, 0, width - 1)
    y0c = np.clip(y0, 0, height - 1)
    y1c = np.clip(y0 + 1, 0, height - 1)

    top = rgb[y0c, x0c] * (1.0 - fx) + rgb[y0c, x1c] * fx
    bottom = rgb[y1c, x0c] * (1.0 - fx) + rgb[y1c, x1c] * fx
    out = top * (1.0 - fy) + bottom * fy
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def scaled_roi(frame_shape: tuple[int, int]) -> tuple[float, float, float, float]:
    height, width = frame_shape[:2]
    sx, sy = width / BASE_SIZE[0], height / BASE_SIZE[1]
    # 非等比缩放会破坏环形状
    if abs(sx - sy) / max(sx, sy) > 0.01:
        print(f"WARNING: non-uniform scale sx={sx:.4f} sy={sy:.4f}; ring will be distorted")
    cx, cy = ROI_CENTER[0] * sx, ROI_CENTER[1] * sy
    return cx, cy, INNER_R * sx, OUTER_R * sx


def self_check_azimuth() -> None:
    size = 224
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    cx, cy = size / 2.0, size / 2.0
    # 每个方位角用唯一半径，避免不同亮点落在同一展开行上互相干扰
    dots = ((0.0, 28.0), (90.0, 34.0), (180.0, 40.0), (270.0, 46.0))
    for azimuth_deg, radius in dots:
        px = int(round(cx + radius * np.sin(np.deg2rad(azimuth_deg))))
        py = int(round(cy - radius * np.cos(np.deg2rad(azimuth_deg))))
        canvas[py, px] = 255
    out = unwrap(canvas, cx, cy, INNER_R, OUTER_R)
    for azimuth_deg, radius in dots:
        row = int(round(radius - 0.5 - INNER_R))
        col = int(out[row].sum(axis=1).argmax())
        if abs(col - azimuth_deg) > 1:
            raise RuntimeError(
                f"azimuth self-check failed: dot at {azimuth_deg}° landed in column {col}"
            )
    print("azimuth self-check ok: 0°->col 0 (north), clockwise, 1°/column")
