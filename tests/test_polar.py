"""polar.unwrap 展开约定的回归测试（约定见 CONTEXT.md「极坐标展开」与 polar.py）：

- 列 = 方位角：0° 正北为第 0 列，顺时针为正，1°/列；
- 行 = 半径：第 i 行像素中心对应 INNER_R + (i + 0.5)·step，内径在上。
"""

from __future__ import annotations

import numpy as np
import pytest

from polar import INNER_R, OUTER_R, unwrap

# 方位角取坐标轴方向、半径取整数：亮点像素坐标可精确计算，展开后该方位角
# 一列、亮点半径两侧两行上的采样点恰好命中亮点像素，能量（128）严格高于
# 其他任何格子（<=127），argmax 结果确定无误。
DOTS = (
    (0.0, 28.0),
    (90.0, 34.0),
    (180.0, 40.0),
    (270.0, 46.0),
)


def unwrap_dot(azimuth_deg: float, radius: float) -> np.ndarray:
    """在纯黑画布中心按约定画一个 (方位角, 半径) 亮点，返回展开图的每像素亮度。"""
    size = 224
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    cx = cy = size / 2.0
    px = int(round(cx + radius * np.sin(np.deg2rad(azimuth_deg))))
    py = int(round(cy - radius * np.cos(np.deg2rad(azimuth_deg))))
    canvas[py, px] = 255
    return unwrap(canvas, cx, cy, INNER_R, OUTER_R).sum(axis=2)


@pytest.mark.parametrize(("azimuth_deg", "radius"), DOTS)
def test_azimuth_maps_to_column(azimuth_deg: float, radius: float) -> None:
    brightness = unwrap_dot(azimuth_deg, radius)
    row = int(radius - INNER_R - 0.5)  # 亮点半径两侧两行之一，两行能量相同
    assert brightness[row].argmax() == int(azimuth_deg)


@pytest.mark.parametrize(("azimuth_deg", "radius"), DOTS)
def test_radius_maps_to_row(azimuth_deg: float, radius: float) -> None:
    brightness = unwrap_dot(azimuth_deg, radius)
    row = np.unravel_index(brightness.argmax(), brightness.shape)[0]
    assert abs(row - (radius - INNER_R - 0.5)) <= 0.5


def test_inner_radius_is_upper_row() -> None:
    small = unwrap_dot(0.0, 20.0)
    large = unwrap_dot(0.0, 50.0)
    assert small.argmax() < large.argmax()
