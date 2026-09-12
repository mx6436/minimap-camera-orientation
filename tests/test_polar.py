"""采集/显示几何的回归测试（条带几何与采样见 `tests/test_preprocess.py`）。

极坐标展开的几何与采样（列 = 方位角、行 = 半径）已由 #25 收拢到定义模块
`endfield/preprocess.py`；本文件只测本模块仍持有的采集几何：`scaled_roi` 的
等比缩放与非等比告警。
"""

from __future__ import annotations

from endfield.polar import BASE_SIZE, INNER_R, OUTER_R, ROI_CENTER, scaled_roi


def test_scaled_roi_passes_baseline_through() -> None:
    cx, cy, r_in, r_out = scaled_roi((BASE_SIZE[1], BASE_SIZE[0]))

    assert (cx, cy) == ROI_CENTER
    assert (r_in, r_out) == (INNER_R, OUTER_R)


def test_scaled_roi_scales_center_and_radii() -> None:
    cx, cy, r_in, r_out = scaled_roi((1080, 1920))

    assert (cx, cy) == (ROI_CENTER[0] * 1.5, ROI_CENTER[1] * 1.5)
    assert (r_in, r_out) == (INNER_R * 1.5, OUTER_R * 1.5)


def test_scaled_roi_warns_on_non_uniform_scale(capsys) -> None:
    scaled_roi((720, 2560))

    assert "non-uniform scale" in capsys.readouterr().out
