"""采集/显示几何的回归测试：`scaled_roi` 的等比缩放与非等比告警。

条带几何与采样见 `tests/test_preprocess.py`。
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
