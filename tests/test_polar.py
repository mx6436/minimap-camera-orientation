"""采集/显示几何的回归测试：帧基准缩放与 `scaled_roi` 的等比缩放与非等比告警。

条带几何与采样见 `tests/test_preprocess.py`。
"""

from __future__ import annotations

import numpy as np

from endfield.polar import BASE_SIZE, scaled_roi, to_base_frame
from endfield.preprocess import INNER_R, OUTER_R, ROI_CENTER


def test_scaled_roi_passes_baseline_through() -> None:
    cx, cy, r_in, r_out = scaled_roi((BASE_SIZE[1], BASE_SIZE[0]))

    assert (cx, cy) == ROI_CENTER
    assert (r_in, r_out) == (INNER_R, OUTER_R)


def test_scaled_roi_scales_center_and_radii() -> None:
    cx, cy, r_in, r_out = scaled_roi((1080, 1920))

    assert (cx, cy) == (ROI_CENTER[0] * 1.5, ROI_CENTER[1] * 1.5)
    assert (r_in, r_out) == (INNER_R * 1.5, OUTER_R * 1.5)


def test_to_base_frame_scales_non_base_frames_to_720p() -> None:
    frame = np.zeros((360, 640, 3), dtype=np.uint8)

    scaled = to_base_frame(frame)

    assert scaled.shape == (BASE_SIZE[1], BASE_SIZE[0], 3)
    assert scaled.dtype == np.uint8


def test_to_base_frame_passes_baseline_through_without_copy() -> None:
    frame = np.zeros((BASE_SIZE[1], BASE_SIZE[0], 3), dtype=np.uint8)

    assert to_base_frame(frame) is frame
