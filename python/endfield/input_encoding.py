"""模型输入编码：条带域 -> 张量域（`[obs.BGR, ref.BGR, ref.A]` 与参考缺失占比）。

条带本身的采样由 MaaEnd C++（`CameraOrientationPreprocess` 的
`BuildOrientationStrips`）产出；本模块只做张量装配，训练读取共用。
"""

from __future__ import annotations

import numpy as np

from endfield.strip import OBSERVED_CHANNELS, REFERENCE_CHANNELS, STRIP_H, STRIP_W

# 参考条带语义的唯一文本：运行档案与交付图 metadata 共用，避免两处漂移
REFERENCE_STRIP_SEMANTICS = (
    "reference = MapLocator zone asset sampled once on the strip grid at "
    "(x,y)+(q_roi-pole)*scale with the zone's ZoneTemplateScale (ValleyIV_Base 15/16, "
    "otherwise 1:1); out-of-bounds reads 0 (reference gap); ref.BGR = "
    "rgb*(a/255) + 255*(1 - a/255) composited once in the strip domain over a white "
    "backdrop (transparent pixels, including out-of-bounds reads, are white), "
    "ref.A = raw continuous alpha; both streams defined by MaaEnd C++ "
    "CameraOrientationPreprocess (BuildOrientationStrips)"
)


def assemble_ref_pair(observed: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """观测条带与参考条带 -> 参考配对张量 `[obs.BGR, ref.BGR, ref.A]`。"""
    if observed.shape != (STRIP_H, STRIP_W, OBSERVED_CHANNELS):
        raise ValueError(
            f"observed strip must be {STRIP_H}x{STRIP_W}x{OBSERVED_CHANNELS}, got {observed.shape}"
        )
    if reference.shape != (STRIP_H, STRIP_W, REFERENCE_CHANNELS):
        raise ValueError(
            f"reference strip must be {STRIP_H}x{STRIP_W}x{REFERENCE_CHANNELS}, "
            f"got {reference.shape}"
        )
    return np.concatenate([observed, reference], axis=2)


def reference_gap_fraction(reference: np.ndarray) -> float:
    """参考条带的缺失像素占比：`ref.A < 255`（成片透明/越界与抗锯齿细边同计）。

    输入为 42x360x4 参考条带（各半径等权）；供训练时的样本过滤使用。
    """
    if reference.ndim != 3 or reference.shape[2] != REFERENCE_CHANNELS:
        raise ValueError(f"reference strip must be HxWx{REFERENCE_CHANNELS}, got {reference.shape}")
    return float(np.mean(reference[..., 3] < 255))
