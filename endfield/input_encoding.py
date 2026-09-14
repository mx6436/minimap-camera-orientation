"""模型输入编码：条带域 -> 张量域（`[obs.BGR, ref.BGR, ref.A]` 与参考缺失占比）。

条带本身的采样在 `placement`（底图定位面）；本模块只做张量装配，训练读取与实机
推理共用（ADR 0004）。
"""

from __future__ import annotations

import numpy as np

from endfield.preprocess import IMG_H, IMG_W


def assemble_ref_pair(observed: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """观测条带与参考条带 -> 参考配对张量 `[obs.BGR, ref.BGR, ref.A]`。"""
    if observed.shape != (IMG_H, IMG_W, 3):
        raise ValueError(f"observed strip must be {IMG_H}x{IMG_W}x3, got {observed.shape}")
    if reference.shape != (IMG_H, IMG_W, 4):
        raise ValueError(f"reference strip must be {IMG_H}x{IMG_W}x4, got {reference.shape}")
    return np.concatenate([observed, reference], axis=2)


def reference_gap_fraction(reference: np.ndarray) -> float:
    """参考条带的缺失像素占比：`ref.A < 255`（成片透明/越界与抗锯齿细边同计）。

    输入为 42x360x4 参考条带（各半径等权）；供训练时的样本过滤使用。
    """
    if reference.ndim != 3 or reference.shape[2] != 4:
        raise ValueError(f"reference strip must be HxWx4, got {reference.shape}")
    return float(np.mean(reference[..., 3] < 255))
