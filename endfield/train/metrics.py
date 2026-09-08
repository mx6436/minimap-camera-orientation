"""验证指标：在循环组 Z/360Z 上的概率质量函数直接评价"""

from __future__ import annotations

import numpy as np


def distribution_metrics(probs: np.ndarray, angles: np.ndarray) -> dict[str, float]:
    """分布级指标：在概率质量函数上直接评价，不经过 argmax 解码。

    期望同时遍及验证样本与 PMF：expected_abs_error = E[|Δθ|]，
    rms_error = sqrt(E[Δθ²])。
    """
    if probs.shape[-1] != 360:
        raise ValueError(f"expected final dimension of 360, got {probs.shape}")
    diff = (np.arange(360.0) - angles[:, None] + 180.0) % 360.0 - 180.0
    dist = np.abs(diff)
    log_p = np.log(probs + 1e-12)
    return {
        "expected_abs_error": float(np.mean(np.sum(probs * dist, axis=1))),
        "rms_error": float(np.sqrt(np.mean(np.sum(probs * diff**2, axis=1)))),
        "entropy": float(np.mean(-np.sum(probs * log_p, axis=1))),
        "target_mass_5deg": float(np.mean(np.sum(probs * (dist <= 5.0), axis=1))),
        "peak_prob": float(np.mean(probs.max(axis=1))),
    }
