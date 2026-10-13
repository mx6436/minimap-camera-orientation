"""验证指标：在循环组 Z/360Z 上的概率质量函数直接评价"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any, ClassVar

import numpy as np


@dataclass(frozen=True)
class Metrics:
    """验证集上的分布指标：在概率质量函数上直接评价，不经过 argmax 解码。

    磁盘键（`summary.json` / `history.json` 的 `val_*` 字段）由本模块的 `to_payload` /
    `from_payload` 产生与读取，指标名不在别处拼写。
    """

    expected_abs_error: float = 0.0
    rms_error: float = 0.0
    entropy: float = 0.0
    target_mass_5deg: float = 0.0
    peak_prob: float = 0.0

    # 训练跟踪与早停所用的指标（唯一一处指标名）
    TRACK_FIELD: ClassVar[str] = "rms_error"
    # 磁盘键的前缀
    PREFIX: ClassVar[str] = "val_"

    @classmethod
    def payload_key(cls, name: str) -> str:
        """指标字段名 -> 磁盘键（如 `rms_error` -> `val_rms_error`）。"""
        return f"{cls.PREFIX}{name}"

    def to_payload(self) -> dict[str, float]:
        """指标 -> 磁盘键值对（`val_expected_abs_error` 等）。"""
        return {self.payload_key(item.name): getattr(self, item.name) for item in fields(self)}

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> Metrics:
        """磁盘键值对 -> 指标；缺失或非数值的诊断指标按 0.0（契约字段的强制在 `run_dir`）。"""
        values: dict[str, float] = {}
        for item in fields(cls):
            value = payload.get(cls.payload_key(item.name))
            numeric = isinstance(value, (int, float)) and not isinstance(value, bool)
            values[item.name] = float(value) if numeric else 0.0
        return cls(**values)


def distribution_metrics(probs: np.ndarray, angles: np.ndarray) -> Metrics:
    """分布级指标：期望同时遍及验证样本与 PMF。

    expected_abs_error = E[|Δθ|]，rms_error = sqrt(E[Δθ²])。
    """
    if probs.shape[-1] != 360:
        raise ValueError(f"expected final dimension of 360, got {probs.shape}")
    diff = (np.arange(360.0) - angles[:, None] + 180.0) % 360.0 - 180.0
    dist = np.abs(diff)
    log_p = np.log(probs + 1e-12)
    return Metrics(
        expected_abs_error=float(np.mean(np.sum(probs * dist, axis=1))),
        rms_error=float(np.sqrt(np.mean(np.sum(probs * diff**2, axis=1)))),
        entropy=float(np.mean(-np.sum(probs * log_p, axis=1))),
        target_mass_5deg=float(np.mean(np.sum(probs * (dist <= 5.0), axis=1))),
        peak_prob=float(np.mean(probs.max(axis=1))),
    )
