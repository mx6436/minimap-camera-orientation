"""验证指标：循环角误差汇总与模长诊断（纯 numpy，不触 torch）。"""

from __future__ import annotations

import numpy as np

from endfield.data_utils import circular_error, decode_angle


def _rank_data(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2:
        return 0.0
    return float(np.corrcoef(_rank_data(a), _rank_data(b))[0, 1])


def distribution_metrics(probs: np.ndarray, angles: np.ndarray) -> dict[str, float]:
    """分布级指标：在概率质量函数上直接评价，不经过 argmax 解码。"""
    if probs.shape[-1] != 360:
        raise ValueError(f"expected final dimension of 360, got {probs.shape}")
    diff = (np.arange(360.0) - angles[:, None] + 180.0) % 360.0 - 180.0
    dist = np.abs(diff)
    log_p = np.log(probs + 1e-12)
    return {
        "expected_mae": float(np.mean(np.sum(probs * dist, axis=1))),
        "expected_rmse": float(np.sqrt(np.mean(np.sum(probs * diff**2, axis=1)))),
        "entropy": float(np.mean(-np.sum(probs * log_p, axis=1))),
        "target_mass_5deg": float(np.mean(np.sum(probs * (dist <= 5.0), axis=1))),
        "peak_prob": float(np.mean(probs.max(axis=1))),
    }


def norm_metrics(outputs: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    """模长统计 + 原始 MSE 的精确加法分解。

    |v-y|^2 = (||v||-1)^2 + 2*||v||*(1-cos dtheta)；两项均非负，因此
    norm_err_share = mean((||v||-1)^2) / mean(|v-y|^2) 是精确分解。

    v 与单位目标向量的夹角等于解码后的循环误差，故 arccos 对齐角可兼作
    逐样本误差，供按模长表盘追踪。"""
    norms = np.linalg.norm(outputs, axis=-1)
    raw_mse = np.sum((outputs - targets) ** 2, axis=-1)
    total = float(np.mean(raw_mse))
    align = np.sum(outputs * targets, axis=-1) / np.maximum(norms, 1e-12)
    angular = np.degrees(np.arccos(np.clip(align, -1.0, 1.0)))
    low = norms <= np.percentile(norms, 20)
    return {
        "norm_mean": float(np.mean(norms)),
        "norm_p5": float(np.percentile(norms, 5)),
        "norm_p95": float(np.percentile(norms, 95)),
        "raw_mse_total": total,
        "norm_err_share": float(np.mean((norms - 1.0) ** 2) / total) if total > 0 else 0.0,
        "spearman_norm_err": _spearman(norms, angular),
        "norm_low20_mae": float(np.mean(angular[low])),
        "norm_low20_gt10_share": float(np.mean(angular[low] > 10.0)),
    }


def metrics_from_outputs(outputs: np.ndarray, angles: np.ndarray) -> dict[str, float]:
    predicted = decode_angle(outputs)
    errors = circular_error(predicted, angles)
    return {
        "circular_mae": float(np.mean(errors)),
        "circular_rmse": float(np.sqrt(np.mean(errors**2))),
        "circular_median": float(np.median(errors)),
        "within_1_degree": float(np.mean(errors <= 1.0)),
        "within_3_degrees": float(np.mean(errors <= 3.0)),
        "within_5_degrees": float(np.mean(errors <= 5.0)),
        "within_10_degrees": float(np.mean(errors <= 10.0)),
    }
