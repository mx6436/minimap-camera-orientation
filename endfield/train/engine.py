"""torch 训练/评估循环：按架构分派的损失定义与逐 epoch 的训练、整集评估。"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from endfield.data_utils import circular_error
from endfield.model import (
    ARCHITECTURE_CONE,
    decode_logits,
    smoothed_targets,
    target_angles,
)
from endfield.train.metrics import metrics_from_outputs, norm_metrics


def combined_loss(outputs: torch.Tensor, targets: torch.Tensor, norm_lambda: float) -> torch.Tensor:
    """原始 sin/cos 上的 MSE，外加可选的二次模长锚。

    带锚时，方向对齐度为 c 的逐样本均衡模长为 r = (c + 2*lam) / (1 + 2*lam)：
    c = 1 时 r = 1，随 c 单调，完全不确定的样本下限为 2*lam/(1+2*lam)。"""
    mse = F.mse_loss(outputs, targets)
    if norm_lambda <= 0.0:
        return mse
    norms = torch.linalg.vector_norm(outputs, dim=-1)
    return mse + norm_lambda * ((norms - 1.0) ** 2).mean()


def loss_from_outputs(
    architecture: str,
    outputs: torch.Tensor,
    targets: torch.Tensor,
    norm_lambda: float,
    device: torch.device,
) -> torch.Tensor:
    if architecture == ARCHITECTURE_CONE:
        labels = smoothed_targets(target_angles(targets.numpy())).to(device)
        return F.cross_entropy(outputs, labels)
    return combined_loss(outputs, targets.to(device), norm_lambda)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    norm_lambda: float,
    device: torch.device,
    architecture: str,
) -> float:
    model.train()
    total = 0.0
    samples = 0
    for features, targets in loader:
        optimizer.zero_grad(set_to_none=True)
        outputs = model(features.to(device))
        loss = loss_from_outputs(architecture, outputs, targets, norm_lambda, device)
        loss.backward()
        optimizer.step()
        batch_size = len(features)
        total += loss.item() * batch_size
        samples += batch_size
    return total / samples


def eval_loss(
    model: nn.Module,
    loader: DataLoader,
    norm_lambda: float,
    device: torch.device,
    architecture: str,
) -> tuple[float, dict[str, float]]:
    model.eval()
    total = 0.0
    samples = 0
    outputs: list[np.ndarray] = []
    targets_list: list[np.ndarray] = []
    with torch.no_grad():
        for features, targets in loader:
            prediction = model(features.to(device))
            batch_size = len(features)
            total += (
                loss_from_outputs(architecture, prediction, targets, norm_lambda, device).item()
                * batch_size
            )
            samples += batch_size
            if architecture == ARCHITECTURE_CONE:
                decoded, _ = decode_logits(prediction)
                outputs.append(decoded)
            else:
                outputs.append(prediction.cpu().numpy())
            targets_list.append(targets.numpy())
    target_array = np.concatenate(targets_list).astype(np.float64)
    # 目标是单位圆上的 [sin, cos]，反解回的角度与文件名标注等价（往返误差 ~1e-5°）
    angles = np.degrees(np.arctan2(target_array[:, 0], target_array[:, 1])) % 360.0
    if architecture == ARCHITECTURE_CONE:
        decoded = np.concatenate(outputs)
        errors = circular_error(decoded, angles)
        return total / samples, {
            "circular_mae": float(np.mean(errors)),
            "circular_rmse": float(np.sqrt(np.mean(errors**2))),
            "circular_median": float(np.median(errors)),
            "within_1_degree": float(np.mean(errors <= 1.0)),
            "within_3_degrees": float(np.mean(errors <= 3.0)),
            "within_5_degrees": float(np.mean(errors <= 5.0)),
            "within_10_degrees": float(np.mean(errors <= 10.0)),
        }
    output_array = np.concatenate(outputs)
    metrics = metrics_from_outputs(output_array, angles)
    metrics.update(norm_metrics(output_array, target_array))
    return total / samples, metrics
