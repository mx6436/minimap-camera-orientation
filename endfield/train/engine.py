"""torch 训练/评估循环：KL(q||p) 损失定义与逐 epoch 的训练、整集评估。"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from endfield.model import smoothed_targets, target_angles
from endfield.train.metrics import distribution_metrics


def loss_from_outputs(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    device: torch.device,
    sigma: float,
) -> torch.Tensor:
    labels = smoothed_targets(target_angles(targets.numpy()), sigma=sigma).to(device)
    log_probs = F.log_softmax(outputs, dim=-1)
    cross_entropy = -(labels * log_probs).sum(dim=-1)
    target_entropy = -(labels * torch.log(labels + 1e-12)).sum(dim=-1)
    # 交叉熵的下确界是目标分布自身的熵 H(q) > 0；减去这一定值即得
    # KL(q||p)，梯度不变而下确界为 0，loss 才能直接读作"距理想分布还差多少"
    return (cross_entropy - target_entropy).mean()


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    sigma: float,
) -> float:
    model.train()
    total = 0.0
    samples = 0
    for features, targets in loader:
        optimizer.zero_grad(set_to_none=True)
        outputs = model(features.to(device))
        loss = loss_from_outputs(outputs, targets, device, sigma)
        loss.backward()
        optimizer.step()
        batch_size = len(features)
        total += loss.item() * batch_size
        samples += batch_size
    return total / samples


def eval_loss(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    sigma: float,
) -> tuple[float, dict[str, float]]:
    model.eval()
    total = 0.0
    samples = 0
    probs_list: list[np.ndarray] = []
    targets_list: list[np.ndarray] = []
    with torch.no_grad():
        for features, targets in loader:
            prediction = model(features.to(device))
            batch_size = len(features)
            total += loss_from_outputs(prediction, targets, device, sigma).item() * batch_size
            samples += batch_size
            probs_list.append(torch.softmax(prediction, dim=1).cpu().numpy())
            targets_list.append(targets.numpy())
    target_array = np.concatenate(targets_list).astype(np.float64)
    # 目标是单位圆上的 [sin, cos]，反解回的角度与文件名标注等价（往返误差 ~1e-5°）
    angles = np.degrees(np.arctan2(target_array[:, 0], target_array[:, 1])) % 360.0
    return total / samples, distribution_metrics(np.concatenate(probs_list), angles)
