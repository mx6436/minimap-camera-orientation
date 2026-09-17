"""torch 训练/评估循环：KL(q||p) 损失定义与逐 epoch 的训练、整集评估。

损失按样本权重 `w` 归一到 `Σw·KL / Σw`；val 侧权重恒为 1.0，退化为普通均值。
"""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from endfield.model import smoothed_targets
from endfield.train.metrics import Metrics, distribution_metrics


def autocast_context(device: torch.device, precision: str) -> AbstractContextManager[None]:
    """bf16 时返回 autocast 上下文，否则空操作。"""
    if precision == "bf16":
        return torch.autocast(device.type, dtype=torch.bfloat16)
    return nullcontext()


def loss_from_outputs(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    device: torch.device,
    sigma: float,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """逐样本 KL(q||p) 后取均值；给了 `weights` 则取 `Σw·KL / Σw`。

    `w = N` 与「把该样本复制 N 份后在扩容集合上取普通均值」逐个数值相等，这是加权
    语义的验收口径。
    """
    outputs = outputs.float()  # bf16 logits 转回 fp32：log_softmax 对 bf16 舍入敏感
    labels = smoothed_targets(targets.numpy(), sigma=sigma).to(device)
    log_probs = F.log_softmax(outputs, dim=-1)
    cross_entropy = -(labels * log_probs).sum(dim=-1)
    target_entropy = -(labels * torch.log(labels + 1e-12)).sum(dim=-1)
    # 交叉熵的下确界是目标分布自身的熵 H(q) > 0；减去这一定值即得
    # KL(q||p)，梯度不变而下确界为 0，loss 才能直接读作"距理想分布还差多少"
    per_sample = cross_entropy - target_entropy
    if weights is None:
        return per_sample.mean()
    weights = weights.to(device).float()
    return (per_sample * weights).sum() / weights.sum()


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    sigma: float,
    precision: str = "fp32",
) -> float:
    model.train()
    total = 0.0
    samples = 0.0
    for features, targets, weights in loader:
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, precision):
            outputs = model(features.to(device))
        loss = loss_from_outputs(outputs, targets, device, sigma, weights)
        loss.backward()
        optimizer.step()
        # 逐 batch 的 loss 已是加权均值，按权重和聚合成整轮加权均值
        batch_weight = float(weights.sum())
        total += loss.item() * batch_weight
        samples += batch_weight
    return total / samples


def eval_loss(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    sigma: float,
    precision: str = "fp32",
) -> tuple[float, Metrics]:
    model.eval()
    total = 0.0
    samples = 0
    probs_list: list[np.ndarray] = []
    targets_list: list[np.ndarray] = []
    with torch.no_grad():
        # val 侧权重恒为 1.0（没有困难目录），第三项不入损失
        for features, targets, _weights in loader:
            with autocast_context(device, precision):
                prediction = model(features.to(device))
            batch_size = len(features)
            total += loss_from_outputs(prediction, targets, device, sigma).item() * batch_size
            samples += batch_size
            probs_list.append(torch.softmax(prediction.float(), dim=1).cpu().numpy())
            targets_list.append(targets.numpy())
    target_array = np.concatenate(targets_list).astype(np.float64)
    return total / samples, distribution_metrics(np.concatenate(probs_list), target_array % 360.0)
