"""训练/评估循环的权重口径：`w = N` 与「把样本复制 N 份」逐个数值相等。"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from endfield.train.engine import eval_loss, loss_from_outputs, train_epoch

BINS = 360
SIGMA = 3.0
DEVICE = torch.device("cpu")
TARGETS = torch.tensor([10.0, 200.0])


def sample_logits(count: int = 2, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(count, BINS, generator=generator)


class FixedLogits(nn.Module):
    """把 batch 的特征当作 logits 原样返回；挂一个零参数让 backward/step 可用。"""

    def __init__(self) -> None:
        super().__init__()
        self.zero = nn.Parameter(torch.zeros(1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features + self.zero * 0


def per_sample_kl(logits: torch.Tensor, targets: torch.Tensor) -> list[float]:
    """逐样本 KL：单样本 batch 取无权重均值即该样本的 KL。"""
    return [
        loss_from_outputs(logits[i : i + 1], targets[i : i + 1], DEVICE, SIGMA).item()
        for i in range(len(targets))
    ]


def test_weighted_loss_matches_duplicating_the_sample() -> None:
    """一条样本权重 5：损失 = 该样本复制 5 份后的普通均值（等价性验收口径）。"""
    logits = sample_logits()
    targets = TARGETS
    weights = torch.tensor([5.0, 1.0])
    duplicated = torch.tensor([5, 1])

    weighted = loss_from_outputs(logits, targets, DEVICE, SIGMA, weights)
    expanded = loss_from_outputs(
        logits.repeat_interleave(duplicated, dim=0),
        targets.repeat_interleave(duplicated),
        DEVICE,
        SIGMA,
    )

    assert weighted.item() == pytest.approx(expanded.item())


def test_weighted_loss_matches_per_sample_weighting() -> None:
    logits = sample_logits()
    weights = torch.tensor([5.0, 1.0])
    first, second = per_sample_kl(logits, TARGETS)

    got = loss_from_outputs(logits, TARGETS, DEVICE, SIGMA, weights).item()

    assert got == pytest.approx((5.0 * first + second) / 6.0)


def test_train_epoch_returns_weighted_mean() -> None:
    logits = sample_logits()
    weights = torch.tensor([5.0, 1.0])
    model = FixedLogits()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)

    returned = train_epoch(model, [(logits, TARGETS, weights)], optimizer, DEVICE, SIGMA)

    first, second = per_sample_kl(logits, TARGETS)
    assert returned == pytest.approx((5.0 * first + second) / 6.0)


def test_train_epoch_aggregates_batches_by_weight_sum() -> None:
    """整轮按权重和聚合，不是各批加权均值的普通平均。"""
    logits = sample_logits()
    model = FixedLogits()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    batches = [
        (logits[0:1], TARGETS[0:1], torch.tensor([5.0])),
        (logits[1:2], TARGETS[1:2], torch.tensor([1.0])),
    ]

    returned = train_epoch(model, batches, optimizer, DEVICE, SIGMA)

    first, second = per_sample_kl(logits, TARGETS)
    assert returned == pytest.approx((5.0 * first + second) / 6.0)
    assert returned != pytest.approx((first + second) / 2.0)


def test_eval_loss_ignores_sample_weights() -> None:
    """val 侧没有困难目录：第三项（恒为 1.0）不入损失，也不改读数。"""
    logits = sample_logits()
    model = FixedLogits()
    plain = eval_loss(model, [(logits, TARGETS, torch.ones(2))], DEVICE, SIGMA)[0]
    heavy = eval_loss(model, [(logits, TARGETS, torch.full((2,), 5.0))], DEVICE, SIGMA)[0]

    first, second = per_sample_kl(logits, TARGETS)
    assert plain == pytest.approx(heavy)
    assert plain == pytest.approx((first + second) / 2.0)
