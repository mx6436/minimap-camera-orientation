#!/usr/bin/env python3
"""The approximately one-million-parameter angle regression CNN.

Supports two input representations (see CONTEXT.md):
- RGBA ring crops (default): 4 channels, zero padding;
- polar unwraps: 3 channels, circular padding to bridge the 0/360 seam.
"""
from __future__ import annotations

import torch
from torch import nn


class AngleCNN(nn.Module):
    def __init__(self, in_channels: int = 4, padding_mode: str = "zeros") -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1, bias=False, padding_mode=padding_mode),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1, bias=False, padding_mode=padding_mode),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1, bias=False, padding_mode=padding_mode),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 192, 3, padding=1, bias=False, padding_mode=padding_mode),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(192, 256, 3, padding=1, bias=False, padding_mode=padding_mode),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        # Preserve coarse spatial layout for direction-sensitive features.
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.head = nn.Sequential(
            nn.Linear(256 * 4 * 4, 58),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(58, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.head(x)


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


EXPECTED_PARAMETER_COUNT = 995_952


def expected_parameter_count(in_channels: int) -> int:
    """参数量只随第一层卷积权重变化：32 * in_channels * 3 * 3（基准 4 通道）。"""
    return EXPECTED_PARAMETER_COUNT + 32 * 9 * (in_channels - 4)
