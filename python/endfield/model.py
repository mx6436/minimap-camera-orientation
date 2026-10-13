"""摄像机角度预测模型（AzimuthNet）：逐像素预测颜色类别后验，沿半径以
加权和与跨半径 soft-min 双路聚合为 2×16 通道方位角剖面，再经循环一维卷积
匹配滤波输出每个方位角的 logits；交叉熵训练。推断解码在 MaaEnd C++
（`CameraOrientationPredictor`）。方位角轴全程
不下采样、只做循环卷积，对输入平移精确等变。

角向可达半径 = 主干各层 dilation 之和 + score 头的 ±1：每层 k=3 取 ±d，方位轴又
不下采样，所以每层贡献的半径就是它的 dilation。这是逐像素阶段能看到的角向跨度，
须覆盖视锥半宽（≥40°）。
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from endfield.strip import INPUT_CHANNELS, STRIP_H, STRIP_W

TARGET_SIGMA = 3.0
MATCH_DILATIONS = (1, 8, 16)
# 主干各块输出通道数与方位轴 dilation：可达半径 = Σd + score 头的 ±1 ≥ 视锥半宽 40°。
# 宽度是运算量的二次项、dilation 在 MAC 上免费，所以预算从宽度里出、可达范围不动；
# d 取 (2,4,8,26) 是为了让可达偏移集合稠密（3 块组合必然有洞，穷举无解）。
TRUNK_CHANNELS = (32, 32, 48, 48)
TRUNK_AZIMUTH_DILATIONS = (2, 4, 8, 26)
COVERAGE_TAU = 0.15


class CircularConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel: int, dilation: int) -> None:
        super().__init__()
        self.pad = (kernel - 1) // 2 * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, (self.pad, self.pad), mode="circular"))


# 架构版本：checkpoint 的键集与参数量随架构变化；arch 5 = 窄主干 32-32-48-48 +
# 角向可达 ±40；arch 4 的宽主干（32-64-128）与 arch 3 的 3 通道首层（polar）不兼容。
ARCH_VERSION = 5


class AzimuthNet(nn.Module):
    """逐像素评分 -> 径向双路聚合 -> 角向匹配滤波，输出 Z/360Z 上逐方位角的
    logits。

    方位角轴不做下采样、只做循环卷积，因此对输入平移精确等变。
    """

    def __init__(
        self,
        score_channels: int = 16,
        trunk_channels: tuple[int, ...] = TRUNK_CHANNELS,
        in_channels: int = INPUT_CHANNELS,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        channels = [in_channels, *trunk_channels]
        layers: list[nn.Module] = []
        for i, dilation in enumerate(TRUNK_AZIMUTH_DILATIONS):
            layers += [
                nn.CircularPad2d((dilation, dilation, 0, 0)),
                nn.ZeroPad2d((0, 0, 1, 1)),
                nn.Conv2d(
                    channels[i],
                    channels[i + 1],
                    3,
                    dilation=(1, dilation),
                    padding=0,
                    bias=False,
                ),
                nn.GroupNorm(16, channels[i + 1]),
                nn.ReLU(inplace=True),
            ]
            if i == 0:
                layers.append(nn.AvgPool2d((2, 1)))
        self.trunk = nn.Sequential(*layers)
        self.score = nn.Sequential(
            nn.CircularPad2d((1, 1, 0, 0)),
            nn.ZeroPad2d((0, 0, 1, 1)),
            nn.Conv2d(channels[-1], score_channels, 3),
        )
        with torch.no_grad():
            radial_rows = self.trunk(torch.zeros(1, in_channels, STRIP_H, STRIP_W)).shape[-2]
        self.radial = nn.Parameter(torch.zeros(score_channels, radial_rows))
        d1, d2, d3 = MATCH_DILATIONS
        self.filter = nn.Sequential(
            CircularConv1d(2 * score_channels, 32, 9, d1),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
            CircularConv1d(32, 32, 9, d2),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
            CircularConv1d(32, 1, 9, d3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.trunk(x)
        scores = torch.sigmoid(self.score(features))
        radial_weights = torch.softmax(self.radial, dim=1)
        mean_profile = torch.einsum("bcrw,cr->bcw", scores, radial_weights)
        cover_profile = COVERAGE_TAU * (
            math.log(scores.shape[2]) - torch.logsumexp(-scores / COVERAGE_TAU, dim=2)
        )
        profile = torch.cat((mean_profile, cover_profile), dim=1)
        return self.filter(profile).squeeze(1)


def fold_input_conventions(net: nn.Module) -> None:
    """把 /255 归一化折入首层卷积权重（交付输入的 HWC→CHW 转置在 wrapper 里）。

    标量缩放对卷积是线性变换，模型首层卷积无偏置，折叠精确无损。
    模型内部训练契约是 BGR NCHW [0,1]，交付输入无需通道翻转。
    """
    conv0 = next(m for m in net.trunk if isinstance(m, nn.Conv2d))
    if conv0.bias is not None:
        raise ValueError("first conv is expected to be bias-free for exact folding")
    conv0.weight.data = conv0.weight.detach() / 255.0


class ExportWrapper(nn.Module):
    """交付图的外层：uint8 NHWC 条带 -> 首层卷积 -> softmax 概率质量函数。"""

    def __init__(self, net: nn.Module) -> None:
        super().__init__()
        self.net = net

    def forward(self, strip_bgr: torch.Tensor) -> torch.Tensor:
        features = strip_bgr.permute(0, 3, 1, 2).float()
        return torch.softmax(self.net(features), dim=1)


def smoothed_targets(angles: np.ndarray, sigma: float = TARGET_SIGMA) -> torch.Tensor:
    """角度 -> Z/360Z 上的循环高斯概率质量函数，供交叉熵使用。"""
    bins = np.arange(360, dtype=np.float64)
    dist = (bins[None, :] - angles[:, None] + 180.0) % 360.0 - 180.0
    weights = np.exp(-0.5 * (dist / sigma) ** 2)
    weights /= weights.sum(axis=1, keepdims=True)
    return torch.from_numpy(weights.astype(np.float32))


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


EXPECTED_PARAMETER_COUNT = 72_289  # in_channels = 7（ref）


def expected_parameter_count(in_channels: int) -> int:
    """给定首层通道数的期望参数量；非 ref 首层说明架构未定义，直接拒绝。"""
    if in_channels != INPUT_CHANNELS:
        raise ValueError(f"unsupported in_channels: {in_channels!r}")
    return EXPECTED_PARAMETER_COUNT


def choose_device(value: str | None = None) -> torch.device:
    if value in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but is not available")
    return device


def _checkpoint_in_channels(state: object, path: Path | str) -> int:
    """从 checkpoint 权重推断首层通道数（旧 checkpoint 只有权重形状这一个来源）。"""
    weight = state.get("trunk.2.weight") if isinstance(state, dict) else None
    if not torch.is_tensor(weight) or weight.ndim != 4:
        raise ValueError(f"incompatible checkpoint weights: {path}")
    return int(weight.shape[1])


def load_model(path: Path | str, device: torch.device | str = "cpu") -> nn.Module:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError(f"invalid checkpoint: {path}")
    if checkpoint.get("arch") != ARCH_VERSION:
        raise ValueError(f"unsupported checkpoint arch version {checkpoint.get('arch')!r}: {path}")
    in_channels = _checkpoint_in_channels(checkpoint["model"], path)
    model = AzimuthNet(in_channels=in_channels)
    if count_trainable_parameters(model) != expected_parameter_count(in_channels):
        raise RuntimeError("unexpected model parameter count")
    # 键集不匹配即旧架构（AngleCNN）或过期的 checkpoint，与其报晦涩的
    # state_dict 错误不如直接拒绝
    if set(checkpoint["model"]) != set(model.state_dict()):
        raise ValueError(f"incompatible checkpoint weights: {path}")
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval()
