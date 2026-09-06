"""摄像机角度预测模型（AzimuthNet）：逐像素预测颜色类别后验，沿半径聚合为
360 维方位角剖面，再经循环一维卷积匹配滤波输出每个方位角的 logits；
交叉熵训练，argmax 解码。方位角轴全程不下采样、只做循环卷积，对输入平移
精确等变。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from endfield.polar import IMG_H, IMG_W

# 目标平滑 σ：扇形边缘是软过渡，σ 过小会让交叉熵梯度集中在 bin 边界上抖动。
# 仅作 train.toml 未配置时的默认值；实际训练经 target_sigma 配置项传入。
TARGET_SIGMA = 3.0
REFINE_RADIUS = 5
# 匹配滤波的线性支撑集是三层跨度的和集 {d1·a+d2·b+d3·c}；首层跨度必须为 1，
# 否则和集只落在公因子的格点上（如 (4,8,16) 时仅模 4 同余列），线性部分
# 分解为互不相干的剩余类滤波器，输出带周期性纹波且无法经训练消除
MATCH_DILATIONS = (1, 8, 16)
# trunk 角向 dilation 逐层加倍：零参数地把逐像素打分的上下文扩到
# ±15°（图标在 r≈30px 处张角约 ±11°），径向保持无 dilation。
TRUNK_AZIMUTH_DILATIONS = (2, 4, 8)


class CircularConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel: int, dilation: int) -> None:
        super().__init__()
        self.pad = (kernel - 1) // 2 * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, (self.pad, self.pad), mode="circular"))


# 架构版本：checkpoint 缺失或不一致即拒绝加载；跨度等不影响参数量与键集的
# 变更必须递增此值，否则旧权重会被静默加载后输出无效结果
ARCH_VERSION = 2


class AzimuthNet(nn.Module):
    """逐像素评分 -> 径向聚合 -> 角向匹配滤波，输出 Z/360Z 上逐方位角的 logits。

    方位角轴不做下采样、只做循环卷积，因此对输入平移精确等变。
    """

    def __init__(self, score_channels: int = 16, trunk_channels: int = 128) -> None:
        super().__init__()
        channels = [3, 32, 64, trunk_channels]
        layers: list[nn.Module] = []
        for i, dilation in enumerate(TRUNK_AZIMUTH_DILATIONS):
            # 角向 pad 量随 dilation 增大，径向边界仍补零：内外径边界不连通
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
            # 仅块 1 后池化：42→21 行全数保留，不再丢弃最外圈半径行
            if i == 0:
                layers.append(nn.AvgPool2d((2, 1)))
        self.trunk = nn.Sequential(*layers)
        self.score = nn.Sequential(
            nn.CircularPad2d((1, 1, 0, 0)),
            nn.ZeroPad2d((0, 0, 1, 1)),
            nn.Conv2d(trunk_channels, score_channels, 3),
        )
        with torch.no_grad():
            radial_rows = self.trunk(torch.zeros(1, 3, IMG_H, IMG_W)).shape[-2]
        self.radial = nn.Parameter(torch.zeros(score_channels, radial_rows))
        d1, d2, d3 = MATCH_DILATIONS
        self.filter = nn.Sequential(
            CircularConv1d(score_channels, 32, 9, d1),
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
        profile = torch.einsum("bcrw,cr->bcw", scores, radial_weights)
        return self.filter(profile).squeeze(1)


def target_angles(targets: np.ndarray) -> np.ndarray:
    """sin/cos 目标反解回角度（往返误差 ~1e-5°，见 engine.py 同样用法）。"""
    return np.degrees(np.arctan2(targets[:, 0], targets[:, 1])) % 360.0


def smoothed_targets(angles: np.ndarray, sigma: float = TARGET_SIGMA) -> torch.Tensor:
    """角度 -> Z/360Z 上的循环高斯概率质量函数，供交叉熵使用。"""
    bins = np.arange(360, dtype=np.float64)
    dist = (bins[None, :] - angles[:, None] + 180.0) % 360.0 - 180.0
    weights = np.exp(-0.5 * (dist / sigma) ** 2)
    weights /= weights.sum(axis=1, keepdims=True)
    return torch.from_numpy(weights.astype(np.float32))


def decode_probs(probs: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """argmax 定峰后取 ±REFINE_RADIUS 窗口内的循环加权平均，得到亚度级角度。

    置信度为 360 个概率方向向量（方向 = bin 方位角，长度 = softmax 概率）的合成
    模长，再乘以解码方向与合成方向夹角的余弦：分布集中且窗口均值对准合成方向时
    接近 1，均匀、多峰对消或窗口均值被次要峰拉偏时趋 0。
    """
    values = probs.detach().cpu().numpy()
    centers = values.argmax(axis=1)
    radians = np.deg2rad(np.arange(360))
    sin_sum = values @ np.sin(radians)
    cos_sum = values @ np.cos(radians)
    confidence = np.hypot(sin_sum, cos_sum)
    offsets = np.arange(-REFINE_RADIUS, REFINE_RADIUS + 1)
    decoded = np.empty(len(centers), dtype=np.float64)
    for i, center in enumerate(centers):
        cols = (center + offsets) % 360
        weights = values[i, cols]
        radians = np.deg2rad(cols)
        angle = np.degrees(
            np.arctan2(np.sum(weights * np.sin(radians)), np.sum(weights * np.cos(radians)))
        )
        value = angle % 360.0
        decoded[i] = 0.0 if value == 360.0 else value
        resultant = np.degrees(np.arctan2(sin_sum[i], cos_sum[i]))
        confidence[i] *= np.cos(np.deg2rad(decoded[i] - resultant))
    return decoded, confidence


def decode_logits(logits: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    return decode_probs(torch.softmax(logits.detach().cpu(), dim=1))


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


EXPECTED_PARAMETER_COUNT = 126_561


def choose_device(value: str | None = None) -> torch.device:
    if value in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but is not available")
    return device


def load_model(path: Path | str, device: torch.device | str = "cpu") -> nn.Module:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError(f"invalid checkpoint: {path}")
    if checkpoint.get("arch") != ARCH_VERSION:
        raise ValueError(
            f"checkpoint arch version {checkpoint.get('arch')!r} != {ARCH_VERSION}: {path}"
        )
    model = AzimuthNet()
    if count_trainable_parameters(model) != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError("unexpected model parameter count")
    # 键集不匹配即旧架构（AngleCNN）或过期的 checkpoint，与其报晦涩的
    # state_dict 错误不如直接拒绝
    if set(checkpoint["model"]) != set(model.state_dict()):
        raise ValueError(f"incompatible checkpoint weights: {path}")
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval()


def predict_angle(model: nn.Module, strip_rgb: np.ndarray) -> tuple[float, float]:
    """strip_rgb: 极坐标展开的输出（RGB uint8，HWC 排布，见 CONTEXT.md）。
    返回 (角度 [0,360), 置信度)：置信度为 360 概率方向向量的合成模长
    乘以解码方向与合成方向夹角的余弦。
    """
    angle, confidence, _ = predict_probs(model, strip_rgb)
    return angle, confidence


def predict_probs(model: nn.Module, strip_rgb: np.ndarray) -> tuple[float, float, np.ndarray]:
    """predict_angle 附带第三返回值：Z/360Z 上的 softmax 概率质量函数。"""
    array = strip_rgb.astype(np.float32) / 255.0
    features = torch.from_numpy(array.transpose(2, 0, 1)).unsqueeze(0)
    features = features.to(next(model.parameters()).device)
    with torch.no_grad():
        output = model(features)
    probs = torch.softmax(output.detach().cpu(), dim=1)
    angles, confidence = decode_probs(probs)
    return float(angles[0]), float(confidence[0]), probs.numpy()[0]
