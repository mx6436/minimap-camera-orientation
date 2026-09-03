"""摄像机角度预测模型，两种架构可选（`architecture` 配置键）：

- ConeCNN（默认）：逐像素预测颜色类别后验，沿半径聚合为 360 维方位角剖面，
  再经循环一维卷积匹配滤波输出每个方位角的 logits；交叉熵训练，argmax 解码。
  方位角轴全程不下采样、只做循环卷积，对输入平移精确等变，
  旋转增广只提供数据多样性，不承担等变性。
- AngleCNN：旧基线，保留供对照复现。循环填充仅作用于方位角轴（连通 0/360 接缝）；
  半径轴上下边界是内径与外径，边界外即环外，零填充。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from endfield.data_utils import decode_angle

DEFAULT_HEAD_GRID = (2, 22)
DEFAULT_HEAD_CHANNELS: int | None = 64
DEFAULT_RADIUS_POOL = "avg"
DEFAULT_NORM = "group"
DEFAULT_DROPOUT = 0.0
GROUP_NORM_GROUPS = 16

ARCHITECTURE_ANGLE_CNN = "angle_cnn"
ARCHITECTURE_CONE = "cone"
ARCHITECTURES = (ARCHITECTURE_ANGLE_CNN, ARCHITECTURE_CONE)
DEFAULT_ARCHITECTURE = ARCHITECTURE_CONE

# 目标平滑 σ：扇形边缘是软过渡，σ 过小会让交叉熵梯度集中在 bin 边界上抖动
TARGET_SIGMA = 2.0
REFINE_RADIUS = 5
MATCH_DILATIONS = (4, 8, 16)


class AngleCNN(nn.Module):
    def __init__(
        self,
        dropout: float = DEFAULT_DROPOUT,
        head_grid: tuple[int, int] = DEFAULT_HEAD_GRID,
        head_channels: int | None = DEFAULT_HEAD_CHANNELS,
        radius_pool: str = DEFAULT_RADIUS_POOL,
        norm: str = DEFAULT_NORM,
    ) -> None:
        super().__init__()
        if radius_pool not in ("max", "avg"):
            raise ValueError(f"radius_pool must be 'max' or 'avg', got {radius_pool!r}")
        if norm not in ("batch", "group"):
            raise ValueError(f"norm must be 'batch' or 'group', got {norm!r}")
        if len(head_grid) != 2 or min(head_grid) < 1:
            raise ValueError(f"head_grid must be two positive ints, got {head_grid!r}")
        if head_channels is not None and head_channels < 1:
            raise ValueError(f"head_channels must be None or >= 1, got {head_channels!r}")
        self._head_grid = (head_grid[0], head_grid[1])
        self._head_channels = head_channels
        self._radius_pool = radius_pool
        self._norm = norm

        def norm_layer(channels: int) -> nn.Module:
            if norm == "batch":
                return nn.BatchNorm2d(channels)
            return nn.GroupNorm(GROUP_NORM_GROUPS, channels)

        def trunk_pool() -> nn.Module:
            if radius_pool == "max":
                return nn.MaxPool2d(2)
            # 半径取平均（聚合相邻半径的证据），方位角仍取 max
            return nn.Sequential(nn.AvgPool2d((2, 1)), nn.MaxPool2d((1, 2)))

        channels = [3, 32, 64, 128, 192, 256]
        layers: list[nn.Module] = []
        for i in range(5):
            # 循环填充仅限方位角轴；半径轴的上下边界不是同一位置，边界外即环外，零填充
            layers.append(
                nn.Sequential(
                    nn.CircularPad2d((1, 1, 0, 0)),
                    nn.ZeroPad2d((0, 0, 1, 1)),
                    nn.Conv2d(channels[i], channels[i + 1], 3, padding=0, bias=False),
                )
            )
            layers.append(norm_layer(channels[i + 1]))
            layers.append(nn.ReLU(inplace=True))
            if i < 4:
                layers.append(trunk_pool())
        self.features = nn.Sequential(*layers)

        grid_h, grid_w = head_grid
        if head_channels is None:
            self.pool = nn.AdaptiveAvgPool2d((grid_h, grid_w))
            head_in = 256 * grid_h * grid_w
        else:
            self.pool = nn.Sequential(
                nn.Conv2d(256, head_channels, 1), nn.AdaptiveAvgPool2d((grid_h, grid_w))
            )
            head_in = head_channels * grid_h * grid_w
        # 保留粗粒度空间布局，供方向敏感特征使用
        self.head = nn.Sequential(
            nn.Linear(head_in, 58),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(58, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.head(x)

    def is_default_architecture(self) -> bool:
        return (
            self.head_grid == DEFAULT_HEAD_GRID
            and self.head_channels == DEFAULT_HEAD_CHANNELS
            and self.radius_pool == DEFAULT_RADIUS_POOL
            and self.norm == DEFAULT_NORM
        )

    @property
    def head_grid(self) -> tuple[int, int]:
        return self._head_grid

    @property
    def head_channels(self) -> int | None:
        return self._head_channels

    @property
    def radius_pool(self) -> str:
        return self._radius_pool

    @property
    def norm(self) -> str:
        return self._norm


class CircularConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel: int, dilation: int) -> None:
        super().__init__()
        self.pad = (kernel - 1) // 2 * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, (self.pad, self.pad), mode="circular"))


class ConeCNN(nn.Module):
    """逐像素评分 -> 径向聚合 -> 角向匹配滤波，输出 360 bin logits。

    方位角轴不做下采样、只做循环卷积，因此对输入平移精确等变；
    旋转增广只提供数据多样性，不承担等变性。
    """

    def __init__(self, score_channels: int = 16, trunk_channels: int = 128) -> None:
        super().__init__()
        channels = [3, 32, 64, trunk_channels]
        layers: list[nn.Module] = []
        for i in range(3):
            layers += [
                nn.CircularPad2d((1, 1, 0, 0)),
                nn.ZeroPad2d((0, 0, 1, 1)),
                nn.Conv2d(channels[i], channels[i + 1], 3, padding=0, bias=False),
                nn.GroupNorm(16, channels[i + 1]),
                nn.ReLU(inplace=True),
            ]
            if i < 2:
                layers.append(nn.AvgPool2d((2, 1)))
        self.trunk = nn.Sequential(*layers)
        self.score = nn.Conv2d(trunk_channels, score_channels, 1)
        self.radial = nn.Parameter(torch.zeros(score_channels, 11))
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
    """角度 -> 360 bin 循环高斯分布，供交叉熵使用。"""
    bins = np.arange(360, dtype=np.float64)
    dist = (bins[None, :] - angles[:, None] + 180.0) % 360.0 - 180.0
    weights = np.exp(-0.5 * (dist / sigma) ** 2)
    weights /= weights.sum(axis=1, keepdims=True)
    return torch.from_numpy(weights.astype(np.float32))


def decode_logits(logits: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """argmax 定峰后取 ±REFINE_RADIUS 窗口内的循环加权平均，得到亚度级角度；
    置信度为峰值 bin 的 softmax 概率。"""
    values = logits.detach().cpu()
    probs = torch.softmax(values, dim=1)
    centers = probs.argmax(dim=1).numpy()
    confidence = probs.numpy()[np.arange(len(centers)), centers]
    offsets = np.arange(-REFINE_RADIUS, REFINE_RADIUS + 1)
    decoded = np.empty(len(centers), dtype=np.float64)
    for i, center in enumerate(centers):
        cols = (center + offsets) % 360
        weights = probs[i, cols].numpy()
        radians = np.deg2rad(cols)
        angle = np.degrees(
            np.arctan2(np.sum(weights * np.sin(radians)), np.sum(weights * np.cos(radians)))
        )
        value = angle % 360.0
        decoded[i] = 0.0 if value == 360.0 else value
    return decoded, confidence


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


EXPECTED_PARAMETER_COUNT = 937_872
EXPECTED_CONE_PARAMETER_COUNT = 110_017


def build_model(architecture: str, config: dict) -> nn.Module:
    if architecture == ARCHITECTURE_CONE:
        return ConeCNN()
    if architecture == ARCHITECTURE_ANGLE_CNN:
        return AngleCNN(**model_kwargs_from_config(config))
    raise ValueError(f"unknown architecture: {architecture!r}")


def model_kwargs_from_config(config: dict) -> dict:
    grid = config.get("head_grid", list(DEFAULT_HEAD_GRID))
    return dict(
        dropout=float(config.get("dropout", DEFAULT_DROPOUT)),
        head_grid=(int(grid[0]), int(grid[1])),
        head_channels=config.get("head_channels"),
        radius_pool=str(config.get("radius_pool", DEFAULT_RADIUS_POOL)),
        norm=str(config.get("norm", DEFAULT_NORM)),
    )


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
    config = checkpoint.get("config") or {}
    # 无 architecture 键的旧 checkpoint 一律是 AngleCNN
    architecture = str(config.get("architecture", ARCHITECTURE_ANGLE_CNN))
    model = build_model(architecture, config)
    if architecture == ARCHITECTURE_CONE:
        expected = EXPECTED_CONE_PARAMETER_COUNT
    elif model.is_default_architecture():
        expected = EXPECTED_PARAMETER_COUNT
    else:
        expected = None
    if expected is not None and count_trainable_parameters(model) != expected:
        raise RuntimeError("unexpected model parameter count")
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval()


def predict_angle(model: nn.Module, strip_rgb: np.ndarray) -> tuple[float, float]:
    """strip_rgb: 极坐标展开的输出（RGB uint8，HWC 排布，见 CONTEXT.md）。
    返回 (角度 [0,360), 置信度)：置信度语义随架构——ConeCNN 为峰值 softmax 概率，
    AngleCNN 为输出向量范数。
    """
    array = strip_rgb.astype(np.float32) / 255.0
    features = torch.from_numpy(array.transpose(2, 0, 1)).unsqueeze(0)
    features = features.to(next(model.parameters()).device)
    with torch.no_grad():
        output = model(features)
    if isinstance(model, ConeCNN):
        angles, confidence = decode_logits(output)
        return float(angles[0]), float(confidence[0])
    output = output.cpu().numpy()
    return float(decode_angle(output)[0]), float(np.linalg.norm(output[0]))
