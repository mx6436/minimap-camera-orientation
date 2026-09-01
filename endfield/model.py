"""角度回归 CNN。

输入表示：极坐标展开（见 CONTEXT.md）——3 通道，循环填充连通 0/360 接缝。

- head_grid：保留完整方位角剖面，而不是平均成 90 度单元格。
- head_channels：使线性 head 在细网格上的开销可控。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from endfield.data_utils import decode_angle

DEFAULT_HEAD_GRID = (2, 22)
DEFAULT_HEAD_CHANNELS: int | None = 64
DEFAULT_RADIUS_POOL = "avg"
DEFAULT_NORM = "group"
DEFAULT_DROPOUT = 0.0
GROUP_NORM_GROUPS = 16


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
            layers.append(
                nn.Conv2d(
                    channels[i],
                    channels[i + 1],
                    3,
                    padding=1,
                    bias=False,
                    padding_mode="circular",
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


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


EXPECTED_PARAMETER_COUNT = 937_872


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


def load_model(path: Path | str, device: torch.device | str = "cpu") -> AngleCNN:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError(f"invalid checkpoint: {path}")
    model = AngleCNN(**model_kwargs_from_config(checkpoint.get("config") or {}))
    if (
        model.is_default_architecture()
        and count_trainable_parameters(model) != EXPECTED_PARAMETER_COUNT
    ):
        raise RuntimeError("unexpected model parameter count")
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval()


def predict_angle(model: nn.Module, strip_rgb: np.ndarray) -> tuple[float, float]:
    """strip_rgb: 极坐标展开的输出（RGB uint8，HWC 排布，见 CONTEXT.md）。
    返回 (角度 [0,360)，输出向量范数——可作置信度的代理指标)。
    """
    array = strip_rgb.astype(np.float32) / 255.0
    features = torch.from_numpy(array.transpose(2, 0, 1)).unsqueeze(0)
    features = features.to(next(model.parameters()).device)
    with torch.no_grad():
        output = model(features).cpu().numpy()
    return float(decode_angle(output)[0]), float(np.linalg.norm(output[0]))
