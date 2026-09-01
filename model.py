"""The angle regression CNN (baseline: 937,872 trainable parameters).

Input representation: polar unwraps (see CONTEXT.md) — 3 channels, with
circular padding to bridge the 0/360 seam.

The constructor defaults are the production baseline:
(2, 22) readout grid, 64-channel 1x1 compression conv, avg pooling on the
radius axis, GroupNorm. Structural variants stay parametrized for controlled
experiments:

- head_grid: readout pooling grid (radius rows, azimuth cols). Default
  (2, 22) — keeps the full azimuth profile instead of averaging it into
  90-degree cells.
- head_channels: 1x1 conv compressing trunk channels before the head pool
  (None keeps 256). Default 64; keeps the linear head affordable on the fine
  grid.
- radius_pool: "max" (MaxPool2d(2) on both axes) or "avg" (radius axis
  averaged instead of max'd; azimuth stays max). Default "avg".
- norm: "batch" (BatchNorm2d) or "group" (GroupNorm, 16 groups). Default
  "group".

This module also owns checkpoint loading (load_model) and device selection
(choose_device), so the inference scripts never import train.py.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

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
            raise ValueError(
                f"head_channels must be None or >= 1, got {head_channels!r}"
            )
        self._head_grid = tuple(head_grid)
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
            # radius averaged (evidence across neighbouring radii), azimuth kept max
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
        # Preserve coarse spatial layout for direction-sensitive features.
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
    return sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )


EXPECTED_PARAMETER_COUNT = 937_872


def model_kwargs_from_config(config: dict) -> dict:
    """Rebuild constructor kwargs from a checkpoint config dict.

    Missing keys fall back to the current defaults. Checkpoints saved before
    the architecture keys existed therefore rebuild as the baseline
    architecture, and their state dicts fail to load loudly instead of
    silently mismatching."""
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
    """Load a trained AngleCNN from a checkpoint file.

    A checkpoint is {"model": state_dict, "config": architecture kwargs};
    extra keys left over from pre-refactor checkpoints (optimizer/scheduler/
    RNG state) are ignored, so existing best.pt files load unchanged. Old
    checkpoints whose config predates the architecture keys rebuild as the
    baseline architecture and fail loudly on the state-dict mismatch.
    """
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
