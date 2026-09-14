"""参考条带采样：底图定位 + 观测 ROI -> 条带对。

一次采样只有一条路径（`ReferenceSampler.strips`）：资产读取与 float32 转换按资产
路径复用，「同一底图逐样本复用」与「同一底图逐帧复用」是同一个实现。条带域之上的
7 通道张量编码在 `endfield/input_encoding.py`。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from endfield import preprocess
from endfield.polar import imread_png
from placement.placement import Placement


class MissingZoneAsset(Exception):
    """MapLocator zone 找不到对应底图资产（实机按「定位不可用」处理，不中断）。"""

    def __init__(self, zone: str) -> None:
        super().__init__(f"no MapLocator asset for zone {zone!r}")
        self.zone = zone


def load_reference_image(path: Path) -> np.ndarray:
    """读取参考底图资产原图：BGR 或 BGRA uint8，未做黑底合成。"""
    image = imread_png(path)
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError(f"{path}: expected 3/4-channel PNG, got shape {image.shape}")
    return image


class ReferenceSampler:
    """底图定位 -> 条带对；底图资产按路径复用，只读一次、只转一次 float32。"""

    def __init__(self, assets_root: Path) -> None:
        self._assets_root = Path(assets_root)
        self._prepared: dict[Path, torch.Tensor] = {}

    def strips(
        self, observed_roi: np.ndarray, placement: Placement
    ) -> tuple[np.ndarray, np.ndarray]:
        """118x120 BGR 观测 ROI + 底图定位 -> `(obs 42x360x3, ref 42x360x4)` uint8。

        采样语义全在前处理定义模块；本方法只负责找到底图、准备一次、复用。zone 拿不到
        资产时抛 `MissingZoneAsset`（调用方决定跳过还是中断）。
        """
        asset_path = placement.asset_path(self._assets_root)
        if asset_path is None:
            raise MissingZoneAsset(placement.zone)
        prepared = self._prepared.get(asset_path)
        if prepared is None:
            prepared = self._prepared[asset_path] = preprocess.prepare_asset(
                load_reference_image(asset_path)
            )
        return preprocess.strip_pair_prepared(
            observed_roi, prepared, placement.x, placement.y, placement.scale
        )
