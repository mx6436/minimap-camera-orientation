"""live.py 的实机输入侧：帧基准缩放与 ref 参考配对构造。

- `to_base_frame` 把任意分辨率帧缩回训练基准 720p（观测 ROI 因此回到 118x120，
  gamescope 当前 1280x720 为 1:1 直通）；
- `ref_pair_at` 与 `prepare_data.py --mode ref` 走同一条前处理路径
  （定义模块 `endfield/preprocess.py` 的 `strip_pair()`）。

运行档案（record.json）的读取与 checkpoint 通道核对在 `endfield/run_record.py`。
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from endfield.locate import record_scale, zone_asset_path
from endfield.polar import BASE_SIZE
from endfield.preprocess import observed_roi
from endfield.ref import load_reference_image, ref_pair


class MissingZoneAsset(Exception):
    """MapLocator zone 找不到对应底图资产（实机按「定位不可用」处理，不中断）。"""

    def __init__(self, zone: str) -> None:
        super().__init__(f"no MapLocator asset for zone {zone!r}")
        self.zone = zone


def to_base_frame(frame: np.ndarray) -> np.ndarray:
    """任意分辨率帧 -> 1280x720 训练基准；已是基准则原样返回（不复制）。"""
    width, height = BASE_SIZE
    if frame.shape[1] == width and frame.shape[0] == height:
        return frame
    interpolation = cv2.INTER_AREA if frame.shape[1] > width else cv2.INTER_LINEAR
    return cv2.resize(frame, BASE_SIZE, interpolation=interpolation)


def _reference_asset(
    record: dict,
    assets_root: Path,
    cache: dict[Path, np.ndarray] | None,
) -> tuple[str, np.ndarray]:
    """定位记录 -> (zone, 原始底图资产)；zone 无资产抛 MissingZoneAsset。

    cache 按资产路径复用读取的原始 BGRA 资产，供实机循环跨帧使用。
    """
    zone = str(record.get("zone", ""))
    asset_path = zone_asset_path(zone, Path(assets_root))
    if asset_path is None:
        raise MissingZoneAsset(zone)
    if cache is None:
        return zone, load_reference_image(asset_path)
    asset = cache.get(asset_path)
    if asset is None:
        asset = cache[asset_path] = load_reference_image(asset_path)
    return zone, asset


def ref_pair_at(
    frame: np.ndarray,
    record: dict,
    assets_root: Path,
    cache: dict[Path, np.ndarray] | None = None,
) -> np.ndarray:
    """720p 基准帧 + MapLocator 定位记录 -> 42x360x7 参考配对 `[obs.BGR, ref.BGR, ref.A]`。

    参考裁剪的尺度取定位记录的 `scale` 字段；同一 (zone, x, y, scale) 下与
    prepare_data.py --mode ref 的两路产物同源（同一定义模块）。
    """
    zone, asset = _reference_asset(record, assets_root, cache)
    if "x" not in record or "y" not in record:
        raise ValueError(f"locate record lacks x/y: {record!r}")
    return ref_pair(
        observed_roi(frame),
        asset,
        float(record["x"]),
        float(record["y"]),
        record_scale(record),
    )
