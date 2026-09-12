"""live.py 的实机输入侧：run record 解析、帧基准缩放与 ref 条带构造。

- `load_run_config` 从 run 的 record.json 读 `input_mode`（旧 record 无 input_mode 时按
  polar 兼容；旧 pair v2 record 映射为 ref）与 ref 模式需要的 MapLocator 资产根，
  实机推理不新增用户必须传的模式参数；
- `to_base_frame` 把任意分辨率帧缩回训练基准 720p（观测 ROI 因此回到 118x120，
  gamescope 当前 1280x720 为 1:1 直通）；
- `ref_strip_at` 与 `prepare_data.py --mode ref` 走同一条前处理路径
  （reference_crop / ref_strip），保证实机输入与训练产物逐字节一致。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from endfield.locate import zone_asset_path
from endfield.polar import BASE_SIZE
from endfield.ref import load_reference_image, observed_roi, ref_strip, zone_scale

REPO_ROOT = Path(__file__).resolve().parents[1]
INPUT_MODES = ("polar", "ref")
# 各模式记录资产根的 record.json 字段
ASSETS_ROOT_KEYS = {"ref": "ref_reference_assets_root"}
# 旧 pair v2 record（ref 定名之前）的兼容映射：编码 v2 与 ref 逐字节一致；
# v1（黑底合成）语义不同，明确拒绝。
LEGACY_PAIR_MODE = "pair"
LEGACY_PAIR_ENCODING_V2 = 2
LEGACY_PAIR_ASSETS_KEY = "pair_reference_assets_root"


class MissingZoneAsset(Exception):
    """MapLocator zone 找不到对应底图资产（实机按「定位不可用」处理，不中断）。"""

    def __init__(self, zone: str) -> None:
        super().__init__(f"no MapLocator asset for zone {zone!r}")
        self.zone = zone


@dataclass(frozen=True)
class RunConfig:
    """run record.json 决定的实机输入配置；polar 不需要定位与资产。"""

    input_mode: str
    assets_root: Path | None


def _resolve_assets_root(path: Path, record: dict, key: str) -> Path:
    assets = record.get(key)
    if not isinstance(assets, str) or not assets:
        raise ValueError(f"{path}: ref run lacks {key}")
    root = Path(assets)
    if not root.is_absolute():
        root = REPO_ROOT / root
    return root


def load_run_config(run_dir: Path) -> RunConfig:
    path = run_dir / "record.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    # version 27 及以前的 run（polar 唯一时期）没有 input_mode 字段，按 polar 兼容
    input_mode = record.get("input_mode", "polar")
    assets_key = ASSETS_ROOT_KEYS.get(input_mode)
    if input_mode == LEGACY_PAIR_MODE:
        # 旧 pair v2 run（version 30 及以前）编码与 ref 相同，映射为 ref；
        # 缺 pair_encoding 即 v1 黑底合成，与 ref 编码不同，拒绝。
        if record.get("pair_encoding") != LEGACY_PAIR_ENCODING_V2:
            raise ValueError(
                f"{path}: legacy pair run without pair_encoding=2 is not ref-compatible"
            )
        input_mode = "ref"
        assets_key = LEGACY_PAIR_ASSETS_KEY
    if input_mode not in INPUT_MODES:
        raise ValueError(f"{path}: unsupported input_mode {record.get('input_mode')!r}")
    if input_mode == "polar":
        return RunConfig(input_mode="polar", assets_root=None)
    assert assets_key is not None
    return RunConfig(
        input_mode=input_mode, assets_root=_resolve_assets_root(path, record, assets_key)
    )


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


def ref_strip_at(
    frame: np.ndarray,
    record: dict,
    assets_root: Path,
    cache: dict[Path, np.ndarray] | None = None,
) -> np.ndarray:
    """720p 基准帧 + MapLocator 定位记录 -> 42x360x7 ref 张量 `[obs.BGR, ref.BGR, ref.A]`。

    同一 (zone, x, y) 下与 prepare_data.py --mode ref 的两路产物逐字节一致。
    """
    zone, asset = _reference_asset(record, assets_root, cache)
    if "x" not in record or "y" not in record:
        raise ValueError(f"locate record lacks x/y: {record!r}")
    return ref_strip(
        observed_roi(frame),
        asset,
        float(record["x"]),
        float(record["y"]),
        zone_scale(zone),
    )
