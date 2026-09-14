"""参考输入（ref）前处理：MapLocator 底图资产与观测 ROI 的读取、7 通道编码。

本模块的机械动作：

- 资产 I/O（`load_reference_image`）；
- 3 通道资产的入口归一化（`normalize_asset`，转发定义模块）；
- 7 通道拼接 `[obs.BGR, ref.BGR, ref.A]`（`ref_tensor` / `ref_strip`）与缺口占比。

几何约定（定义模块持有）：资产坐标 = `(x, y) + (q_roi - ROI_POLE) * scale`，`scale` 取
定位记录的 `ZoneTemplateScale` 字段；越界读 0 = 参考缺失；条带域一次合成
`ref.BGR = rgb * (a/255) + obs * (1 - a/255)`（alpha==0 处逐像素等于观测）。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from endfield import preprocess
from endfield.polar import imread_png
from endfield.preprocess import ROI_H, ROI_POLE, ROI_W

__all__ = [
    "MAP_ASSETS_ROOT",
    "REF_CHANNELS",
    "REF_SUBDIR",
    "ROI_W",
    "ROI_H",
    "ROI_POLE",
    "load_reference_image",
    "normalize_asset",
    "ref_tensor",
    "reference_strip",
    "ref_strip",
    "reference_gap_fraction",
]

REPO_ROOT = Path(__file__).resolve().parents[1]
# 本地 MaaEnd 工作副本的 MapLocator 底图目录（gitignored），见 local/maplocator/README.local.md
MAP_ASSETS_ROOT = REPO_ROOT / "local" / "maplocator" / "resource" / "image" / "MapLocator"

# 输入通道数：[obs.BGR, ref.BGR, ref.A]
REF_CHANNELS = 7
# 数据根中参考流的并行子目录（processed_ref/ref、train_ref/ref 等）
REF_SUBDIR = "ref"


def load_reference_image(path: Path) -> np.ndarray:
    """读取参考底图资产原图：BGR 或 BGRA uint8，未做黑底合成。"""
    image = imread_png(path)
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError(f"{path}: expected 3/4-channel PNG, got shape {image.shape}")
    return image


def normalize_asset(asset: np.ndarray) -> np.ndarray:
    """3 通道资产补 255 alpha 成全不透明 BGRA；4 通道原样（入口归一化）。"""
    return preprocess.normalize_asset(asset)


def ref_tensor(observed: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """观测 BGR 条带与参考 BGRA 条带 -> 7 通道 ref 张量 `[obs.BGR, ref.BGR, ref.A]`。"""
    if observed.shape != (preprocess.IMG_H, preprocess.IMG_W, 3):
        raise ValueError(
            f"observed strip must be {preprocess.IMG_H}x{preprocess.IMG_W}x3, got {observed.shape}"
        )
    if reference.shape != (preprocess.IMG_H, preprocess.IMG_W, 4):
        raise ValueError(
            f"reference strip must be {preprocess.IMG_H}x{preprocess.IMG_W}x4, "
            f"got {reference.shape}"
        )
    return np.concatenate([observed, reference], axis=2)


def reference_strip(
    observed: np.ndarray, asset: np.ndarray, x: float, y: float, scale: float = 1.0
) -> np.ndarray:
    """118x120 观测 ROI + 原始底图资产 -> 42x360x4 参考条带（定义模块唯一实现）。"""
    _, reference = preprocess.strips(observed, normalize_asset(asset), x, y, scale)
    return reference


def ref_strip(
    observed: np.ndarray, asset: np.ndarray, x: float, y: float, scale: float = 1.0
) -> np.ndarray:
    """118x120 观测 ROI + 原始底图资产 -> 42x360x7 ref 张量（训练与 live 共用编码）。"""
    observed_strip, reference = preprocess.strips(observed, normalize_asset(asset), x, y, scale)
    return ref_tensor(observed_strip, reference)


def reference_gap_fraction(reference: np.ndarray) -> float:
    """参考条带的缺失像素占比：`ref.A < 255`（成片透明/越界与抗锯齿细边同计）。

    输入为 42x360x4 参考条带（各半径等权）；供训练时的样本过滤使用。
    """
    if reference.ndim != 3 or reference.shape[2] != 4:
        raise ValueError(f"reference strip must be HxWx4, got {reference.shape}")
    return float(np.mean(reference[..., 3] < 255))
