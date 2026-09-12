"""参考输入（ref）前处理：观测与 MapLocator 参考底图各自极坐标展开后的 7 通道张量。

ref = 7 通道 `[obs.BGR, ref.BGR, ref.A]`：观测与参考各自展开后拼接，不预先相减。
本模块同时提供参考底图侧的共用几何（资产读取、zone 尺度比、ROI 裁剪），供
`prepare_data.py --mode ref` 与 `live.py` 实机推理复用，保证两侧逐字节一致。

参考 BGR 采用观测背底合成 `ref.BGR = black_ref + obs_roi*(1 - alpha/255)`
（ROI 域、`unwrap` 之前，float 计算、四舍五入回 uint8、>255 饱和）：alpha==0 处
逐像素等于观测（参考缺失处 copy 观测）、alpha==255 处等于黑底合成
`rgb*alpha/255`、中间连续过渡。`ref.A` 为资产原始连续 alpha（不二值化、不设阈值），
裁剪越界与资产透明同为「参考缺失」= 0；观测侧无 alpha，保持 3 通道。

几何约定（见 map #8 的决策票）：
- 参考裁剪与观测同视野：118x120、中心 = MapLocator 的 (x, y)；尺度按 zone 的
  ZoneTemplateScale（绝大多数 1:1 直接裁，ValleyIV_Base 15/16）；
- 越界处裁到边界、外侧填 0（黑），不失败；
- 极坐标展开：先在笛卡尔域裁 ROI，再套 polar.unwrap（极点 = ROI 中心）。
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from endfield.polar import IMG_H, IMG_W, INNER_R, OUTER_R, ROI_CENTER, imread_png, unwrap

REPO_ROOT = Path(__file__).resolve().parents[1]
# 本地 MaaEnd 工作副本的 MapLocator 底图目录（gitignored），见 local/maplocator/README.local.md
MAP_ASSETS_ROOT = REPO_ROOT / "local" / "maplocator" / "resource" / "image" / "MapLocator"

# 观测 ROI 与参考裁剪共享的几何（720p 基准，与 MapLocator kDefaultMinimapRoi 一致）
ROI_W, ROI_H = 118, 120
ROI_POLE = (ROI_W / 2.0, ROI_H / 2.0)

# 输入通道数：[obs.BGR, ref.BGR, ref.A]
REF_CHANNELS = 7
# 数据根中参考流的并行子目录（processed_ref/ref、train_ref/ref 等）
REF_SUBDIR = "ref"

# 底图资产与观测小地图的像素尺度比，镜像 MapLocator.cpp 的 ZoneTemplateScale：匹配时
# MapLocator 把观测模板按该比例缩放到底图尺度，参考裁剪必须镜像同一比例（裁
# ROI*s 的资产窗口再缩回 ROI），否则 ValleyIV_Base 的 6.7% 尺度差会在边缘累积成
# 数像素错位。来源：MapLocator 源码复核 + 全量 accepted 样本上的梯度 NCC 峰值实测
# （ValleyIV_Base s=15/16 显著优于 1.0，其余 zone 1.0 显著优于 15/16）。
ZONE_SCALES: dict[str, float] = {"ValleyIV_Base": 15.0 / 16.0}


def zone_scale(zone_id: str) -> float:
    """zone 的底图尺度比（1.0 = 底图与观测 1:1）。"""
    return ZONE_SCALES.get(zone_id, 1.0)


def composite_on_black(image: np.ndarray) -> np.ndarray:
    """RGBA -> BGR，按黑底合成 rgb*alpha/255；已是 BGR 则原样返回。"""
    if image.ndim == 3 and image.shape[2] == 4:
        alpha = image[..., 3:4].astype(np.float32) / 255.0
        return np.round(image[..., :3].astype(np.float32) * alpha).astype(np.uint8)
    return image


def load_reference_image(path: Path) -> np.ndarray:
    """读取参考底图资产原图：BGR 或 BGRA uint8，未做黑底合成。"""
    image = imread_png(path)
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError(f"{path}: expected 3/4-channel PNG, got shape {image.shape}")
    return image


def crop_centered(image: np.ndarray, cx: float, cy: float, width: int, height: int) -> np.ndarray:
    """以 (cx, cy) 为中心裁 width x height；越界外侧填 0（黑），不失败。"""
    pad_x, pad_y = width, height
    padded = cv2.copyMakeBorder(
        image, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )
    x0 = int(round(cx)) - width // 2 + pad_x
    y0 = int(round(cy)) - height // 2 + pad_y
    return padded[y0 : y0 + height, x0 : x0 + width].copy()


def reference_crop(asset: np.ndarray, x: float, y: float, scale: float = 1.0) -> np.ndarray:
    """zone 资产在 (x,y) 处的参考裁剪：裁 ROI*scale 的窗口再缩回 ROI 几何。

    scale=1.0（绝大多数 zone）就是 118x120 的 1:1 直接裁剪；ValleyIV_Base 的底图
    相对观测缩放过，必须按 zone_scale 缩小窗口后放大回 ROI，与观测同视野。
    """
    width, height = round(ROI_W * scale), round(ROI_H * scale)
    crop = crop_centered(asset, x, y, width, height)
    if scale == 1.0:
        return crop
    return cv2.resize(crop, (ROI_W, ROI_H), interpolation=cv2.INTER_LINEAR)


def observed_roi(frame: np.ndarray) -> np.ndarray:
    """从 720p 基准整帧裁出 118x120 观测 ROI（中心 = polar.ROI_CENTER）。"""
    left = int(ROI_CENTER[0]) - ROI_W // 2
    top = int(ROI_CENTER[1]) - ROI_H // 2
    if top < 0 or left < 0 or top + ROI_H > frame.shape[0] or left + ROI_W > frame.shape[1]:
        raise ValueError(
            f"frame {frame.shape[1]}x{frame.shape[0]} too small for {ROI_W}x{ROI_H} "
            f"ROI at {ROI_CENTER}"
        )
    return frame[top : top + ROI_H, left : left + ROI_W]


def reference_alpha_plane(asset: np.ndarray) -> np.ndarray:
    """底图资产的原始 alpha 平面；3 通道资产视为完全不透明（255）。"""
    if asset.ndim == 3 and asset.shape[2] == 4:
        return asset[..., 3]
    return np.full(asset.shape[:2], 255, dtype=np.uint8)


def reference_planes(
    asset: np.ndarray, x: float, y: float, scale: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """资产在 (x,y) 的参考裁剪 -> (黑底合成 BGR, 原始 alpha)，几何与 reference_crop 一致。"""
    bgr = reference_crop(composite_on_black(asset), x, y, scale)
    alpha = reference_crop(reference_alpha_plane(asset), x, y, scale)
    return bgr, alpha


def compose_observed_backdrop(
    black_ref: np.ndarray, alpha: np.ndarray, observed: np.ndarray
) -> np.ndarray:
    """参考 BGR 的观测背底合成 `black_ref + observed*(1 - alpha/255)`。

    全部在 ROI 域做（`unwrap` 之前）：float 计算、四舍五入回 uint8，>255 饱和
    而非回绕。alpha==0 处 black_ref 为 0（黑底合成），结果即观测像素。
    """
    if (
        black_ref.shape[:2] != observed.shape[:2]
        or black_ref.shape != observed.shape
        or alpha.shape != black_ref.shape[:2]
    ):
        raise ValueError(
            f"shape mismatch: black_ref {black_ref.shape}, alpha {alpha.shape}, "
            f"observed {observed.shape}"
        )
    weight = (1.0 - alpha.astype(np.float32) / 255.0)[..., None]
    composed = black_ref.astype(np.float32) + observed.astype(np.float32) * weight
    return np.clip(np.round(composed), 0, 255).astype(np.uint8)


def _unwrap_plane(plane: np.ndarray) -> np.ndarray:
    return unwrap(plane, ROI_POLE[0], ROI_POLE[1], INNER_R, OUTER_R)


def reference_strip(
    observed: np.ndarray, asset: np.ndarray, x: float, y: float, scale: float = 1.0
) -> np.ndarray:
    """118x120 观测 ROI + 原始底图资产 -> 42x360x4 参考展开条带。

    BGR = 观测背底合成（参考缺失处逐像素等于观测），A = 资产原始连续 alpha。
    """
    black_ref, alpha = reference_planes(asset, x, y, scale)
    bgr = compose_observed_backdrop(black_ref, alpha, observed)
    return np.dstack([_unwrap_plane(bgr), _unwrap_plane(alpha)])


def ref_tensor(observed: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """观测 BGR 条带与参考 BGRA 条带 -> 7 通道 ref 张量 `[obs.BGR, ref.BGR, ref.A]`。"""
    if observed.shape != (IMG_H, IMG_W, 3):
        raise ValueError(f"observed strip must be {IMG_H}x{IMG_W}x3, got {observed.shape}")
    if reference.shape != (IMG_H, IMG_W, 4):
        raise ValueError(f"reference strip must be {IMG_H}x{IMG_W}x4, got {reference.shape}")
    return np.concatenate([observed, reference], axis=2)


def ref_strip(
    observed: np.ndarray, asset: np.ndarray, x: float, y: float, scale: float = 1.0
) -> np.ndarray:
    """118x120 观测 ROI + 原始底图资产 -> 42x360x7 ref 张量（训练与 live 共用编码）。"""
    reference = reference_strip(observed, asset, x, y, scale)
    return ref_tensor(_unwrap_plane(observed), reference)


def reference_gap_fraction(reference: np.ndarray) -> float:
    """参考条带的缺失像素占比：`ref.A < 255`（成片透明/越界与抗锯齿细边同计）。

    输入为 42x360x4 参考条带（各半径等权）；供训练时的样本过滤使用。
    """
    if reference.ndim != 3 or reference.shape[2] != 4:
        raise ValueError(f"reference strip must be HxWx4, got {reference.shape}")
    return float(np.mean(reference[..., 3] < 255))
