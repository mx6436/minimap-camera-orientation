"""前处理定义模块（#25）：训练、数据生成、live 与交付共用的唯一实现。

语义 = #23 决议的 clean_ideal：

- 观测：在 118x120 观测 ROI 上按条带网格一次双线性采样（`padding_mode="border"`）；
- 参考：资产坐标 = `(x, y) + (q_roi - ROI_POLE) * scale`（精确亚像素中心与
  `ZoneTemplateScale`），RGB 与 alpha 按同一网格各一次采样（`padding_mode="zeros"`，
  越界 = 参考缺失）；
- 合成在条带域一次完成：`ref.BGR = rgb * (a/255) + obs * (1 - a/255)`；
- 每个输出一次 Round（半偶）+ Cast 回 uint8。

几何约定（与 CONTEXT.md「极坐标展开」一致）：极点 = ROI 内 (59.0, 60.0) 像素中心；
角度 -> x 轴，第 j 列的像素中心对应方位角 j 度（正北 = 列 0，顺时针为正）；
半径 -> y 轴，第 i 行对应 `r_in + (i + 0.5) * step`（内径在上），基准下
`r_in = 12`、`r_out = 54`、`step = 1`，条带 42x360。
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

# ROI 几何（720p 基准下 MapLocator 的小地图 ROI）
ROI_W, ROI_H = 118, 120
ROI_POLE = (59.0, 60.0)
INNER_R, OUTER_R = 12.0, 54.0
# 输出条带：42 行（半径）× 360 列（1 度/列）
IMG_H, IMG_W = 42, 360
# 导出 dummy 的资产尺寸（图内 H/W 为动态维，仅用于捕获图结构）
DYNAMIC_ASSET_HW = (140, 160)
# 导出图的 opset：缓存戳的「图版本」（endfield/preprocess_cache.py），与导出参数同源
OPSET_VERSION = 18


def strip_roi_uv() -> torch.Tensor:
    """条带网格在 ROI 像素中心坐标系下的坐标 `[IMG_H, IMG_W, 2]`（u 右、v 下）。"""
    step = (OUTER_R - INNER_R) / IMG_H
    radii = INNER_R + step * (torch.arange(IMG_H, dtype=torch.float32) + 0.5)
    theta = torch.deg2rad(torch.arange(IMG_W, dtype=torch.float32))
    u = ROI_POLE[0] + radii[:, None] * torch.sin(theta)[None, :]
    v = ROI_POLE[1] - radii[:, None] * torch.cos(theta)[None, :]
    return torch.stack([u, v], dim=-1)


def _normalized(u: torch.Tensor, v: torch.Tensor, width: float, height: float) -> torch.Tensor:
    """像素中心坐标 -> `GridSample(align_corners=0)` 的 grid `[1, H, W, 2]`。"""
    return torch.stack([(2.0 * u + 1.0) / width - 1.0, (2.0 * v + 1.0) / height - 1.0], dim=-1)[
        None
    ]


def _to_uint8(value: torch.Tensor) -> torch.Tensor:
    """Round（半偶）+ 饱和 + Cast：每个输出的唯一取整点。"""
    return torch.round(value).clamp(0.0, 255.0).to(torch.uint8)


def _require_roi(roi: np.ndarray) -> np.ndarray:
    roi = np.asarray(roi)
    if roi.dtype != np.uint8 or roi.shape != (ROI_H, ROI_W, 3):
        raise ValueError(
            f"minimap ROI must be uint8 {ROI_H}x{ROI_W}x3 (HxWxBGR), got {roi.dtype} {roi.shape}"
        )
    return roi


def _require_bgra(asset: np.ndarray) -> np.ndarray:
    asset = np.asarray(asset)
    if asset.dtype != np.uint8 or asset.ndim != 3 or asset.shape[2] != 4:
        raise ValueError(
            f"asset must be uint8 HxWxBGRA, got {asset.dtype} {asset.shape};"
            " normalize 3-channel assets with normalize_asset() at the entry"
        )
    return asset


def normalize_asset(asset: np.ndarray) -> np.ndarray:
    """3 通道资产补 255 alpha 成全不透明 BGRA；4 通道原样返回。"""
    asset = np.asarray(asset)
    if asset.dtype != np.uint8 or asset.ndim != 3 or asset.shape[2] not in (3, 4):
        raise ValueError(f"asset must be uint8 HxWx3 or HxWx4, got {asset.dtype} {asset.shape}")
    if asset.shape[2] == 4:
        return asset
    alpha = np.full(asset.shape[:2], 255, dtype=np.uint8)
    return np.dstack([asset, alpha])


def sample_minimap(minimap: torch.Tensor) -> torch.Tensor:
    """NHWC uint8 观测 ROI -> float32 NCHW 条带采样值（观测与合成的共同输入）。"""
    u, v = strip_roi_uv().unbind(-1)
    return F.grid_sample(
        minimap.permute(0, 3, 1, 2).float(),
        _normalized(u, v, ROI_W, ROI_H),
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )


def observed_strip(roi: np.ndarray) -> np.ndarray:
    """118x120 BGR uint8 观测 ROI -> 42x360x3 uint8 观测条带。"""
    roi = _require_roi(roi)
    with torch.no_grad():
        sampled = sample_minimap(torch.from_numpy(roi)[None])
    return _to_uint8(sampled)[0].permute(1, 2, 0).numpy()


def sample_asset(
    asset: torch.Tensor, x: torch.Tensor, y: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """NHWC uint8 BGRA 资产 -> float32 NCHW 4 通道采样值（与观测同一条带网格）。

    资产坐标 = `(x, y) + (q_roi - ROI_POLE) * scale`：精确亚像素中心与精确 scale，
    图内不再做整数取整 / 中间裁剪窗 / 中间重采样。越界读 0（参考缺失）。
    """
    u, v = strip_roi_uv().unbind(-1)
    au = x + (u - ROI_POLE[0]) * scale
    av = y + (v - ROI_POLE[1]) * scale
    return F.grid_sample(
        asset.permute(0, 3, 1, 2).float(),
        _normalized(au, av, asset.shape[2], asset.shape[1]),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )


def _batch_strips(
    minimap: torch.Tensor,
    asset: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """NHWC uint8 -> `(observed NHWC uint8, reference NHWC uint8)`，一张图一次前向。"""
    obs_float = sample_minimap(minimap)
    sampled = sample_asset(asset, x, y, scale)
    rgb, alpha = sampled[:, :3], sampled[:, 3:4]
    weight = alpha / 255.0
    composed = rgb * weight + obs_float * (1.0 - weight)
    observed = _to_uint8(obs_float).permute(0, 2, 3, 1)
    reference = torch.cat([_to_uint8(composed), _to_uint8(alpha)], dim=1).permute(0, 2, 3, 1)
    return observed, reference


def strips(
    roi: np.ndarray, asset: np.ndarray, x: float, y: float, scale: float
) -> tuple[np.ndarray, np.ndarray]:
    """观测 ROI + BGRA 资产 -> `(obs 42x360x3, ref 42x360x4)` uint8 条带。

    数据生成、live 与 conformance 参考侧共用的唯一入口；资产须为 BGRA
    （3 通道入口先过 `normalize_asset`）。
    """
    roi = _require_roi(roi)
    asset = _require_bgra(asset)
    with torch.no_grad():
        observed, reference = _batch_strips(
            torch.from_numpy(roi)[None],
            torch.from_numpy(asset)[None],
            torch.tensor(float(x)),
            torch.tensor(float(y)),
            torch.tensor(float(scale)),
        )
    return observed[0].numpy(), reference[0].numpy()


class PreprocessGraph(nn.Module):
    """clean_ideal 前处理的导出图：NHWC uint8 -> `(observed, reference)` uint8。"""

    def forward(
        self,
        minimap: torch.Tensor,
        asset: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        scale: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _batch_strips(minimap, asset, x, y, scale)


def definition_hash() -> str:
    """定义模块内容的 sha256：manifest 与图 metadata 的「定义同源」凭据。"""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


_DESCRIPTION = (
    "Endfield minimap camera-orientation preprocess: polar ring strips from the minimap ROI "
    "and the MapLocator zone reference, exported from endfield/preprocess.py (#25)"
)
_INPUT_SPEC = (
    "minimap: uint8 [1,120,118,3] NHWC BGR 720p-baseline ROI; asset: uint8 [1,H,W,4] NHWC "
    "BGRA zone asset with dynamic H/W; x/y/scale: float32 scalars in asset pixel space "
    "(x,y = MapLocator position, scale = ZoneTemplateScale from the locate record)"
)
_OUTPUT_SPEC = (
    "observed: uint8 [1,42,360,3] obs.BGR; reference: uint8 [1,42,360,4] ref.BGR + ref.A. "
    "7-channel assembly [obs.BGR, ref.BGR, ref.A] and gap dispatch stay in the consumer"
)
_GEOMETRY_SPEC = (
    "pole (59.0,60.0) in ROI pixel-center coords, r_in=12, r_out=54, 42x360 (1 deg/column, "
    "north = column 0, clockwise positive); one-shot bilinear sampling: minimap on the strip "
    "grid with padding border, asset at (x,y)+(q_roi-pole)*scale with padding zeros "
    "(out-of-bounds = reference gap); strip-domain composite "
    "ref.BGR = rgb*(a/255) + obs*(1-a/255); one Round (half-to-even) + Cast per output"
)


def _attach_metadata(path: Path) -> None:
    import onnx

    model = onnx.load(str(path))
    metadata = {
        "description": _DESCRIPTION,
        "input_spec": _INPUT_SPEC,
        "output_spec": _OUTPUT_SPEC,
        "geometry_spec": _GEOMETRY_SPEC,
        "definition_hash": definition_hash(),
    }
    for key, value in metadata.items():
        entry = model.metadata_props.add()
        entry.key = key
        entry.value = value
    onnx.save(model, path)


def export_onnx(output: Path) -> Path:
    """导出 `preprocess.onnx`（opset18；契约见图 metadata）。"""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    model = PreprocessGraph().eval()
    minimap = torch.zeros(1, ROI_H, ROI_W, 3, dtype=torch.uint8)
    asset = torch.zeros(1, *DYNAMIC_ASSET_HW, 4, dtype=torch.uint8)
    args = (minimap, asset, torch.tensor(0.0), torch.tensor(0.0), torch.tensor(1.0))
    torch.onnx.export(
        model,
        args,
        output,
        opset_version=OPSET_VERSION,
        input_names=["minimap", "asset", "x", "y", "scale"],
        output_names=["observed", "reference"],
        dynamic_shapes=(None, {1: "H", 2: "W"}, None, None, None),
        external_data=False,
    )
    _attach_metadata(output)
    return output
