"""前处理定义模块：条带几何、双线性采样与参考合成。

观测：在 118x120 观测 ROI 上按条带网格一次双线性采样（`padding_mode="border"`）。
参考：先由 `(x, y, scale)` 与条带几何裁出覆盖全部采样点及双线性支撑的采样窗（裁到
资产边界），把资产坐标减去窗口原点、按窗口宽高归一化后采样（`padding_mode="zeros"`，
越界 = 参考缺失）；窗口优先与整图采样数值等价（observed 逐字节、reference ≤1 LSB），
但图内只搬运窗口像素。
合成在条带域一次完成：`ref.BGR = rgb * (a/255) + obs * (1 - a/255)`；每个输出一次
Round（半偶）+ Cast 回 uint8。

几何约定：极点 = ROI 内 (59.0, 60.0) 像素中心；角度 -> x 轴，第 j 列的像素中心对应
方位角 j 度（正北 = 列 0，顺时针为正）；半径 -> y 轴，第 i 行对应
`r_in + (i + 0.5) * step`（内径在上），基准下 `r_in = 12`、`r_out = 54`、`step = 1`，
条带 42x360。
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


def prepare_asset(asset: np.ndarray) -> torch.Tensor:
    """3/4 通道资产（3 通道补 255 alpha）-> float32 NCHW `[1,4,H,W]`。

    图内 GridSample 只接受 float32；同一底图要反复采样时先 `prepare_asset` 一次，
    再逐样本调 `strips_prepared` 复用，避免逐样本重复转换窗口。
    """
    array = normalize_asset(asset)
    return torch.from_numpy(array)[None].permute(0, 3, 1, 2).float()


def _require_prepared(asset_float: torch.Tensor) -> torch.Tensor:
    shape = getattr(asset_float, "shape", None)
    if (
        not isinstance(asset_float, torch.Tensor)
        or asset_float.dtype != torch.float32
        or asset_float.dim() != 4
        or asset_float.shape[1] != 4
    ):
        raise ValueError(
            "prepared asset must be a float32 NCHW tensor with 4 channels "
            f"(a prepare_asset() output), got {type(asset_float).__name__} "
            f"{getattr(asset_float, 'dtype', None)} {shape}"
        )
    return asset_float


# 采样窗在 (x,y) 四周的额外安全边距（像素）：覆盖双线性支撑与浮点舍入。
WINDOW_PAD = 2.0


def _sampling_extent(scale: torch.Tensor) -> torch.Tensor:
    """采样点相对 (x, y) 的最大偏移（像素）：条带网格最大半径 * scale。

    行 i 的半径 = `INNER_R + (i + 0.5) * step`，i 最大 `IMG_H - 1`，故最大半径为
    `INNER_R + (IMG_H - 0.5) * step`（= 外径 - 半步长）；`(u, v) - ROI_POLE` 的两轴
    偏移都不超过它。几何变更时窗口随模块常数自动更新。
    """
    step = (OUTER_R - INNER_R) / IMG_H
    max_radius = INNER_R + (IMG_H - 0.5) * step
    return scale * max_radius


def _window_axis(
    center: torch.Tensor, extent: torch.Tensor, size: object
) -> tuple[torch.Tensor, torch.Tensor]:
    """单轴采样窗 `[start, end)`：覆盖 center±extent 与双线性支撑，裁到资产边界且非空。

    `floor(min)` 起、`floor(max) + 2` 止覆盖坐标的 floor / floor+1 两个支撑像素；
    空窗（资产完全在窗外）退化为 1 像素，采样坐标全在窗外 → grid_sample 读 0。
    """
    if not isinstance(center, torch.Tensor):
        center = torch.tensor(center)
    if not isinstance(extent, torch.Tensor):
        extent = torch.tensor(extent)
    start = torch.clamp(torch.floor(center - extent - WINDOW_PAD), min=0.0, max=size - 1)
    end = torch.clamp(torch.floor(center + extent + WINDOW_PAD) + 2.0, min=start + 1.0, max=size)
    return start.to(torch.int64), end.to(torch.int64)


def _asset_window(
    x: torch.Tensor, y: torch.Tensor, scale: torch.Tensor, height: object, width: object
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """覆盖全部采样点及双线性支撑的窗口 `[w0, w1) x [h0, h1)`（裁到资产边界、非空）。"""
    extent = _sampling_extent(scale)
    w0, w1 = _window_axis(x, extent, width)
    h0, h1 = _window_axis(y, extent, height)
    return w0, w1, h0, h1


def _sample_asset_crop(
    crop: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    scale: torch.Tensor,
    w0: torch.Tensor,
    h0: torch.Tensor,
    crop_w: torch.Tensor,
    crop_h: torch.Tensor,
) -> torch.Tensor:
    """float32 NCHW 窗口上按条带网格一次双线性采样；坐标减去窗口原点后按窗口宽高归一化。"""
    u, v = strip_roi_uv().unbind(-1)
    au = x + (u - ROI_POLE[0]) * scale - w0.to(torch.float32)
    av = y + (v - ROI_POLE[1]) * scale - h0.to(torch.float32)
    return F.grid_sample(
        crop,
        _normalized(au, av, crop_w.to(torch.float32), crop_h.to(torch.float32)),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )


def _sample_asset_float(
    asset_float: torch.Tensor, x: torch.Tensor, y: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """float32 NCHW BGRA 资产 -> float32 NCHW 4 通道采样值（窗口优先；越界读 0）。"""
    w0, w1, h0, h1 = _asset_window(x, y, scale, asset_float.shape[2], asset_float.shape[3])
    crop = asset_float[:, :, h0:h1, w0:w1]
    return _sample_asset_crop(crop, x, y, scale, w0, h0, w1 - w0, h1 - h0)


def sample_asset(
    asset: torch.Tensor, x: torch.Tensor, y: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """NHWC uint8 BGRA 资产 -> float32 NCHW 4 通道采样值（与观测同一条带网格）。

    资产坐标 = `(x, y) + (q_roi - ROI_POLE) * scale`；越界读 0（参考缺失）。
    """
    w0, w1, h0, h1 = _asset_window(x, y, scale, asset.shape[1], asset.shape[2])
    crop = asset[:, h0:h1, w0:w1, :].permute(0, 3, 1, 2).float()
    return _sample_asset_crop(crop, x, y, scale, w0, h0, w1 - w0, h1 - h0)


def _compose_strips(
    obs_float: torch.Tensor, sampled: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """float32 NCHW 观测采样 + 资产采样 -> `(observed, reference)` NHWC uint8 条带。"""
    rgb, alpha = sampled[:, :3], sampled[:, 3:4]
    weight = alpha / 255.0
    composed = rgb * weight + obs_float * (1.0 - weight)
    observed = _to_uint8(obs_float).permute(0, 2, 3, 1)
    reference = torch.cat([_to_uint8(composed), _to_uint8(alpha)], dim=1).permute(0, 2, 3, 1)
    return observed, reference


def _batch_strips(
    minimap: torch.Tensor,
    asset_float: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """NHWC uint8 观测 + float32 NCHW 资产 -> `(observed, reference)` NHWC uint8。"""
    obs_float = sample_minimap(minimap)
    sampled = _sample_asset_float(asset_float, x, y, scale)
    return _compose_strips(obs_float, sampled)


def strips(
    roi: np.ndarray, asset: np.ndarray, x: float, y: float, scale: float
) -> tuple[np.ndarray, np.ndarray]:
    """观测 ROI + BGRA 资产 -> `(obs 42x360x3, ref 42x360x4)` uint8 条带。

    资产须为 BGRA（3 通道入口先过 `normalize_asset`）。同一底图复用先 `prepare_asset`，
    再走 `strips_prepared` 避免逐样本窗口转换（两入口逐字节等价）。
    """
    return strips_prepared(roi, prepare_asset(_require_bgra(asset)), x, y, scale)


def strips_prepared(
    roi: np.ndarray, asset_float: torch.Tensor, x: float, y: float, scale: float
) -> tuple[np.ndarray, np.ndarray]:
    """观测 ROI + `prepare_asset()` 产物 -> `(obs 42x360x3, ref 42x360x4)` uint8 条带。

    与 `strips()` 逐字节等价；底图 float32 转换只做一次，供同一 zone 的批量数据生成逐样本复用。
    """
    roi = _require_roi(roi)
    asset_float = _require_prepared(asset_float)
    with torch.no_grad():
        observed, reference = _batch_strips(
            torch.from_numpy(roi)[None],
            asset_float,
            torch.tensor(float(x)),
            torch.tensor(float(y)),
            torch.tensor(float(scale)),
        )
    return observed[0].numpy(), reference[0].numpy()


class PreprocessGraph(nn.Module):
    """前处理导出图：NHWC uint8 -> `(observed, reference)` uint8。"""

    def forward(
        self,
        minimap: torch.Tensor,
        asset: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        scale: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _compose_strips(sample_minimap(minimap), sample_asset(asset, x, y, scale))


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
    "north = column 0, clockwise positive); window-first asset sampling (#36): a data-dependent "
    "window covering all sample points and their bilinear support (pixel-level floor/floor+1, "
    "plus margin) is clipped to the asset bounds and non-empty (empty window reads all-zero), "
    "then asset coordinates are shifted by the window origin and normalized by the window "
    "extent; one-shot bilinear sampling: minimap on the strip grid with padding border, asset at "
    "(x,y)+(q_roi-pole)*scale with padding zeros (out-of-bounds = reference gap); strip-domain "
    "composite ref.BGR = rgb*(a/255) + obs*(1-a/255); one Round (half-to-even) + Cast per output"
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
    # legacy exporter（dynamo=False）：数据相关 Slice 的窗口裁剪已验证可导出（#35/#36）；
    # dynamo 路径对数据相关边界直接失败。
    torch.onnx.export(
        model,
        args,
        output,
        opset_version=OPSET_VERSION,
        dynamo=False,
        input_names=["minimap", "asset", "x", "y", "scale"],
        output_names=["observed", "reference"],
        dynamic_axes={"asset": {1: "H", 2: "W"}},
        external_data=False,
    )
    _attach_metadata(output)
    return output
