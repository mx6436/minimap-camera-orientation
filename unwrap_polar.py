#!/usr/bin/env python3
"""将 data/raw 中的 PNG 极坐标展开为 RGB 矩形样本，输出到 data/processed_polar。

展开约定（与 CONTEXT.md "极坐标展开" 词条一致）：
- 极点为 ROI 中心 (108, 111)，角度零点为正北，顺时针为正；
- 角度 → x 轴：第 j 列的像素中心对应方位角 j 度（1°/列，x=0 是正北，
  0/360 接缝位于第 359 列与第 0 列之间）；
- 半径 → y 轴：第 i 行的像素中心对应半径 12.5 + i（i = 0..43），
  即内径 12、外径 56 之间 1:1 采样，内径在上；
- 第 j 列、第 i 行的输出像素由源图中 (cx + r·sin θ, cy − r·cos θ) 处
  双线性采样得到（反向映射），θ = j°，r = 12.5 + i；
- 输出 360x44 RGB PNG（无透明区域，天然全有效），文件名保持不变。

距离与角度均按像素中心约定，与 crop_ring.py 的环形掩膜（12 <= d <= 56）
一致：行半径 12.5..55.5 全部落在环内。
"""

from pathlib import Path

import numpy as np
from PIL import Image

ROI_CENTER = (108.0, 111.0)
INNER_R = 12.0
OUTER_R = 56.0
IMG_W = 360  # 1°/列
IMG_H = 44   # r = 12.5 .. 55.5

RAW_DIR = Path(__file__).resolve().parent / "data" / "raw"
OUT_DIR = Path(__file__).resolve().parent / "data" / "processed_polar"


def unwrap_polar(arr: np.ndarray) -> np.ndarray:
    """HxWx3 uint8 源图 → IMG_H x IMG_W x3 uint8 极坐标展开图。"""
    height, width = arr.shape[:2]
    cx, cy = ROI_CENTER
    # 双线性插值需要在源图内取到邻居像素：半径最大 55.5，floor+1 最多到 cx+56。
    if not (INNER_R + 1 <= cx <= width - OUTER_R - 1 and INNER_R + 1 <= cy <= height - OUTER_R - 1):
        raise ValueError(
            f"image {width}x{height} too small for ROI center {ROI_CENTER} "
            f"with outer radius {OUTER_R}"
        )

    radii = INNER_R + 0.5 + np.arange(IMG_H, dtype=np.float64)          # (H,)
    theta = np.deg2rad(np.arange(IMG_W, dtype=np.float64))              # (W,) 列 j = 方位角 j°
    src_x = cx + radii[:, None] * np.sin(theta)[None, :]                # (H, W)
    src_y = cy - radii[:, None] * np.cos(theta)[None, :]                # (H, W) 正北向上

    x0 = np.floor(src_x).astype(np.int64)
    y0 = np.floor(src_y).astype(np.int64)
    fx = (src_x - x0)[..., None]                                        # (H, W, 1)
    fy = (src_y - y0)[..., None]
    x0c = np.clip(x0, 0, width - 1)
    x1c = np.clip(x0 + 1, 0, width - 1)
    y0c = np.clip(y0, 0, height - 1)
    y1c = np.clip(y0 + 1, 0, height - 1)

    top = arr[y0c, x0c] * (1.0 - fx) + arr[y0c, x1c] * fx               # (H, W, 3)
    bottom = arr[y1c, x0c] * (1.0 - fx) + arr[y1c, x1c] * fx
    out = top * (1.0 - fy) + bottom * fy
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def load_source_rgb(path: Path) -> np.ndarray:
    """读取源 PNG 为 RGB；若有 alpha，先按 crop_ring.py 语义把透明像素 RGB 置零。"""
    with Image.open(path) as im:
        if im.format != "PNG":
            raise ValueError(f"{path.name}: expected PNG data, got {im.format}")
        rgba = np.asarray(im.convert("RGBA"), dtype=np.uint8).copy()
    rgba[rgba[..., 3] == 0, :3] = 0
    return rgba[..., :3]


def self_check_azimuth() -> None:
    """合成图自检：已知方位角的亮点必须落在对应列（顺时针、正北 = 第 0 列）。"""
    size = 224
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    cx, cy = size / 2.0, size / 2.0
    # 每个方位角用唯一半径，避免不同亮点落在同一展开行上互相干扰
    dots = ((0.0, 28.0), (90.0, 34.0), (180.0, 40.0), (270.0, 46.0))
    for azimuth_deg, radius in dots:
        px = int(round(cx + radius * np.sin(np.deg2rad(azimuth_deg))))
        py = int(round(cy - radius * np.cos(np.deg2rad(azimuth_deg))))
        canvas[py, px] = 255
    # 临时把 ROI 中心对到合成图中心再展开
    global ROI_CENTER
    original = ROI_CENTER
    ROI_CENTER = (cx, cy)
    try:
        out = unwrap_polar(canvas)
    finally:
        ROI_CENTER = original
    for azimuth_deg, radius in dots:
        row = int(round(radius - 0.5 - INNER_R))
        col = int(out[row].sum(axis=1).argmax())  # 该行最亮的列
        if abs(col - azimuth_deg) > 1:
            raise RuntimeError(
                f"azimuth self-check failed: dot at {azimuth_deg}° landed in column {col}"
            )
    print("azimuth self-check ok: 0°->col 0 (north), clockwise, 1°/column")


def main() -> None:
    pngs = sorted(RAW_DIR.glob("*.png"))
    if not pngs:
        raise SystemExit(f"no png found in {RAW_DIR}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for old in OUT_DIR.glob("*.png"):
        old.unlink()

    for i, src in enumerate(pngs, 1):
        arr = load_source_rgb(src)
        Image.fromarray(unwrap_polar(arr), "RGB").save(OUT_DIR / src.name)
        if i % 50 == 0 or i == len(pngs):
            print(f"[{i}/{len(pngs)}] {src.name}")

    output_names = sorted(p.name for p in OUT_DIR.glob("*.png"))
    input_names = sorted(p.name for p in pngs)
    if output_names != input_names:
        raise RuntimeError("processed_polar PNG names do not exactly match raw PNG names")
    for name in output_names:
        with Image.open(OUT_DIR / name) as im:
            if im.size != (IMG_W, IMG_H) or im.mode != "RGB":
                raise RuntimeError(f"invalid processed polar image: {name}")
    print(f"done: {len(pngs)} images -> {OUT_DIR}")

    self_check_azimuth()


if __name__ == "__main__":
    main()
