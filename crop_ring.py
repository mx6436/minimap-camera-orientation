#!/usr/bin/env python3
"""将 data/raw 中的 PNG 裁剪为圆环形样本，输出到 data/processed。

ROI 中心 (108, 111)，内径 11，外径 56，硬掩膜（无过渡）：
保留像素中心到 ROI 中心距离满足 11 <= d <= 56 的像素，
其余像素置为全透明。输出 112x112 外接正方形 RGBA PNG，文件名保持不变。

距离按像素中心计算：像素 (i, j) 的中心为 (j + 0.5, i + 0.5)。
"""

from pathlib import Path

import numpy as np
from PIL import Image

ROI_CENTER = (108.0, 111.0)
INNER_R = 11.0
OUTER_R = 56.0
BOX = 112  # 外径 56 的外接正方形边长

RAW_DIR = Path(__file__).resolve().parent / "data" / "raw"
OUT_DIR = Path(__file__).resolve().parent / "data" / "processed"


def ring_mask(w: int, h: int) -> np.ndarray:
    """整帧 (w x h) 的环形掩膜，0/255，硬边界。

    按像素中心到 ROI 中心的距离平方比较，避免开方。
    """
    cx, cy = ROI_CENTER
    xs = np.arange(w) + 0.5 - cx
    ys = np.arange(h) + 0.5 - cy
    d2 = xs[None, :] ** 2 + ys[:, None] ** 2
    keep = (d2 >= INNER_R**2) & (d2 <= OUTER_R**2)
    return (keep * 255).astype(np.uint8)


def crop_box() -> tuple[int, int, int, int]:
    """外接正方形裁剪框 (left, top, right, bottom)，right/bottom 为开区间。"""
    cx, cy = ROI_CENTER
    left = int(round(cx - BOX / 2))
    top = int(round(cy - BOX / 2))
    return left, top, left + BOX, top + BOX


def process(src: Path, dst: Path, mask: np.ndarray) -> None:
    left, top, right, bottom = crop_box()
    im = Image.open(src).convert("RGBA")
    box = im.crop((left, top, right, bottom))
    arr = np.asarray(box)
    # 与源图自带 alpha 取小，再叠加环形掩膜
    alpha = np.minimum(arr[..., 3], mask[top:bottom, left:right])
    out = np.dstack([arr[..., :3], alpha])
    Image.fromarray(out, "RGBA").save(dst)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pngs = sorted(RAW_DIR.glob("*.png"))
    if not pngs:
        raise SystemExit(f"no png found in {RAW_DIR}")

    masks: dict[tuple[int, int], np.ndarray] = {}
    for i, src in enumerate(pngs, 1):
        with Image.open(src) as im:
            size = im.size
        if size not in masks:
            masks[size] = ring_mask(*size)
        process(src, OUT_DIR / src.name, masks[size])
        if i % 50 == 0 or i == len(pngs):
            print(f"[{i}/{len(pngs)}] {src.name}")

    print(f"done: {len(pngs)} images -> {OUT_DIR}")

    # 自检：第一张输出的环内像素数应约为 π*(56²-11²) ≈ 9472
    first = OUT_DIR / pngs[0].name
    alpha = np.asarray(Image.open(first).convert("RGBA"))[..., 3]
    n = int((alpha > 0).sum())
    print(f"self-check: {first.name} ring pixels = {n} (expected ~{np.pi * (OUTER_R**2 - INNER_R**2):.0f})")


if __name__ == "__main__":
    main()
