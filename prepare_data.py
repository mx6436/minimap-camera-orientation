#!/usr/bin/env python3
"""数据前处理：从 data/raw 生成处理样本并完成训练/验证划分。

流水线（一条命令完成，除 data/raw 与 data/val_manifest.json 外全部输出
在每次运行时清空重写，见 docs/adr/0001）：

  1. 清空 data/processed，从 data/raw 全量生成 --input 指定格式的样本；
  2. 清空 data/train、data/val，按 --split 指定的模式划分并复制；
  3. 重写 data/split_manifest.json 记录本次划分结果。

输入格式 --input（必填）：
  rgba  环形裁剪 112x112 RGBA：
        ROI 中心 (108, 111)，内径 12，外径 56，硬掩膜（无过渡）。
        保留像素中心到 ROI 中心距离满足 12 <= d <= 56 的像素，
        其余像素置为全透明（RGB 同步置零）。输出 112x112 外接正方形
        RGBA PNG，文件名保持不变。内径孔洞同时滤除位于中心圆内的箭头。
  polar 极坐标展开 360x44 RGB：
        极点为 ROI 中心 (108, 111)，角度零点为正北，顺时针为正；
        角度 -> x 轴：第 j 列的像素中心对应方位角 j 度（1°/列，x=0 是
        正北，0/360 接缝位于第 359 列与第 0 列之间）；
        半径 -> y 轴：第 i 行的像素中心对应半径 12.5 + i（i = 0..43），
        即内径 12、外径 56 之间 1:1 采样，内径在上；
        输出像素由源图 (cx + r·sin θ, cy − r·cos θ) 处双线性采样得到
        （反向映射）。输出无透明区域，天然全有效；行半径 12.5..55.5
        全部落在环内，箭头同样被排除。

切分模式 --split：
  random   按 SEED 与 30° 角度分箱随机留出约 15% 作验证集（每箱至少
           一张验证图）。同输入同种子结果确定，但新增 raw 数据后重跑
           会重新洗牌，已有验证文件可能被换出。
  manifest 清单切分：验证集成员由 data/val_manifest.json 直接指定，
           val = 清单 ∩ processed（清单引用 processed 中不存在的文件名
           则报错），train = processed 其余全部。用于留出地图（Held-out
           Map）的跨地图泛化验证：验证地图的样本一律不进入训练集。

两种格式共用同一套文件名与划分结果（split_manifest.json 的
train_files/val_files 逐文件一致），切换格式只需重跑本脚本并让
train.py 的 --input 跟着切换。
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from data_utils import (
    SEED,
    atomic_json_dump,
    load_json,
    parse_angle,
    png_names,
    validate_manifest_names,
)

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
TRAIN = ROOT / "data" / "train"
VAL = ROOT / "data" / "val"
SPLIT_MANIFEST = ROOT / "data" / "split_manifest.json"
VAL_MANIFEST = ROOT / "data" / "val_manifest.json"

VALIDATION_FRACTION = 0.15
ANGLE_BIN_DEGREES = 30
BIN_COUNT = 360 // ANGLE_BIN_DEGREES
SPLIT_MANIFEST_VERSION = 2
VAL_MANIFEST_VERSION = 1

# ---- 几何常量（rgba 与 polar 共用同一 ROI 约定）----

ROI_CENTER = (108.0, 111.0)
INNER_R = 12.0
OUTER_R = 56.0
BOX = 112   # rgba：外径 56 的外接正方形边长
IMG_W = 360  # polar：1°/列
IMG_H = 44   # polar：r = 12.5 .. 55.5


# ================= 第 1 步：raw -> processed =================

def clear_pngs(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for path in directory.glob("*.png"):
        path.unlink()


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


def process_rgba(src: Path, dst: Path, mask: np.ndarray) -> None:
    left, top, right, bottom = crop_box()
    with Image.open(src) as im:
        if im.format != "PNG":
            raise ValueError(f"{src.name}: expected PNG data, got {im.format}")
        width, height = im.size
        if not (0 <= left < right <= width and 0 <= top < bottom <= height):
            raise ValueError(f"{src.name}: crop box does not fit inside image size {im.size}")
        arr = np.asarray(im.convert("RGBA"), dtype=np.uint8)
    box = arr[top:bottom, left:right].copy()
    # 与源图自带 alpha 取小，再叠加环形掩膜。
    alpha = np.minimum(box[..., 3], mask[top:bottom, left:right])
    box[..., 3] = alpha
    box[alpha == 0, :3] = 0
    Image.fromarray(box, "RGBA").save(dst)


def unwrap_polar(arr: np.ndarray, center: tuple[float, float] = ROI_CENTER) -> np.ndarray:
    """HxWx3 uint8 源图 -> IMG_H x IMG_W x3 uint8 极坐标展开图。"""
    height, width = arr.shape[:2]
    cx, cy = center
    # 双线性插值需要在源图内取到邻居像素：半径最大 55.5，floor+1 最多到 cx+56。
    if not (INNER_R + 1 <= cx <= width - OUTER_R - 1 and INNER_R + 1 <= cy <= height - OUTER_R - 1):
        raise ValueError(
            f"image {width}x{height} too small for ROI center {center} "
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
    """读取源 PNG 为 RGB；若有 alpha，先按环形裁剪语义把透明像素 RGB 置零。"""
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
    out = unwrap_polar(canvas, center=(cx, cy))
    for azimuth_deg, radius in dots:
        row = int(round(radius - 0.5 - INNER_R))
        col = int(out[row].sum(axis=1).argmax())  # 该行最亮的列
        if abs(col - azimuth_deg) > 1:
            raise RuntimeError(
                f"azimuth self-check failed: dot at {azimuth_deg}° landed in column {col}"
            )
    print("azimuth self-check ok: 0°->col 0 (north), clockwise, 1°/column")


def generate_processed(fmt: str) -> list[str]:
    """清空 data/processed，从 data/raw 全量生成 fmt 格式样本，返回文件名列表。"""
    pngs = sorted(RAW_DIR.glob("*.png"))
    if not pngs:
        raise SystemExit(f"no png found in {RAW_DIR}")
    clear_pngs(PROCESSED)

    if fmt == "rgba":
        masks: dict[tuple[int, int], np.ndarray] = {}
        for i, src in enumerate(pngs, 1):
            with Image.open(src) as im:
                size = im.size
            if size not in masks:
                masks[size] = ring_mask(*size)
            process_rgba(src, PROCESSED / src.name, masks[size])
            if i % 250 == 0 or i == len(pngs):
                print(f"[{i}/{len(pngs)}] {src.name}")
    else:
        for i, src in enumerate(pngs, 1):
            arr = load_source_rgb(src)
            Image.fromarray(unwrap_polar(arr), "RGB").save(PROCESSED / src.name)
            if i % 250 == 0 or i == len(pngs):
                print(f"[{i}/{len(pngs)}] {src.name}")

    processed_names = png_names(PROCESSED)
    input_names = sorted(p.name for p in pngs)
    if processed_names != input_names:
        raise RuntimeError("processed PNG names do not exactly match raw PNG names")
    if fmt == "rgba":
        # 自检：全部输出的透明像素 RGB 必须为零；第一张的环内像素数
        # 应约为 π*(56²-11²) ≈ 9472。
        for name in processed_names:
            with Image.open(PROCESSED / name) as im:
                if im.size != (BOX, BOX) or im.mode != "RGBA":
                    raise RuntimeError(f"invalid processed image: {name}")
                arr = np.asarray(im, dtype=np.uint8).copy()
            transparent = arr[..., 3] == 0
            if np.any(transparent & np.any(arr[..., :3] != 0, axis=-1)):
                raise RuntimeError(f"transparent RGB is not zeroed: {name}")
        with Image.open(PROCESSED / processed_names[0]) as im:
            alpha = np.asarray(im.convert("RGBA"), dtype=np.uint8)[..., 3]
        n = int((alpha > 0).sum())
        print(f"ring-pixel self-check: {processed_names[0]} = {n} (expected ~{np.pi * (OUTER_R**2 - INNER_R**2):.0f})")
    else:
        for name in processed_names:
            with Image.open(PROCESSED / name) as im:
                if im.size != (IMG_W, IMG_H) or im.mode != "RGB":
                    raise RuntimeError(f"invalid processed polar image: {name}")
        self_check_azimuth()
    print(f"processed={len(processed_names)} format={fmt} -> {PROCESSED}")
    return processed_names


# ================= 第 2 步：processed -> train/val =================

def stratify_split(processed_names: list[str]) -> tuple[list[str], list[str]]:
    """随机切分：按 30 度角度分箱，每箱随机留出约 15%（至少一张）作验证集。"""
    buckets: dict[int, list[str]] = {i: [] for i in range(BIN_COUNT)}
    for name in processed_names:
        buckets[int(parse_angle(Path(name)) // ANGLE_BIN_DEGREES)].append(name)
    rng = random.Random(SEED)
    train_names: list[str] = []
    val_names: list[str] = []
    for bucket in buckets.values():
        bucket = sorted(bucket)
        rng.shuffle(bucket)
        if len(bucket) <= 1:
            train_names.extend(bucket)
            continue
        count = max(1, round(len(bucket) * VALIDATION_FRACTION))
        count = min(count, len(bucket) - 1)
        val_names.extend(bucket[:count])
        train_names.extend(bucket[count:])
    train_names.sort()
    val_names.sort()
    if not train_names or not val_names or sorted(train_names + val_names) != processed_names:
        raise RuntimeError("invalid train/validation split")
    return train_names, val_names


def manifest_split(processed_names: list[str]) -> tuple[list[str], list[str]]:
    """清单切分：val = 清单 ∩ processed（严格校验），train = 其余全部。"""
    if not VAL_MANIFEST.exists():
        raise SystemExit(f"{VAL_MANIFEST} missing; manifest split requires a validation manifest")
    manifest = load_json(VAL_MANIFEST)
    if manifest.get("version") != VAL_MANIFEST_VERSION:
        raise ValueError(f"{VAL_MANIFEST}: unsupported version {manifest.get('version')!r}")
    val_names = validate_manifest_names(manifest.get("files", []), set(processed_names), "val manifest")
    if not val_names:
        raise SystemExit(f"{VAL_MANIFEST}: manifest is empty")
    available = set(processed_names)
    train_names = sorted(available - set(val_names))
    if not train_names:
        raise SystemExit(f"{VAL_MANIFEST}: manifest covers every processed file; train set would be empty")
    return train_names, val_names


def copy_split(train_names: list[str], val_names: list[str]) -> None:
    clear_pngs(TRAIN)
    clear_pngs(VAL)
    for name in train_names:
        shutil.copy2(PROCESSED / name, TRAIN / name)
    for name in val_names:
        shutil.copy2(PROCESSED / name, VAL / name)
    if png_names(TRAIN) != train_names or png_names(VAL) != val_names:
        raise RuntimeError("copied train/validation files do not match the split")
    print(f"train={len(train_names)} -> {TRAIN}")
    print(f"val={len(val_names)} -> {VAL}")


def write_split_manifest(fmt: str, mode: str, train_names: list[str], val_names: list[str]) -> None:
    record: dict = {
        "version": SPLIT_MANIFEST_VERSION,
        "input_format": fmt,
        "split_mode": mode,
        "train_files": train_names,
        "val_files": val_names,
    }
    if mode == "random":
        record["seed"] = SEED
        record["validation_fraction"] = VALIDATION_FRACTION
        record["angle_bin_degrees"] = ANGLE_BIN_DEGREES
    else:
        record["val_manifest"] = str(VAL_MANIFEST.relative_to(ROOT))
    atomic_json_dump(SPLIT_MANIFEST, record)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input", choices=("rgba", "polar"), required=True,
                        help="processed sample format: rgba ring crop or polar unwrap")
    parser.add_argument("--split", choices=("random", "manifest"), default="random",
                        help="split mode: seeded random (default) or val_manifest.json-driven")
    args = parser.parse_args()

    processed_names = generate_processed(args.input)

    if args.split == "manifest":
        train_names, val_names = manifest_split(processed_names)
    else:
        train_names, val_names = stratify_split(processed_names)
    if set(train_names) & set(val_names) or sorted(train_names + val_names) != processed_names:
        raise RuntimeError("train/validation split does not exactly cover processed files")
    copy_split(train_names, val_names)
    write_split_manifest(args.input, args.split, train_names, val_names)
    print(f"split_mode={args.split} manifest={SPLIT_MANIFEST}")


if __name__ == "__main__":
    main()
