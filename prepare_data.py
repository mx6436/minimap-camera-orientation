#!/usr/bin/env python3
"""数据前处理：从 data/raw 生成极坐标展开样本并完成训练/验证划分。

流水线（一条命令完成，除 data/raw 与 data/val_manifest.json 外全部输出
在每次运行时清空重写，见 docs/adr/0001）：

  1. 清空 data/processed，从 data/raw 全量生成极坐标展开样本
     （360x44 RGB，展开约定见 CONTEXT.md「极坐标展开」与 polar.py）；
  2. 清空 data/train、data/val，按 --split 指定的模式划分并复制；
  3. 重写 data/split_manifest.json 记录本次划分结果。

切分模式 --split：
  random   按 SEED 与 30° 角度分箱随机留出约 15% 作验证集（每箱至少
           一张验证图）。同输入同种子结果确定，但新增 raw 数据后重跑
           会重新洗牌，已有验证文件可能被换出。
  manifest 清单切分：验证集成员由 data/val_manifest.json 直接指定，
           val = 清单 ∩ processed（清单引用 processed 中不存在的文件名
           则报错），train = processed 其余全部。用于留出地图（Held-out
           Map）的跨地图泛化验证：验证地图的样本一律不进入训练集。
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

from PIL import Image

from data_utils import (
    SEED,
    atomic_json_dump,
    load_json,
    parse_angle,
    png_names,
    validate_manifest_names,
)
from polar import IMG_H, IMG_W, INNER_R, OUTER_R, ROI_CENTER, load_source_rgb, self_check_azimuth, unwrap

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


def clear_pngs(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for path in directory.glob("*.png"):
        path.unlink()


def generate_processed() -> list[str]:
    pngs = sorted(RAW_DIR.glob("*.png"))
    if not pngs:
        raise SystemExit(f"no png found in {RAW_DIR}")
    clear_pngs(PROCESSED)

    for i, src in enumerate(pngs, 1):
        arr = load_source_rgb(src)
        Image.fromarray(unwrap(arr, *ROI_CENTER, INNER_R, OUTER_R), "RGB").save(PROCESSED / src.name)
        if i % 250 == 0 or i == len(pngs):
            print(f"[{i}/{len(pngs)}] {src.name}")

    processed_names = png_names(PROCESSED)
    input_names = sorted(p.name for p in pngs)
    if processed_names != input_names:
        raise RuntimeError("processed PNG names do not exactly match raw PNG names")
    for name in processed_names:
        with Image.open(PROCESSED / name) as im:
            if im.size != (IMG_W, IMG_H) or im.mode != "RGB":
                raise RuntimeError(f"invalid processed polar image: {name}")
    self_check_azimuth()
    print(f"processed={len(processed_names)} format=polar -> {PROCESSED}")
    return processed_names


def stratify_split(processed_names: list[str]) -> tuple[list[str], list[str]]:
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


def write_split_manifest(mode: str, train_names: list[str], val_names: list[str]) -> None:
    record: dict = {
        "version": SPLIT_MANIFEST_VERSION,
        "input_format": "polar",
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
    parser.add_argument("--split", choices=("random", "manifest"), default="random",
                        help="split mode: seeded random (default) or val_manifest.json-driven")
    args = parser.parse_args()

    processed_names = generate_processed()

    if args.split == "manifest":
        train_names, val_names = manifest_split(processed_names)
    else:
        train_names, val_names = stratify_split(processed_names)
    if set(train_names) & set(val_names) or sorted(train_names + val_names) != processed_names:
        raise RuntimeError("train/validation split does not exactly cover processed files")
    copy_split(train_names, val_names)
    write_split_manifest(args.split, train_names, val_names)
    print(f"split_mode={args.split} manifest={SPLIT_MANIFEST}")


if __name__ == "__main__":
    main()
