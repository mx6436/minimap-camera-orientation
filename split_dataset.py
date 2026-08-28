#!/usr/bin/env python3
"""Copy a reproducible, angle-stratified train/validation split from processed images."""
from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

from data_utils import SEED, atomic_json_dump, load_json, parse_angle, png_names, validate_manifest_names

ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "data" / "processed"
PROCESSED_POLAR = ROOT / "data" / "processed_polar"
TRAIN = ROOT / "data" / "train"
VAL = ROOT / "data" / "val"
TRAIN_POLAR = ROOT / "data" / "train_polar"
VAL_POLAR = ROOT / "data" / "val_polar"
SPLIT_MANIFEST = ROOT / "data" / "split_manifest.json"

VALIDATION_FRACTION = 0.15
ANGLE_BIN_DEGREES = 30
BIN_COUNT = 360 // ANGLE_BIN_DEGREES


def clear_pngs(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for path in directory.glob("*.png"):
        path.unlink()


def stratify_split(processed_names: list[str]) -> tuple[list[str], list[str]]:
    """Split by 30-degree angle bins so every bin keeps at least one validation image."""
    buckets: dict[int, list[str]] = {i: [] for i in range(BIN_COUNT)}
    for name in processed_names:
        buckets[parse_angle(Path(name)) // ANGLE_BIN_DEGREES].append(name)
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


def read_or_create_split(processed_names: list[str], resplit: bool) -> tuple[list[str], list[str]]:
    available = set(processed_names)
    if SPLIT_MANIFEST.exists() and not resplit:
        manifest = load_json(SPLIT_MANIFEST)
        if (
            manifest.get("version") != 1
            or manifest.get("seed") != SEED
            or manifest.get("source") != "data/processed"
            or manifest.get("validation_fraction") != VALIDATION_FRACTION
            or manifest.get("angle_bin_degrees") != ANGLE_BIN_DEGREES
        ):
            raise ValueError(f"{SPLIT_MANIFEST}: split policy mismatch; use --resplit")
        if manifest.get("input_files") != processed_names:
            raise ValueError(f"{SPLIT_MANIFEST}: processed file list changed; use --resplit")
        train_names = manifest.get("train_files")
        val_names = manifest.get("val_files")
        if (
            not isinstance(train_names, list)
            or not isinstance(val_names, list)
            or not all(isinstance(name, str) for name in train_names + val_names)
            or train_names != sorted(train_names)
            or val_names != sorted(val_names)
            or sorted(train_names + val_names) != processed_names
            or len(train_names) != len(set(train_names))
            or len(val_names) != len(set(val_names))
            or set(train_names) & set(val_names)
            or not set(train_names).issubset(available)
            or not set(val_names).issubset(available)
        ):
            raise ValueError(f"{SPLIT_MANIFEST}: invalid file lists; use --resplit")
        return train_names, val_names
    return stratify_split(processed_names)


def copy_polar_from_manifest() -> None:
    """按现有 manifest 复制极坐标展开图，保证与 RGBA 管线逐文件同划分。

    不支持 --resplit：划分必须与 RGBA 管线完全一致，重新掷骰子只能
    在 RGBA 管线上进行。
    """
    if not SPLIT_MANIFEST.exists():
        raise SystemExit(f"{SPLIT_MANIFEST} missing; run split_dataset.py (rgba) first")
    manifest = load_json(SPLIT_MANIFEST)
    if (
        manifest.get("version") != 1
        or manifest.get("seed") != SEED
        or manifest.get("source") != "data/processed"
        or manifest.get("validation_fraction") != VALIDATION_FRACTION
        or manifest.get("angle_bin_degrees") != ANGLE_BIN_DEGREES
    ):
        raise SystemExit(f"{SPLIT_MANIFEST}: split policy mismatch")
    polar_names = png_names(PROCESSED_POLAR)
    if not polar_names:
        raise SystemExit(f"no PNG files found in {PROCESSED_POLAR}; run unwrap_polar.py first")
    train_names = validate_manifest_names(manifest.get("train_files", []), set(polar_names), "train")
    val_names = validate_manifest_names(manifest.get("val_files", []), set(polar_names), "val")
    if sorted(train_names + val_names) != polar_names:
        raise SystemExit(f"{SPLIT_MANIFEST}: split does not exactly cover {PROCESSED_POLAR} files")

    clear_pngs(TRAIN_POLAR)
    clear_pngs(VAL_POLAR)
    for name in train_names:
        shutil.copy2(PROCESSED_POLAR / name, TRAIN_POLAR / name)
    for name in val_names:
        shutil.copy2(PROCESSED_POLAR / name, VAL_POLAR / name)
    if png_names(TRAIN_POLAR) != train_names or png_names(VAL_POLAR) != val_names:
        raise RuntimeError("copied train_polar/val_polar files do not match the split manifest")
    print(f"processed_polar={len(polar_names)} train_polar={len(train_names)} val_polar={len(val_names)}")
    print(f"manifest={SPLIT_MANIFEST} (reused, files identical to rgba pipeline)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", choices=("rgba", "polar"), default="rgba",
                        help="rgba: split data/processed (default); polar: copy data/processed_polar by the existing manifest")
    parser.add_argument("--resplit", action="store_true", help="create a new seeded train/validation split (rgba only)")
    args = parser.parse_args()
    if args.input == "polar":
        if args.resplit:
            raise SystemExit("--resplit is not allowed with --input polar")
        copy_polar_from_manifest()
        return

    processed_names = png_names(PROCESSED)
    if not processed_names:
        raise SystemExit(f"no PNG files found in {PROCESSED}")
    train_names, val_names = read_or_create_split(processed_names, args.resplit)
    if set(train_names) & set(val_names) or sorted(train_names + val_names) != processed_names:
        raise RuntimeError("train/validation split does not exactly cover processed files")

    clear_pngs(TRAIN)
    clear_pngs(VAL)
    for name in train_names:
        shutil.copy2(PROCESSED / name, TRAIN / name)
    for name in val_names:
        shutil.copy2(PROCESSED / name, VAL / name)

    if args.resplit or not SPLIT_MANIFEST.exists():
        atomic_json_dump(SPLIT_MANIFEST, {
            "version": 1,
            "seed": SEED,
            "source": "data/processed",
            "validation_fraction": VALIDATION_FRACTION,
            "angle_bin_degrees": ANGLE_BIN_DEGREES,
            "input_files": processed_names,
            "train_files": train_names,
            "val_files": val_names,
        })

    if png_names(TRAIN) != train_names or png_names(VAL) != val_names:
        raise RuntimeError("copied train/validation files do not match the split manifest")
    print(f"processed={len(processed_names)} train={len(train_names)} val={len(val_names)}")
    print(f"manifest={SPLIT_MANIFEST}")


if __name__ == "__main__":
    main()
