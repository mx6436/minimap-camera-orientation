"""数据前处理：从 data/raw 生成极坐标展开样本。

训练/验证划分由 data/val_manifest.json 声明，train = processed 全集减去清单所列验证集。
train/val 是 processed 的符号链接视图，内容始终反映 processed 当前状态，悬空链接在生成时校验。
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from endfield.data_utils import (
    load_json,
    png_names,
    validate_manifest_names,
)
from endfield.polar import (
    IMG_H,
    IMG_W,
    INNER_R,
    OUTER_R,
    ROI_CENTER,
    load_source_rgb,
    unwrap,
)

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
TRAIN = ROOT / "data" / "train"
VAL = ROOT / "data" / "val"
VAL_MANIFEST = ROOT / "data" / "val_manifest.json"

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
        Image.fromarray(unwrap(arr, *ROI_CENTER, INNER_R, OUTER_R), "RGB").save(
            PROCESSED / src.name
        )
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
    print(f"processed={len(processed_names)} format=polar -> {PROCESSED}")
    return processed_names


def manifest_split(processed_names: list[str]) -> tuple[list[str], list[str]]:
    if not VAL_MANIFEST.exists():
        raise SystemExit(f"{VAL_MANIFEST} missing; manifest split requires a validation manifest")
    manifest = load_json(VAL_MANIFEST)
    if manifest.get("version") != VAL_MANIFEST_VERSION:
        raise ValueError(f"{VAL_MANIFEST}: unsupported version {manifest.get('version')!r}")
    val_names = validate_manifest_names(
        manifest.get("files", []), set(processed_names), "val manifest"
    )
    if not val_names:
        raise SystemExit(f"{VAL_MANIFEST}: manifest is empty")
    available = set(processed_names)
    train_names = sorted(available - set(val_names))
    if not train_names:
        raise SystemExit(
            f"{VAL_MANIFEST}: manifest covers every processed file; train set would be empty"
        )
    return train_names, val_names


def link_split(train_names: list[str], val_names: list[str]) -> None:
    clear_pngs(TRAIN)
    clear_pngs(VAL)
    for directory, names in ((TRAIN, train_names), (VAL, val_names)):
        for name in names:
            (directory / name).symlink_to(Path("..") / PROCESSED.name / name)
            if not (directory / name).is_file():
                raise RuntimeError(f"split link does not resolve: {directory / name}")
    if png_names(TRAIN) != train_names or png_names(VAL) != val_names:
        raise RuntimeError("linked train/validation files do not match the split")
    print(f"train={len(train_names)} -> {TRAIN}")
    print(f"val={len(val_names)} -> {VAL}")


def main() -> None:
    processed_names = generate_processed()
    train_names, val_names = manifest_split(processed_names)
    if set(train_names) & set(val_names) or sorted(train_names + val_names) != processed_names:
        raise RuntimeError("train/validation split does not exactly cover processed files")
    link_split(train_names, val_names)


if __name__ == "__main__":
    main()
