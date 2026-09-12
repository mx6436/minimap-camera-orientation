"""数据前处理：从 data/raw 生成模型输入样本。

polar（默认）：极坐标展开。训练/验证划分由 data/val_manifest.json 声明，
train = processed 全集减去清单所列验证集。train/val 是 processed 的符号链接视图，
内容始终反映 processed 当前状态，悬空链接在生成时校验。

ref：以 MapLocator 批量定位产物（data/locator/locate.jsonl）为姿态来源，参考底图 =
zone 资产在 (x, y) 处、与观测同视野的裁剪（按 zone 的 ZoneTemplateScale：
ValleyIV_Base 裁 ROI*15/16 再缩回 118x120，其余 1:1 直接裁）；观测与参考各自极坐标
展开后拼接为 7 通道 [obs.BGR, ref.BGR, ref.A]（不预先相减）。参考 BGR = 观测背底合成
black_ref + obs_roi*(1 - alpha/255)（ROI 域、四舍五入回 uint8；alpha==0 处逐像素等于
观测），ref.A = 资产原始连续 alpha（0 = 参考缺失）。样本范围 = 定位产物中
accepted=true 的记录；定位不可用（失败/held/低分）与资产缺失的样本跳过并计数。
val = manifest ∩ processed，manifest 引用但无 ref 输出的样本跳过并计数；引用
data/raw 中不存在的名字仍报错。

每种模式各自清空并重写自己的 processed/train/val 目录；data/raw 与
data/val_manifest.json 永不被脚本改动。
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from endfield.data_utils import (
    load_json,
    png_names,
    validate_manifest_names,
)
from endfield.locate import accept, load_records, zone_asset_path
from endfield.polar import (
    IMG_H,
    IMG_W,
    INNER_R,
    OUTER_R,
    ROI_CENTER,
    imread_png,
    load_source_bgr,
    unwrap,
)
from endfield.ref import (
    MAP_ASSETS_ROOT,
    REF_SUBDIR,
    load_reference_image,
    observed_roi,
    ref_strip,
    zone_scale,
)

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
TRAIN = ROOT / "data" / "train"
VAL = ROOT / "data" / "val"
PROCESSED_REF = ROOT / "data" / "processed_ref"
TRAIN_REF = ROOT / "data" / "train_ref"
VAL_REF = ROOT / "data" / "val_ref"
VAL_MANIFEST = ROOT / "data" / "val_manifest.json"
LOCATE_PATH = ROOT / "data" / "locator" / "locate.jsonl"

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
        strip = unwrap(load_source_bgr(src), *ROI_CENTER, INNER_R, OUTER_R)
        if not cv2.imwrite(str(PROCESSED / src.name), strip):
            raise RuntimeError(f"failed to write {PROCESSED / src.name}")
        if i % 250 == 0 or i == len(pngs):
            print(f"[{i}/{len(pngs)}] {src.name}")

    processed_names = png_names(PROCESSED)
    input_names = sorted(p.name for p in pngs)
    if processed_names != input_names:
        raise RuntimeError("processed PNG names do not exactly match raw PNG names")
    for name in processed_names:
        if imread_png(PROCESSED / name).shape != (IMG_H, IMG_W, 3):
            raise RuntimeError(f"invalid processed polar image: {name}")
    print(f"processed={len(processed_names)} format=polar -> {PROCESSED}")
    return processed_names


def load_val_manifest(available_names: set[str], manifest_path: Path = VAL_MANIFEST) -> list[str]:
    """读取并校验 val manifest：版本、重复名、引用名必须存在于 available_names。"""
    if not manifest_path.exists():
        raise SystemExit(f"{manifest_path} missing; manifest split requires a validation manifest")
    manifest = load_json(manifest_path)
    if manifest.get("version") != VAL_MANIFEST_VERSION:
        raise ValueError(f"{manifest_path}: unsupported version {manifest.get('version')!r}")
    return validate_manifest_names(manifest.get("files", []), available_names, "val manifest")


def manifest_split(
    processed_names: list[str], manifest_path: Path = VAL_MANIFEST
) -> tuple[list[str], list[str]]:
    val_names = load_val_manifest(set(processed_names), manifest_path)
    if not val_names:
        raise SystemExit(f"{manifest_path}: manifest is empty")
    available = set(processed_names)
    train_names = sorted(available - set(val_names))
    if not train_names:
        raise SystemExit(
            f"{manifest_path}: manifest covers every processed file; train set would be empty"
        )
    return train_names, val_names


def accepted_split(
    processed_names: list[str],
    raw_names: list[str],
    manifest_path: Path = VAL_MANIFEST,
) -> tuple[list[str], list[str], list[str]]:
    """accepted 管线的划分（ref 模式）：val = manifest ∩ processed。

    没有 processed 输出的 manifest 样本（定位不可用 / 资产缺失）跳过返回；
    manifest 校验对照 data/raw 全集：拼写错误/重复名照旧报错。
    """
    manifest_names = load_val_manifest(set(raw_names), manifest_path)
    processed = set(processed_names)
    val_names = sorted(processed & set(manifest_names))
    skipped = sorted(set(manifest_names) - processed)
    train_names = sorted(processed - set(val_names))
    if not val_names:
        raise SystemExit(
            f"{manifest_path}: no manifest sample has a ref output; val set would be empty"
        )
    if not train_names:
        raise SystemExit(
            f"{manifest_path}: manifest covers every ref sample; train set would be empty"
        )
    return train_names, val_names, skipped


def link_split(
    train_names: list[str],
    val_names: list[str],
    train_dir: Path = TRAIN,
    val_dir: Path = VAL,
    processed_dir: Path = PROCESSED,
    subdirs: tuple[str, ...] = (),
) -> None:
    """把划分后的样本链接到 train/val 目录；subdirs 是并行子树（ref 的 ref/）。

    train/val 是 processed 的符号链接视图，内容始终反映 processed 当前状态，
    悬空链接在生成时校验。
    """
    for subdir in ("", *subdirs):
        for directory, names in ((train_dir, train_names), (val_dir, val_names)):
            target_dir = directory / subdir
            clear_pngs(target_dir)
            # 从 train/val 的并行子树回到 processed 的同一子树：顶层一层，子目录两层
            prefix = Path("..") if not subdir else Path("..", "..")
            for name in names:
                (target_dir / name).symlink_to(prefix / processed_dir.name / subdir / name)
                if not (target_dir / name).is_file():
                    raise RuntimeError(f"split link does not resolve: {target_dir / name}")
            if png_names(target_dir) != names:
                raise RuntimeError(f"linked {target_dir} files do not match the split")
    print(f"train={len(train_names)} -> {train_dir}")
    print(f"val={len(val_names)} -> {val_dir}")


def accepted_records(locate_path: Path) -> tuple[dict[str, dict], dict[str, str]]:
    """定位产物 -> (accepted 记录表, name -> 跳过原因)；无产物 / 无 accepted 硬报错。"""
    records = load_records(locate_path)
    if not records:
        raise SystemExit(f"no MapLocator records: {locate_path} (run locate_dataset.py first)")
    accepted: dict[str, dict] = {}
    skipped: dict[str, str] = {}
    for name, record in records.items():
        ok, reason = accept(record)
        if ok:
            accepted[name] = record
        else:
            skipped[name] = reason
    if not accepted:
        raise SystemExit(f"no accepted MapLocator records: {locate_path}")
    return accepted, skipped


def generate_processed_ref(
    raw_dir: Path = RAW_DIR,
    locate_path: Path = LOCATE_PATH,
    assets_root: Path = MAP_ASSETS_ROOT,
    processed_dir: Path = PROCESSED_REF,
) -> tuple[list[str], dict[str, str]]:
    """对 accepted 定位记录生成 ref 两路展开条带（观测 3ch / 参考 BGRA）。

    每条记录用 `endfield.ref.ref_strip` 算出 7 通道张量，再按
    `[obs.BGR, ref.BGR, ref.A]` 切成两路落盘：观测 `<name>.png`、参考 `ref/<name>.png`。
    返回 (产物名, name -> 跳过原因)；跳过原因含 accept 门原因与 asset_missing。
    """
    accepted, skipped = accepted_records(locate_path)

    clear_pngs(processed_dir)
    clear_pngs(processed_dir / REF_SUBDIR)
    assets: dict[Path, np.ndarray] = {}
    names: list[str] = []
    for index, name in enumerate(sorted(accepted), 1):
        record = accepted[name]
        zone = str(record.get("zone", ""))
        asset_path = zone_asset_path(zone, assets_root)
        if asset_path is None:
            skipped[name] = "asset_missing"
            continue
        asset = assets.get(asset_path)
        if asset is None:
            asset = load_reference_image(asset_path)
            assets[asset_path] = asset
        observed = observed_roi(load_source_bgr(raw_dir / name))
        ref = ref_strip(
            observed, asset, float(record["x"]), float(record["y"]), zone_scale(zone)
        )
        observed_path, reference_path = processed_dir / name, processed_dir / REF_SUBDIR / name
        if not cv2.imwrite(str(observed_path), ref[..., :3]):
            raise RuntimeError(f"failed to write {observed_path}")
        if not cv2.imwrite(str(reference_path), ref[..., 3:]):
            raise RuntimeError(f"failed to write {reference_path}")
        names.append(name)
        if index % 250 == 0 or index == len(accepted):
            print(f"[{index}/{len(accepted)}] {name}")

    if png_names(processed_dir) != names or png_names(processed_dir / REF_SUBDIR) != names:
        raise RuntimeError("processed ref PNG names do not exactly match written samples")
    for name in names:
        if imread_png(processed_dir / name).shape != (IMG_H, IMG_W, 3):
            raise RuntimeError(f"invalid processed ref observation image: {name}")
        if imread_png(processed_dir / REF_SUBDIR / name).shape != (IMG_H, IMG_W, 4):
            raise RuntimeError(f"invalid processed ref reference image: {name}")
    reasons = Counter(skipped.values())
    print(
        f"ref: accepted={len(accepted)} processed={len(names)} skipped={len(skipped)} "
        f"{dict(sorted(reasons.items()))} -> {processed_dir}"
    )
    return names, skipped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--mode",
        choices=("polar", "ref"),
        default="polar",
        help=(
            "前处理模式：polar 极坐标展开（默认）；ref 参考输入"
            "（观测/参考各展开后并列，需先跑 locate_dataset.py）"
        ),
    )
    parser.add_argument(
        "--map-assets-root",
        type=Path,
        default=MAP_ASSETS_ROOT,
        help="MapLocator 底图资产目录（ref 模式；默认本地 MaaEnd 工作副本）",
    )
    return parser.parse_args()


def run_ref(
    assets_root: Path,
    raw_dir: Path = RAW_DIR,
    locate_path: Path = LOCATE_PATH,
    manifest_path: Path = VAL_MANIFEST,
    processed_dir: Path = PROCESSED_REF,
    train_dir: Path = TRAIN_REF,
    val_dir: Path = VAL_REF,
) -> None:
    """ref 全管线：两路展开产物 -> manifest 划分 -> train/val 符号链接视图。"""
    raw_names = sorted(path.name for path in raw_dir.glob("*.png"))
    if not raw_names:
        raise SystemExit(f"no png found in {raw_dir}")
    processed_names, skipped = generate_processed_ref(
        raw_dir, locate_path, assets_root, processed_dir
    )
    train_names, val_names, manifest_skipped = accepted_split(
        processed_names, raw_names, manifest_path
    )
    reasons = Counter(skipped.get(name, "no_locate_record") for name in manifest_skipped)
    print(f"val manifest skipped={len(manifest_skipped)} {dict(sorted(reasons.items()))}")
    if set(train_names) & set(val_names) or sorted(train_names + val_names) != processed_names:
        raise RuntimeError("ref train/validation split does not exactly cover processed files")
    link_split(
        train_names,
        val_names,
        train_dir,
        val_dir,
        processed_dir,
        subdirs=(REF_SUBDIR,),
    )


def main() -> None:
    args = parse_args()
    if args.mode == "ref":
        run_ref(args.map_assets_root)
        return
    processed_names = generate_processed()
    train_names, val_names = manifest_split(processed_names)
    if set(train_names) & set(val_names) or sorted(train_names + val_names) != processed_names:
        raise RuntimeError("train/validation split does not exactly cover processed files")
    link_split(train_names, val_names)


if __name__ == "__main__":
    main()
