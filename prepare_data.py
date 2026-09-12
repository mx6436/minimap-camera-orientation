"""数据前处理：从 data/raw 生成模型输入样本。

polar（默认）：极坐标展开。训练/验证划分由 data/val_manifest.json 声明，
train = processed 全集减去清单所列验证集。train/val 是 processed 的符号链接视图，
内容始终反映 processed 当前状态，悬空链接在生成时校验。产物写完后在 processed 目录
落 `.preprocess.json` 缓存戳（定义哈希 + 图版本 + 输入名单，见
`endfield/preprocess_cache.py`）：戳与当前定义一致且产物文件齐全时跳过重写，
定义变更 / 样本增删 / `--force` 触发重生成。

ref：以 MapLocator 批量定位产物（data/locator/locate.jsonl）为姿态来源；观测与参考
条带由定义模块 `endfield/preprocess.py` 一次生成：资产坐标 =
`(x, y) + (q_roi - 极点) * scale`（scale 取定位记录的 ZoneTemplateScale），参考缺失
（裁剪越界 / 资产 alpha）以 ref.A 表达，条带域一次合成
`ref.BGR = rgb*(a/255) + obs*(1-a/255)`（alpha==0 处逐像素等于观测）。两路拼接为
7 通道 `[obs.BGR, ref.BGR, ref.A]`（不预先相减）。样本范围 = 定位产物中
accepted=true 的记录；定位不可用（失败/held/低分）与资产缺失的样本跳过并计数。
val = manifest ∩ processed，manifest 引用但无 ref 输出的样本跳过并计数；引用
data/raw 中不存在的名字仍报错。

每种模式各自清空并重写自己的 processed/train/val 目录；data/raw 与
data/val_manifest.json 永不被脚本改动。ref 模式的定位产物单独维护在 data/locator/
（不随脚本清空）。两种模式的 processed 目录都挂 `.preprocess.json` 缓存戳
（定义哈希 + 图版本 + 输入指纹，见 `endfield/preprocess_cache.py`）：与当前定义
一致且产物文件齐全时跳过重写，定义变更 / 输入变更 / `--force` 触发重生成。
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch

from endfield import preprocess, preprocess_cache
from endfield.data_utils import (
    load_json,
    png_names,
    validate_manifest_names,
)
from endfield.locate import accept, load_records, record_scale, zone_asset_path
from endfield.polar import (
    IMG_H,
    IMG_W,
    imread_png,
    load_source_bgr,
)
from endfield.ref import (
    MAP_ASSETS_ROOT,
    REF_SUBDIR,
    load_reference_image,
    observed_roi,
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

# 原始截图解码（cv2.imread 释放 GIL）是管线大头：每张 ~10ms，而定义模块前处理 ~1ms。
# 解码/ROI 提取按 CPU 数并行；定义模块调用始终在主线程串行，产物与 workers=1 逐字节一致。
IO_WORKERS = min(16, os.cpu_count() or 1)
IO_CHUNK = 256


def _parallel_load[Item](
    items: Sequence[Item], load: Callable[[Item], np.ndarray], workers: int
) -> Iterator[np.ndarray]:
    """按序、分块并行跑 I/O 密集的 `load(item)`；峰值内存只保留一个块的结果。"""
    if workers <= 1 or not items:
        for item in items:
            yield load(item)
        return
    with ThreadPoolExecutor(max_workers=min(workers, len(items))) as pool:
        for start in range(0, len(items), IO_CHUNK):
            chunk = items[start : start + IO_CHUNK]
            yield from pool.map(load, chunk)


def clear_pngs(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for path in directory.glob("*.png"):
        path.unlink()


def generate_processed(
    raw_dir: Path = RAW_DIR,
    processed_dir: Path = PROCESSED,
    force: bool = False,
    workers: int = IO_WORKERS,
) -> list[str]:
    """raw 全量 -> polar 条带落盘；缓存戳命中且产物齐全时跳过重写。"""
    pngs = sorted(raw_dir.glob("*.png"))
    if not pngs:
        raise SystemExit(f"no png found in {raw_dir}")
    input_names = sorted(p.name for p in pngs)
    if (
        not force
        and png_names(processed_dir) == input_names
        and preprocess_cache.cache_hit(processed_dir, "polar", input_names)
    ):
        print(f"processed={len(input_names)} format=polar cache=hit -> {processed_dir}")
        return input_names

    preprocess_cache.remove_stamp(processed_dir)
    clear_pngs(processed_dir)

    def load_roi(src: Path) -> np.ndarray:
        return observed_roi(load_source_bgr(src))

    total = len(pngs)
    for i, (src, roi) in enumerate(
        zip(pngs, _parallel_load(pngs, load_roi, workers), strict=True), 1
    ):
        strip = preprocess.observed_strip(roi)
        if not cv2.imwrite(str(processed_dir / src.name), strip):
            raise RuntimeError(f"failed to write {processed_dir / src.name}")
        if i % 250 == 0 or i == total:
            print(f"[{i}/{total}] {src.name}", flush=True)

    processed_names = png_names(processed_dir)
    if processed_names != input_names:
        raise RuntimeError("processed PNG names do not exactly match raw PNG names")
    for name in processed_names:
        if imread_png(processed_dir / name).shape != (IMG_H, IMG_W, 3):
            raise RuntimeError(f"invalid processed polar image: {name}")
    stamp = preprocess_cache.write_stamp(processed_dir, "polar", input_names)
    print(
        f"processed={len(processed_names)} format=polar cache=miss "
        f"definition_hash={stamp['definition_hash'][:12]} -> {processed_dir}"
    )
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


def ref_input_entries(resolved: list[tuple[str, dict, Path]]) -> list[str]:
    """ref 前处理的输入指纹条目：样本名 + 它消费的定位字段（zone/x/y/scale）。

    其他定位字段（latencyMs 等）不进指纹：重跑 locate_dataset.py 只刷新时间戳时
    不该触发数据重算；资产路径由 zone 决定，不另记。
    """
    entries = []
    for name, record, _ in resolved:
        entries.append(
            json.dumps(
                [
                    name,
                    str(record.get("zone", "")),
                    float(record["x"]),
                    float(record["y"]),
                    record_scale(record),
                ],
                ensure_ascii=True,
                separators=(",", ":"),
            )
        )
    return entries


def _resolve_ref_inputs(
    accepted: dict[str, dict], assets_root: Path, skipped: dict[str, str]
) -> list[tuple[str, dict, Path]]:
    """accepted 记录 -> 可生成样本 (name, record, asset_path)；缺资产的计入 skipped。"""
    resolved: list[tuple[str, dict, Path]] = []
    for name in sorted(accepted):
        record = accepted[name]
        asset_path = zone_asset_path(str(record.get("zone", "")), assets_root)
        if asset_path is None:
            skipped[name] = "asset_missing"
            continue
        resolved.append((name, record, asset_path))
    return resolved


def _skip_reasons(skipped: dict[str, str]) -> dict[str, int]:
    return dict(sorted(Counter(skipped.values()).items()))


def generate_processed_ref(
    raw_dir: Path = RAW_DIR,
    locate_path: Path = LOCATE_PATH,
    assets_root: Path = MAP_ASSETS_ROOT,
    processed_dir: Path = PROCESSED_REF,
    force: bool = False,
    workers: int = IO_WORKERS,
) -> tuple[list[str], dict[str, str]]:
    """对 accepted 定位记录生成 ref 两路展开条带（观测 3ch / 参考 BGRA）。

    每条记录由定义模块 `preprocess.strips` 一次算出两路条带，再按
    `[obs.BGR, ref.BGR, ref.A]` 切成两路落盘：观测 `<name>.png`、参考 `ref/<name>.png`。
    返回 (产物名, name -> 跳过原因)；跳过原因含 accept 门原因与 asset_missing。
    缓存戳命中且两路产物齐全时跳过重写。
    """
    accepted, skipped = accepted_records(locate_path)
    resolved = _resolve_ref_inputs(accepted, assets_root, skipped)
    entries = ref_input_entries(resolved)
    names = [name for name, _, _ in resolved]
    if (
        not force
        and png_names(processed_dir) == names
        and png_names(processed_dir / REF_SUBDIR) == names
        and preprocess_cache.cache_hit(processed_dir, "ref", entries)
    ):
        print(
            f"ref: accepted={len(accepted)} processed={len(names)} skipped={len(skipped)} "
            f"{_skip_reasons(skipped)} cache=hit -> {processed_dir}"
        )
        return names, skipped

    preprocess_cache.remove_stamp(processed_dir)
    clear_pngs(processed_dir)
    clear_pngs(processed_dir / REF_SUBDIR)
    # 每 zone 的底图只转一次 float32（逐样本整图 Cast 是热点）；采样语义仍在定义模块
    prepared_assets: dict[Path, torch.Tensor] = {}

    def load_observation(item: tuple[str, dict, Path]) -> np.ndarray:
        return observed_roi(load_source_bgr(raw_dir / item[0]))

    for index, ((name, record, asset_path), observed) in enumerate(
        zip(resolved, _parallel_load(resolved, load_observation, workers), strict=True), 1
    ):
        asset_float = prepared_assets.get(asset_path)
        if asset_float is None:
            asset_float = prepared_assets[asset_path] = preprocess.prepare_asset(
                load_reference_image(asset_path)
            )
        observed_strip, reference = preprocess.strips_prepared(
            observed,
            asset_float,
            float(record["x"]),
            float(record["y"]),
            record_scale(record),
        )
        observed_path, reference_path = processed_dir / name, processed_dir / REF_SUBDIR / name
        if not cv2.imwrite(str(observed_path), observed_strip):
            raise RuntimeError(f"failed to write {observed_path}")
        if not cv2.imwrite(str(reference_path), reference):
            raise RuntimeError(f"failed to write {reference_path}")
        if index % 250 == 0 or index == len(resolved):
            print(f"[{index}/{len(resolved)}] {name}", flush=True)

    if png_names(processed_dir) != names or png_names(processed_dir / REF_SUBDIR) != names:
        raise RuntimeError("processed ref PNG names do not exactly match written samples")
    for name in names:
        if imread_png(processed_dir / name).shape != (IMG_H, IMG_W, 3):
            raise RuntimeError(f"invalid processed ref observation image: {name}")
        if imread_png(processed_dir / REF_SUBDIR / name).shape != (IMG_H, IMG_W, 4):
            raise RuntimeError(f"invalid processed ref reference image: {name}")
    stamp = preprocess_cache.write_stamp(processed_dir, "ref", entries)
    print(
        f"ref: accepted={len(accepted)} processed={len(names)} skipped={len(skipped)} "
        f"{_skip_reasons(skipped)} cache=miss definition_hash={stamp['definition_hash'][:12]} "
        f"-> {processed_dir}"
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
    parser.add_argument(
        "--force",
        action="store_true",
        help="忽略 processed 缓存戳命中，强制重生成",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=IO_WORKERS,
        help=f"原始图解码的并行线程数（默认 {IO_WORKERS}=min(16, CPU 数)；1 = 串行）",
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be >= 1")
    return args


def run_ref(
    assets_root: Path,
    raw_dir: Path = RAW_DIR,
    locate_path: Path = LOCATE_PATH,
    manifest_path: Path = VAL_MANIFEST,
    processed_dir: Path = PROCESSED_REF,
    train_dir: Path = TRAIN_REF,
    val_dir: Path = VAL_REF,
    force: bool = False,
    workers: int = IO_WORKERS,
) -> None:
    """ref 全管线：两路展开产物 -> manifest 划分 -> train/val 符号链接视图。"""
    raw_names = sorted(path.name for path in raw_dir.glob("*.png"))
    if not raw_names:
        raise SystemExit(f"no png found in {raw_dir}")
    processed_names, skipped = generate_processed_ref(
        raw_dir, locate_path, assets_root, processed_dir, force, workers
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


def run_polar(
    raw_dir: Path = RAW_DIR,
    manifest_path: Path = VAL_MANIFEST,
    processed_dir: Path = PROCESSED,
    train_dir: Path = TRAIN,
    val_dir: Path = VAL,
    force: bool = False,
    workers: int = IO_WORKERS,
) -> None:
    """polar 全管线：极坐标展开落盘 -> manifest 划分 -> train/val 符号链接视图。"""
    processed_names = generate_processed(raw_dir, processed_dir, force, workers)
    train_names, val_names = manifest_split(processed_names, manifest_path)
    if set(train_names) & set(val_names) or sorted(train_names + val_names) != processed_names:
        raise RuntimeError("train/validation split does not exactly cover processed files")
    link_split(train_names, val_names, train_dir, val_dir, processed_dir)


def main() -> None:
    args = parse_args()
    if args.mode == "ref":
        run_ref(args.map_assets_root, force=args.force, workers=args.workers)
        return
    run_polar(force=args.force, workers=args.workers)


if __name__ == "__main__":
    main()
