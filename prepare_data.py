"""数据前处理：从 data/train_raw 与 data/val_raw 生成模型输入样本（polar / ref 两种模式）。"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch

from endfield import coord_filter, preprocess, preprocess_cache
from endfield.data_utils import png_names, union_png_samples
from endfield.locate import accept, load_records, record_scale, zone_asset_path
from endfield.polar import (
    IMG_H,
    IMG_W,
    imread_png,
    load_source_frame,
)
from endfield.ref import (
    MAP_ASSETS_ROOT,
    REF_SUBDIR,
    load_reference_image,
)

ROOT = Path(__file__).resolve().parent
TRAIN_RAW = ROOT / "data" / "train_raw"
VAL_RAW = ROOT / "data" / "val_raw"
PROCESSED = ROOT / "data" / "processed"
TRAIN = ROOT / "data" / "train"
VAL = ROOT / "data" / "val"
PROCESSED_REF = ROOT / "data" / "processed_ref"
TRAIN_REF = ROOT / "data" / "train_ref"
VAL_REF = ROOT / "data" / "val_ref"
LOCATE_PATH = ROOT / "data" / "locator" / "locate.jsonl"
# ZmdMap 标注数据（MaaEnd assets/data/ZmdMap 的本地镜像；坐标一致性过滤用）
ZMDMAP_DATA_ROOT = ROOT / "local" / "maplocator" / "data" / "ZmdMap"

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


def raw_samples(train_raw_dir: Path = TRAIN_RAW, val_raw_dir: Path = VAL_RAW) -> dict[str, Path]:
    """两侧原始目录并集 -> {样本名: 源文件}。"""
    return union_png_samples((train_raw_dir, val_raw_dir))


def directory_split(
    train_names: Sequence[str], val_names: Sequence[str]
) -> tuple[list[str], list[str]]:
    """目录即划分：两侧名单排序返回。"""
    train, val = sorted(train_names), sorted(val_names)
    if not train:
        raise SystemExit("train split is empty: no usable samples on the train side")
    if not val:
        raise SystemExit("val split is empty: no usable samples on the val side")
    return train, val


def generate_processed(
    samples: Mapping[str, Path],
    processed_dir: Path = PROCESSED,
    force: bool = False,
    workers: int = IO_WORKERS,
) -> list[str]:
    """样本并集 -> polar 条带落盘。"""
    input_names = sorted(samples)
    if not input_names:
        raise SystemExit("no raw png samples in data/train_raw and data/val_raw")
    if (
        not force
        and png_names(processed_dir) == input_names
        and preprocess_cache.cache_hit(processed_dir, "polar", input_names)
    ):
        print(f"processed={len(input_names)} format=polar cache=hit -> {processed_dir}")
        return input_names

    preprocess_cache.remove_stamp(processed_dir)
    clear_pngs(processed_dir)

    def load_roi(name: str) -> np.ndarray:
        return preprocess.observed_roi(load_source_frame(samples[name]))

    total = len(input_names)
    for i, (name, roi) in enumerate(
        zip(input_names, _parallel_load(input_names, load_roi, workers), strict=True), 1
    ):
        strip = preprocess.observed_strip(roi)
        if not cv2.imwrite(str(processed_dir / name), strip):
            raise RuntimeError(f"failed to write {processed_dir / name}")
        if i % 250 == 0 or i == total:
            print(f"[{i}/{total}] {name}", flush=True)

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


def link_split(
    train_names: list[str],
    val_names: list[str],
    train_dir: Path = TRAIN,
    val_dir: Path = VAL,
    processed_dir: Path = PROCESSED,
    subdirs: tuple[str, ...] = (),
) -> None:
    """把划分后的样本链接到 train/val 目录；subdirs 是并行子树（ref 的 ref/）。"""
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
    """定位产物 -> (accepted 记录表, name -> 跳过原因)。"""
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


def _print_side_skips(
    label: str, raw_names: Sequence[str], processed: set[str], skipped: Mapping[str, str]
) -> None:
    """打印一侧的可用/跳过分布（跳过样本已从该侧的划分视图剔除）。"""
    usable = sum(1 for name in raw_names if name in processed)
    reasons = Counter(skipped[name] for name in raw_names if name not in processed)
    print(
        f"{label}: samples={len(raw_names)} usable={usable} "
        f"skipped={sum(reasons.values())} {dict(sorted(reasons.items()))}"
    )


def filter_coord_consistent(
    resolved: list[tuple[str, dict, Path]], zmdmap_root: Path, assets_root: Path
) -> tuple[list[tuple[str, dict, Path]], dict[str, str]]:
    """标注坐标一致性过滤：返回 (保留样本, name -> 拒绝原因)。

    只在出现 MapTracker 命名样本时才读 ZmdMap/Base 换算数据（纯 zone 命名的数据集不需要镜像数据）。
    """
    kept: list[tuple[str, dict, Path]] = []
    rejected: dict[str, str] = {}
    data: coord_filter.FilterData | None = None
    for item in resolved:
        name, record, _ = item
        annotation = coord_filter.parse_annotation(name)
        if data is None and coord_filter.needs_conversion(annotation.zone):
            data = coord_filter.load_filter_data(zmdmap_root, assets_root)
        decision = coord_filter.evaluate(annotation, record, data)
        if decision.keep:
            kept.append(item)
        else:
            rejected[name] = decision.reason
    return kept, rejected


def generate_processed_ref(
    samples: Mapping[str, Path],
    locate_path: Path = LOCATE_PATH,
    assets_root: Path = MAP_ASSETS_ROOT,
    processed_dir: Path = PROCESSED_REF,
    force: bool = False,
    workers: int = IO_WORKERS,
    zmdmap_root: Path = ZMDMAP_DATA_ROOT,
) -> tuple[list[str], dict[str, str]]:
    """对样本并集中 accepted 定位记录生成 ref 两路展开条带（观测 3ch / 参考 BGRA）。

    每条记录由定义模块 `preprocess.strip_pair` 一次算出两路条带，再按
    `[obs.BGR, ref.BGR, ref.A]` 切成两路落盘：观测 `<name>.png`、参考 `ref/<name>.png`。
    返回 (产物名, name -> 跳过原因)；跳过原因含 no_locate_record、accept 门原因与
    asset_missing。
    """
    if not samples:
        raise SystemExit("no raw png samples in data/train_raw and data/val_raw")
    accepted, skipped = accepted_records(locate_path)
    accepted = {name: record for name, record in accepted.items() if name in samples}
    skipped = {name: reason for name, reason in skipped.items() if name in samples}
    for name in samples:
        if name not in accepted:
            skipped.setdefault(name, "no_locate_record")
    resolved = _resolve_ref_inputs(accepted, assets_root, skipped)
    resolved, coord_rejected = filter_coord_consistent(resolved, zmdmap_root, assets_root)
    skipped.update(coord_rejected)
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
        return preprocess.observed_roi(load_source_frame(samples[item[0]]))

    for index, ((name, record, asset_path), observed) in enumerate(
        zip(resolved, _parallel_load(resolved, load_observation, workers), strict=True), 1
    ):
        asset_float = prepared_assets.get(asset_path)
        if asset_float is None:
            asset_float = prepared_assets[asset_path] = preprocess.prepare_asset(
                load_reference_image(asset_path)
            )
        observed_strip, reference = preprocess.strip_pair_prepared(
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
        "--zmdmap-data-root",
        type=Path,
        default=ZMDMAP_DATA_ROOT,
        help=(
            "ZmdMap layout 数据目录（ref 坐标一致性过滤；"
            "默认 local/maplocator/data/ZmdMap，MaaEnd assets/data/ZmdMap 的镜像）"
        ),
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
    train_raw_dir: Path = TRAIN_RAW,
    val_raw_dir: Path = VAL_RAW,
    locate_path: Path = LOCATE_PATH,
    processed_dir: Path = PROCESSED_REF,
    train_dir: Path = TRAIN_REF,
    val_dir: Path = VAL_REF,
    force: bool = False,
    workers: int = IO_WORKERS,
    zmdmap_root: Path = ZMDMAP_DATA_ROOT,
) -> None:
    """ref 全管线：两原始目录并集的两路展开 -> 按目录划分 -> train/val 符号链接视图。"""
    train_raw_names, val_raw_names = directory_split(
        png_names(train_raw_dir), png_names(val_raw_dir)
    )
    processed_names, skipped = generate_processed_ref(
        raw_samples(train_raw_dir, val_raw_dir),
        locate_path,
        assets_root,
        processed_dir,
        force,
        workers,
        zmdmap_root,
    )
    processed = set(processed_names)
    train_names, val_names = directory_split(
        sorted(set(train_raw_names) & processed), sorted(set(val_raw_names) & processed)
    )
    if sorted(train_names + val_names) != processed_names:
        raise RuntimeError("ref train/validation split does not exactly cover processed files")
    _print_side_skips("train_raw", train_raw_names, processed, skipped)
    _print_side_skips("val_raw", val_raw_names, processed, skipped)
    link_split(
        train_names,
        val_names,
        train_dir,
        val_dir,
        processed_dir,
        subdirs=(REF_SUBDIR,),
    )


def run_polar(
    train_raw_dir: Path = TRAIN_RAW,
    val_raw_dir: Path = VAL_RAW,
    processed_dir: Path = PROCESSED,
    train_dir: Path = TRAIN,
    val_dir: Path = VAL,
    force: bool = False,
    workers: int = IO_WORKERS,
) -> None:
    """polar 全管线：两原始目录并集展开 -> 按目录划分 -> train/val 符号链接视图。"""
    train_names, val_names = directory_split(png_names(train_raw_dir), png_names(val_raw_dir))
    processed_names = generate_processed(
        raw_samples(train_raw_dir, val_raw_dir), processed_dir, force, workers
    )
    if set(train_names) & set(val_names) or sorted(train_names + val_names) != processed_names:
        raise RuntimeError("train/validation split does not exactly cover processed files")
    link_split(train_names, val_names, train_dir, val_dir, processed_dir)


def main() -> None:
    args = parse_args()
    if args.mode == "ref":
        run_ref(
            args.map_assets_root,
            force=args.force,
            workers=args.workers,
            zmdmap_root=args.zmdmap_data_root,
        )
        return
    run_polar(force=args.force, workers=args.workers)


if __name__ == "__main__":
    main()
