"""数据准备：原始样本 -> 条带产物 -> 训练/验证划分视图（CONTEXT.md「数据准备」）。

实现吞掉两件事：按输入模式渲染条带产物（原始截图解码按 CPU 数并行，定义模块调用始终
在主线程串行，产物与 `workers=1` 逐字节一致），以及把产物按两侧原始名单切成训练/验证
视图。输入侧的模式差异由调用方注入两个适配器——输入侧值（`PrepareInputs`）与渲染回调
（`RenderFn`）——因此本模块不 import `placement`，`placement → endfield` 的单向依赖
得以保持（ADR 0006）。
"""

from __future__ import annotations

import os
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from endfield import preprocess, preprocess_cache
from endfield.data_utils import png_names
from endfield.dataset import DatasetLayout, directory_split
from endfield.polar import imread_png, load_source_frame
from endfield.preprocess import IMG_H, IMG_W
from endfield.run_record import InputMode

# 原始截图解码（cv2.imread 释放 GIL）是管线大头：每张 ~10ms，而定义模块前处理 ~1ms。
# 解码/ROI 提取按 CPU 数并行；定义模块调用始终在主线程串行，产物与 workers=1 逐字节一致。
IO_WORKERS = min(16, os.cpu_count() or 1)
IO_CHUNK = 256


@dataclass(frozen=True)
class StripPair:
    """条带对：观测条带 3 通道，`ref` 模式另有参考条带 4 通道。"""

    observed: np.ndarray
    reference: np.ndarray | None = None


# 渲染回调：样本名 + 观测 ROI -> 条带对。闭包持有该模式渲染所需的其余上下文
# （如 ref 的底图定位与资产采样器），因此本模块不必知道模式差异从何而来。
RenderFn = Callable[[str, np.ndarray], StripPair]


@dataclass(frozen=True)
class PrepareInputs:
    """数据准备的输入侧事实：要生成的样本、落戳指纹条目、未入选样本与产物溯源。"""

    sources: Sequence[tuple[str, Path]]
    fingerprint: Sequence[str]
    skipped: Mapping[str, str] = field(default_factory=dict)
    provenance: Mapping[str, str] | None = None


@dataclass(frozen=True)
class PrepareReport:
    """一批数据准备的事实：产物名单、未入选样本、缓存命中与本批划分。"""

    mode: InputMode
    names: Sequence[str]
    skipped: Mapping[str, str]
    cache_hit: bool
    definition_hash: str
    train: Sequence[str]
    val: Sequence[str]

    def side_summary(self, raw_names: Iterable[str]) -> tuple[int, dict[str, int]]:
        """一侧原始名单 ->（可用数, 跳过原因分布）；跳过样本已从该侧视图剔除。"""
        names = list(raw_names)
        processed = set(self.names)
        usable = sum(1 for name in names if name in processed)
        reasons = Counter(self.skipped[name] for name in names if name not in processed)
        return usable, dict(sorted(reasons.items()))


@dataclass(frozen=True)
class _Generation:
    names: tuple[str, ...]
    cache_hit: bool
    stamp: Mapping[str, Any]


def polar_inputs(samples: Mapping[str, Path]) -> PrepareInputs:
    """polar 输入侧适配器：两侧原始目录并集 -> 输入侧值（指纹即样本名并集）。"""
    names = sorted(samples)
    if not names:
        raise SystemExit("no raw png samples in data/train_raw and data/val_raw")
    return PrepareInputs(sources=[(name, samples[name]) for name in names], fingerprint=names)


def polar_renderer() -> RenderFn:
    """polar 渲染适配器：观测 ROI -> 观测条带（无参考）。"""

    def render(_name: str, observed_roi: np.ndarray) -> StripPair:
        return StripPair(preprocess.observed_strip(observed_roi))

    return render


def prepare(
    mode: InputMode,
    inputs: PrepareInputs,
    render: RenderFn,
    *,
    layout: DatasetLayout,
    train_side: Sequence[str],
    val_side: Sequence[str],
    force: bool = False,
    workers: int = IO_WORKERS,
) -> PrepareReport:
    """生成条带产物、切出训练/验证视图，返回本批事实。

    `train_side` / `val_side` 是两侧原始名单：与产物求交后的结果即划分视图，被剔除的
    样本从各自一侧消失。产物与缓存戳命中时不重算（`force` 除外）。
    """
    raw_train, raw_val = directory_split(train_side, val_side)
    generation = _generate(mode, inputs, render, layout, force=force, workers=workers)
    processed = set(generation.names)
    train, val = directory_split(
        sorted(set(raw_train) & processed), sorted(set(raw_val) & processed)
    )
    if set(train) & set(val) or sorted(train + val) != list(generation.names):
        raise RuntimeError("train/validation split does not exactly cover processed files")
    _link_split(train, val, layout)
    return PrepareReport(
        mode=InputMode.parse(mode),
        names=generation.names,
        skipped=dict(inputs.skipped),
        cache_hit=generation.cache_hit,
        definition_hash=str(generation.stamp.get("definition_hash", "")),
        train=tuple(train),
        val=tuple(val),
    )


def _generate(
    mode: InputMode,
    inputs: PrepareInputs,
    render: RenderFn,
    layout: DatasetLayout,
    *,
    force: bool,
    workers: int,
) -> _Generation:
    """按输入侧值渲染并落盘两路条带与缓存戳；命中即原样返回，不重算。"""
    mode = InputMode.parse(mode)
    names = [name for name, _ in inputs.sources]
    out_dir = layout.processed_dir
    if (
        not force
        and _outputs_match(layout, names)
        and preprocess_cache.cache_hit(out_dir, mode.value, inputs.fingerprint, inputs.provenance)
    ):
        stamp = preprocess_cache.read_stamp(out_dir) or {}
        return _Generation(names=tuple(names), cache_hit=True, stamp=stamp)

    preprocess_cache.remove_stamp(out_dir)
    for subdir in ("", *layout.subdirs):
        _clear_pngs(out_dir / subdir)
    reference_dir = out_dir / layout.subdirs[0] if layout.subdirs else None

    total = len(names)
    for index, ((name, _), pair) in enumerate(
        zip(inputs.sources, _rendered(inputs.sources, render, workers), strict=True), 1
    ):
        if (pair.reference is not None) is not (mode is InputMode.REF):
            expected = "给出" if mode is InputMode.REF else "不给出"
            raise ValueError(f"{name}: {mode.value} 模式的渲染回调必须{expected}参考条带")
        if not cv2.imwrite(str(out_dir / name), pair.observed):
            raise RuntimeError(f"failed to write {out_dir / name}")
        if pair.reference is not None and reference_dir is not None:
            if not cv2.imwrite(str(reference_dir / name), pair.reference):
                raise RuntimeError(f"failed to write {reference_dir / name}")
        if index % 250 == 0 or index == total:
            print(f"[{index}/{total}] {name}", flush=True)

    _validate_outputs(mode, layout, names)
    stamp = preprocess_cache.write_stamp(out_dir, mode.value, inputs.fingerprint, inputs.provenance)
    return _Generation(names=tuple(names), cache_hit=False, stamp=stamp)


def _rendered(
    sources: Sequence[tuple[str, Path]], render: RenderFn, workers: int
) -> Iterator[StripPair]:
    """按序产出条带对：解码与 ROI 裁剪并行，渲染（定义模块调用）留在主线程。"""

    def load_roi(item: tuple[str, Path]) -> np.ndarray:
        return preprocess.observed_roi(load_source_frame(item[1]))

    for (name, _), roi in zip(sources, _parallel_load(sources, load_roi, workers), strict=True):
        yield render(name, roi)


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


def _outputs_match(layout: DatasetLayout, names: Sequence[str]) -> bool:
    """产物文件的名单与待生成名单一致（含并行子树）；缺文件即未命中。"""
    want = list(names)
    return all(png_names(layout.processed_dir / subdir) == want for subdir in ("", *layout.subdirs))


def _validate_outputs(mode: InputMode, layout: DatasetLayout, names: Sequence[str]) -> None:
    """落盘后自检：名单一致 + 观测/参考两路的形状与模式相符。"""
    out_dir = layout.processed_dir
    reference_dir = out_dir / layout.subdirs[0] if layout.subdirs else None
    observation = "ref observation" if mode is InputMode.REF else "polar"
    if not _outputs_match(layout, names):
        if mode is InputMode.REF:
            raise RuntimeError("processed ref PNG names do not exactly match written samples")
        raise RuntimeError("processed PNG names do not exactly match raw PNG names")
    for name in names:
        if imread_png(out_dir / name).shape != (IMG_H, IMG_W, 3):
            raise RuntimeError(f"invalid processed {observation} image: {name}")
        if reference_dir is None:
            continue
        if imread_png(reference_dir / name).shape != (IMG_H, IMG_W, 4):
            raise RuntimeError(f"invalid processed ref reference image: {name}")


def _link_split(
    train_names: Sequence[str], val_names: Sequence[str], layout: DatasetLayout
) -> None:
    """把两侧名单链到划分视图；悬空链接与名单不符硬报错。"""
    for subdir in ("", *layout.subdirs):
        for directory, names in ((layout.train_dir, train_names), (layout.val_dir, val_names)):
            target_dir = directory / subdir
            _clear_pngs(target_dir)
            # 从 train/val 的并行子树回到 processed 的同一子树：顶层一层，子目录两层
            prefix = Path("..") if not subdir else Path("..", "..")
            for name in names:
                (target_dir / name).symlink_to(prefix / layout.processed_dir.name / subdir / name)
                if not (target_dir / name).is_file():
                    raise RuntimeError(f"split link does not resolve: {target_dir / name}")
            if png_names(target_dir) != list(names):
                raise RuntimeError(f"linked {target_dir} files do not match the split")


def _clear_pngs(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for path in directory.glob("*.png"):
        path.unlink()
