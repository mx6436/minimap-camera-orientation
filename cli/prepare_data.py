"""数据前处理：从 data/train_raw 与 data/val_raw 生成模型输入样本（polar / ref 两种模式）。

编排面：库侧（`endfield.prepare` 与 `placement.ref_inputs`）的 module 在这里拼成命令。
本文件只留参数解析、输入侧适配器的选择与报告打印；管线语义在库里（ADR 0006）。
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

from endfield import dataset, prepare
from endfield.data_utils import png_names
from endfield.run_record import InputMode
from placement import ref_inputs, workspace
from placement.sample import ReferenceSampler


def _skip_reasons(skipped: Mapping[str, str]) -> dict[str, int]:
    return dict(sorted(Counter(skipped.values()).items()))


def _generation_line(report: prepare.PrepareReport, prefix: str) -> str:
    line = f"{prefix} cache={'hit' if report.cache_hit else 'miss'}"
    if not report.cache_hit:
        line += f" definition_hash={report.definition_hash[:12]}"
    return line


def _print_polar(report: prepare.PrepareReport, layout: dataset.DatasetLayout) -> None:
    line = _generation_line(report, f"processed={len(report.names)} format=polar")
    print(f"{line} -> {layout.processed_dir}")


def _print_ref(
    report: prepare.PrepareReport, layout: dataset.DatasetLayout, ref: ref_inputs.RefInputs
) -> None:
    line = _generation_line(
        report,
        f"ref: accepted={len(ref.accepted)} processed={len(report.names)} "
        f"skipped={len(ref.inputs.skipped)} {_skip_reasons(ref.inputs.skipped)}",
    )
    print(f"{line} -> {layout.processed_dir}")


def _print_sides(
    report: prepare.PrepareReport, train_side: Sequence[str], val_side: Sequence[str]
) -> None:
    for label, side in (("train_raw", train_side), ("val_raw", val_side)):
        usable, reasons = report.side_summary(side)
        print(
            f"{label}: samples={len(side)} usable={usable} "
            f"skipped={sum(reasons.values())} {reasons}"
        )


def _print_links(report: prepare.PrepareReport, layout: dataset.DatasetLayout) -> None:
    print(f"train={len(report.train)} -> {layout.train_dir}")
    print(f"val={len(report.val)} -> {layout.val_dir}")


def _run_polar(args: argparse.Namespace) -> None:
    layout = dataset.POLAR_LAYOUT
    train_side = png_names(dataset.TRAIN_RAW_DIR)
    val_side = png_names(dataset.VAL_RAW_DIR)
    report = prepare.prepare(
        InputMode.POLAR,
        prepare.polar_inputs(dataset.raw_samples(dataset.TRAIN_RAW_DIR, dataset.VAL_RAW_DIR)),
        prepare.polar_renderer(),
        layout=layout,
        train_side=train_side,
        val_side=val_side,
        force=args.force,
        workers=args.workers,
    )
    _print_polar(report, layout)
    _print_links(report, layout)


def _run_ref(args: argparse.Namespace) -> None:
    layout = dataset.REF_LAYOUT
    train_side = png_names(dataset.TRAIN_RAW_DIR)
    val_side = png_names(dataset.VAL_RAW_DIR)
    ref = ref_inputs.resolve(
        dataset.raw_samples(dataset.TRAIN_RAW_DIR, dataset.VAL_RAW_DIR),
        locate_path=dataset.LOCATE_PATH,
        assets_root=workspace.assets_root(args.maplocator_root),
    )
    report = prepare.prepare(
        InputMode.REF,
        ref.inputs,
        ref.renderer(ReferenceSampler(ref.assets_root)),
        layout=layout,
        train_side=train_side,
        val_side=val_side,
        force=args.force,
        workers=args.workers,
    )
    _print_ref(report, layout, ref)
    _print_sides(report, train_side, val_side)
    _print_links(report, layout)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--mode",
        choices=tuple(mode.value for mode in InputMode),
        default=InputMode.POLAR.value,
        help=(
            "前处理模式：polar 极坐标展开（默认）；ref 参考输入"
            "（观测/参考各展开后并列，需先跑 locate-dataset）"
        ),
    )
    parser.add_argument(
        "--maplocator-root",
        type=Path,
        default=workspace.WORKSPACE_ROOT,
        help=(
            "本地 MapLocator 工作台根目录（布局、CLI 契约与重建见 "
            f"{workspace.DOC_PATH}）；ref 模式消费其中的 resource/image/MapLocator"
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
        default=prepare.IO_WORKERS,
        help=f"原始图解码的并行线程数（默认 {prepare.IO_WORKERS}=min(16, CPU 数)；1 = 串行）",
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be >= 1")
    return args


def main() -> None:
    args = parse_args()
    if args.mode == "ref":
        _run_ref(args)
        return
    _run_polar(args)


if __name__ == "__main__":
    main()
