"""run 产物契约：run 目录里的文件与训练汇总（`summary.json`）的 schema。

持有 run 目录的形状（`best.pt` / `record.json` / `summary.json` / `history.json` 的路径与占用
标记）与训练汇总的字段 schema，供训练写侧、实机取 checkpoint、交付导出与 conformance 共用。
`record.json` 的 schema 仍由 `endfield/run_record.py` 持有，本模块只是它的调用方。

本模块保持 torch-free：`cli/verify_artifact.py` 的校验路径经 `endfield/bundle.py` 到达这里。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from endfield.atomic_io import atomic_json_dump, load_json
from endfield.run_record import (
    RECORD_NAME,
    RunRecord,
    require_int,
    require_number,
)
from endfield.run_record import (
    read as read_record,
)
from endfield.train.metrics import Metrics

CHECKPOINT_NAME = "best.pt"
SUMMARY_NAME = "summary.json"
HISTORY_NAME = "history.json"

# 写侧会写的文件（不含 loss_curve.png）：目录里出现任一即视为已被一次 run 占用
OCCUPANCY_MARKERS: tuple[str, ...] = (
    CHECKPOINT_NAME,
    RECORD_NAME,
    SUMMARY_NAME,
    HISTORY_NAME,
)

# 交付侧消费的 summary 指标：缺失即报错（其余诊断指标缺失容忍）
_CONTRACT_METRICS = ("expected_abs_error", "rms_error")


@dataclass(frozen=True)
class TrainingSummary:
    """`summary.json` 的 typed view：交付契约字段 + 训练侧诊断。

    `metrics` 里的 rms 误差与期望绝对误差是交付 manifest 与交付图 metadata 的输入；其余字段是
    诊断信息，随 `endfield/train/metrics.py` 的口径演化。
    """

    epoch: int
    val_count: int
    metrics: Metrics
    val_loss: float = 0.0
    best_val_rms_error: float = 0.0
    extra: Mapping[str, Any] = field(default_factory=dict)


def checkpoint_path(run_dir: Path) -> Path:
    """交付 checkpoint 的路径。"""
    return Path(run_dir) / CHECKPOINT_NAME


def record_path(run_dir: Path) -> Path:
    """运行档案的路径。"""
    return Path(run_dir) / RECORD_NAME


def summary_path(run_dir: Path) -> Path:
    """训练汇总的路径。"""
    return Path(run_dir) / SUMMARY_NAME


def history_path(run_dir: Path) -> Path:
    """逐 epoch 训练历史的路径。"""
    return Path(run_dir) / HISTORY_NAME


def occupied(run_dir: Path) -> list[str]:
    """目录里已存在的占用标记：非空即拒绝在同一个 run 目录上重跑。"""
    return [name for name in OCCUPANCY_MARKERS if (Path(run_dir) / name).exists()]


def load_record(run_dir: Path) -> RunRecord:
    """读运行档案；schema 与读取口径由 `endfield/run_record.py` 持有。"""
    return read_record(Path(run_dir))


def load_summary(run_dir: Path) -> TrainingSummary:
    """读训练汇总：交付契约指标缺失或类型不符即报错，诊断指标缺失容忍，未知键保留。"""
    path = summary_path(run_dir)
    try:
        payload = load_json(path)
    except OSError as exc:
        raise ValueError(f"{path}: cannot read training summary: {exc}") from exc
    for name in _CONTRACT_METRICS:
        require_number(path, payload, Metrics.payload_key(name))
    metrics = Metrics.from_payload(payload)
    known = {
        "epoch",
        "val_count",
        "val_loss",
        "best_val_rms_error",
        *metrics.to_payload(),
    }
    return TrainingSummary(
        epoch=require_int(path, payload, "epoch"),
        val_count=require_int(path, payload, "val_count"),
        metrics=metrics,
        val_loss=float(payload.get("val_loss", 0.0)),
        best_val_rms_error=float(payload.get("best_val_rms_error", 0.0)),
        extra={key: value for key, value in payload.items() if key not in known},
    )


def load_run(run_dir: Path) -> tuple[RunRecord, TrainingSummary]:
    """run 目录 -> 运行档案与训练汇总；交付侧与 conformance 的唯一读取入口。"""
    return load_record(run_dir), load_summary(run_dir)


def write_summary(run_dir: Path, summary: TrainingSummary) -> Path:
    """原子写训练汇总：契约字段 + 诊断指标；未知键（extra）原样保留。"""
    payload: dict[str, Any] = {
        "epoch": summary.epoch,
        "val_count": summary.val_count,
        "best_val_rms_error": summary.best_val_rms_error,
        "val_loss": summary.val_loss,
    }
    payload.update(summary.metrics.to_payload())
    payload.update(summary.extra)
    path = summary_path(run_dir)
    atomic_json_dump(path, payload)
    return path
