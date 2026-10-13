"""运行档案（record.json）的共享契约：schema 版本、核心字段的收发与校验。

本模块持有运行档案的磁盘 schema：强类型核心字段（`target_sigma`、
`trainable_parameters`）+ 原样保留的复现 metadata。训练写侧（`endfield/train/record.py`）
与交付读取侧（`cli/export_onnx.py` 经 `endfield/run_dir.py`）共用这一份读写口径。

模块不依赖 torch，也不持有数据目录布局；ref 单模式下没有输入模式与资产根字段。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from endfield.atomic_io import atomic_json_dump, load_json

RECORD_NAME = "record.json"
# 运行档案的磁盘 schema 版本；格式不变时不随所有权变更提升
SCHEMA_VERSION = 32
# envelope 字段：由本模块持有，读取时从 metadata 中剔除
_ENVELOPE_FIELDS = ("version", "target_sigma", "trainable_parameters")


@dataclass(frozen=True)
class RunRecord:
    """record.json 的 typed view：消费字段 + 原样保留的复现 metadata。"""

    target_sigma: float
    trainable_parameters: int
    metadata: Mapping[str, Any] = field(default_factory=dict)


def require_number(path: Path, raw: Mapping[str, Any], key: str) -> float:
    """JSON 对象里的数值字段；缺失或类型不符（bool 不算数值）即报错。"""
    value = raw.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path}: field {key!r} must be a number, got {value!r}")
    return float(value)


def require_int(path: Path, raw: Mapping[str, Any], key: str) -> int:
    """JSON 对象里的整数字段；缺失或类型不符即报错。"""
    value = raw.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{path}: field {key!r} must be an integer, got {value!r}")
    return value


def read(run_dir: Path) -> RunRecord:
    """读 run 目录的运行档案。"""
    run_dir = Path(run_dir)
    path = run_dir / RECORD_NAME
    try:
        raw = load_json(path)
    except OSError as exc:
        raise ValueError(f"{path}: cannot read run record: {exc}") from exc
    version = raw.get("version")
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"{path}: unsupported run record schema version {version!r}, expected {SCHEMA_VERSION}"
        )

    metadata = {key: value for key, value in raw.items() if key not in _ENVELOPE_FIELDS}
    return RunRecord(
        target_sigma=require_number(path, raw, "target_sigma"),
        trainable_parameters=require_int(path, raw, "trainable_parameters"),
        metadata=metadata,
    )


def write(run_dir: Path, record: RunRecord) -> Path:
    """原子写 record.json：metadata 原样 + envelope 与核心字段；返回档案路径。"""
    run_dir = Path(run_dir)
    payload: dict[str, Any] = dict(record.metadata)
    payload["version"] = SCHEMA_VERSION
    payload["target_sigma"] = record.target_sigma
    payload["trainable_parameters"] = record.trainable_parameters
    path = run_dir / RECORD_NAME
    atomic_json_dump(path, payload)
    return path
