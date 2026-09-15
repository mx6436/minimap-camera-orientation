"""运行档案（record.json）的共享契约：输入模式词汇、消费字段读取与通道核对。

本模块是训练、实机推理与工件导出共用的运行契约持有者：`InputMode` 是模式名的唯一
来源，`input_channels` 是模式到输入通道数的唯一定义，`read` / `write` 收发
record.json，`validate_channels` 在 checkpoint 与档案之间核对通道数。

模块不依赖 torch 或训练包，也不持有交付 bundle 的词汇（图文件名、manifest、role
映射）与数据目录布局；实机与交付不得再出现第二份模式表。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from endfield.atomic_io import atomic_json_dump, load_json

REPO_ROOT = Path(__file__).resolve().parents[1]
RECORD_NAME = "record.json"
# 运行档案的磁盘 schema 版本；格式不变时不随所有权变更提升
SCHEMA_VERSION = 31
# ref 模式携带参考底图资产根的字段名
ASSETS_ROOT_FIELD = "ref_reference_assets_root"
# envelope 字段：由本模块持有，读取时从 metadata 中剔除
_ENVELOPE_FIELDS = (
    "version",
    "input_mode",
    "target_sigma",
    "trainable_parameters",
    ASSETS_ROOT_FIELD,
)


class InputMode(StrEnum):
    """模型输入的表示名；贯穿训练、数据生成、实机推理与交付。"""

    POLAR = "polar"
    REF = "ref"

    @classmethod
    def parse(cls, value: object) -> InputMode:
        """字符串或枚举 -> InputMode；未知取值给出含合法模式的 ValueError。"""
        try:
            return cls(value)
        except ValueError:
            expected = ", ".join(mode.value for mode in cls)
            raise ValueError(f"unknown input_mode: {value!r}; expected one of {expected}") from None


_INPUT_CHANNELS: dict[InputMode, int] = {InputMode.POLAR: 3, InputMode.REF: 7}


def input_channels(input_mode: InputMode | str) -> int:
    """输入模式 -> AzimuthNet 首层通道数（模式词汇的唯一定义）。"""
    return _INPUT_CHANNELS[InputMode.parse(input_mode)]


@dataclass(frozen=True)
class RunRecord:
    """record.json 的 typed view：消费字段 + 原样保留的复现 metadata。"""

    input_mode: InputMode
    assets_root: Path | None
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
    """读 run 目录的运行档案。

    缺 `input_mode` 的旧档案按 polar 处理（polar 是当时唯一模式）；ref 档案必须带
    非空的资产根字段，相对路径按仓库根解析。
    """
    run_dir = Path(run_dir)
    path = run_dir / RECORD_NAME
    try:
        raw = load_json(path)
    except OSError as exc:
        raise ValueError(f"{path}: cannot read run record: {exc}") from exc

    if "input_mode" in raw:
        try:
            mode = InputMode.parse(raw["input_mode"])
        except ValueError as exc:
            raise ValueError(f"{path}: {exc}") from None
    else:
        mode = InputMode.POLAR

    assets_root: Path | None = None
    if mode is InputMode.REF:
        value = raw.get(ASSETS_ROOT_FIELD)
        if not isinstance(value, str) or not value:
            raise ValueError(f"{path}: ref run record lacks {ASSETS_ROOT_FIELD}")
        root = Path(value)
        assets_root = root if root.is_absolute() else REPO_ROOT / root

    metadata = {key: value for key, value in raw.items() if key not in _ENVELOPE_FIELDS}
    return RunRecord(
        input_mode=mode,
        assets_root=assets_root,
        target_sigma=require_number(path, raw, "target_sigma"),
        trainable_parameters=require_int(path, raw, "trainable_parameters"),
        metadata=metadata,
    )


def write(run_dir: Path, record: RunRecord) -> Path:
    """原子写 record.json：metadata 原样 + envelope 与核心字段；返回档案路径。

    资产根与输入模式互为条件：ref 必须带资产根，polar 不得带。
    """
    run_dir = Path(run_dir)
    if record.input_mode is InputMode.REF and record.assets_root is None:
        raise ValueError("ref run record requires a reference assets root")
    if record.input_mode is InputMode.POLAR and record.assets_root is not None:
        raise ValueError("polar run record must not carry a reference assets root")

    payload: dict[str, Any] = dict(record.metadata)
    payload["version"] = SCHEMA_VERSION
    payload["input_mode"] = record.input_mode.value
    payload["target_sigma"] = record.target_sigma
    payload["trainable_parameters"] = record.trainable_parameters
    if record.assets_root is not None:
        payload[ASSETS_ROOT_FIELD] = str(record.assets_root)
    path = run_dir / RECORD_NAME
    atomic_json_dump(path, payload)
    return path


def validate_channels(record: RunRecord, in_channels: int) -> None:
    """checkpoint 的输入通道数必须与运行档案的输入模式一致。"""
    expected = input_channels(record.input_mode)
    if in_channels != expected:
        raise ValueError(
            f"checkpoint has {in_channels} input channels but run record input_mode "
            f"{record.input_mode.value!r} expects {expected}"
        )
