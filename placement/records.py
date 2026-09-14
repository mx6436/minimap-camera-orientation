"""定位产物（`locate.jsonl`）的读写与集合操作：一行一条定位记录。"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from endfield.atomic_io import atomic_path

Record = dict[str, Any]

STATUS_SUCCESS = 0
STATUS_READ_FAILED = -1
STATUS_ROI_FAILED = -2


def parse_record(line: str) -> Record:
    """解析 CLI 的一行 JSONL；缺 name/status 视为产物损坏，直接报错。"""
    try:
        record = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSONL line: {line[:160]!r}") from exc
    if not isinstance(record, dict) or "name" not in record or "status" not in record:
        raise ValueError(f"record missing name/status: {line[:160]!r}")
    return record


def load_records(path: Path) -> dict[str, Record]:
    """读取现有产物，按 name 索引；文件不存在视为空。"""
    if not path.exists():
        return {}
    records: dict[str, Record] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = parse_record(line)
            records[record["name"]] = record
    return records


def write_jsonl(path: Path, records: Iterable[Record]) -> None:
    """原子写入产物：每行一条，按 name 排序，保证重复运行结果稳定。"""
    ordered = sorted(records, key=lambda r: r["name"])
    with atomic_path(path) as temp:
        with temp.open("w", encoding="utf-8") as handle:
            for record in ordered:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def pending_names(
    names: Iterable[str],
    done: Mapping[str, Record],
    retry_failed: bool = True,
) -> list[str]:
    """待定位清单：已成功的跳过；失败项默认重跑（定位是确定性的，重跑幂等）。"""
    pending = []
    for name in names:
        record = done.get(name)
        if record is None or (retry_failed and record.get("status") != STATUS_SUCCESS):
            pending.append(name)
    return pending


def merge_records(existing: Iterable[Record], new: Iterable[Record]) -> list[Record]:
    """按 name 合并新旧记录，新的覆盖旧的。"""
    by_name: dict[str, Record] = {record["name"]: record for record in existing}
    by_name.update({record["name"]: record for record in new})
    return [by_name[name] for name in sorted(by_name)]


def split_shards(names: Sequence[str], jobs: int) -> list[list[str]]:
    """把清单切成 jobs 个连续分片（保持顺序，大小尽量均衡）。"""
    if jobs < 1:
        raise ValueError(f"jobs must be >= 1, got {jobs}")
    names = list(names)
    base, extra = divmod(len(names), jobs)
    shards: list[list[str]] = []
    start = 0
    for index in range(jobs):
        size = base + (1 if index < extra else 0)
        shards.append(names[start : start + size])
        start += size
    return shards
