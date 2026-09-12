"""processed* 产物缓存戳（#26）：定义哈希 + 图版本 + 输入指纹。

processed 目录是前处理产物的落盘缓存：整批写完后在目录内落下 `.preprocess.json`，
记录本次产物由哪一版定义生成。命中条件（`cache_hit`）：戳的 schema / 模式 /
`definition_hash`（`endfield/preprocess.py` 的 sha256）/ `graph_version`
（`preprocess.OPSET_VERSION`）与当前一致，且输入指纹（polar 为样本名，ref 另含
zone/x/y/scale）未变；任一不符 = 失效，调用方重生成后覆盖戳。

`git_commit` 只作溯源自证，不参与命中——文档提交不该触发数据重算。戳仅在整批
产物写完后落盘（原子替换）：中途失败留下的半成品目录不会命中。产物文件是否齐全
由调用方按期望名单校验（戳只保证「当时写全了」）。
"""

from __future__ import annotations

import hashlib
import subprocess
from collections.abc import Iterable
from pathlib import Path

from endfield import preprocess
from endfield.data_utils import atomic_json_dump, load_json

STAMP_NAME = ".preprocess.json"
STAMP_SCHEMA_VERSION = 1
REPO_ROOT = Path(__file__).resolve().parents[1]


def git_commit() -> str:
    """当前 HEAD；不在 git 工作区时返回 "unknown"（只作溯源，不参与命中）。"""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def input_fingerprint(entries: Iterable[str]) -> str:
    """输入条目（样本名或规范化的定位字段串）的摘要：排序后逐行取 sha256。"""
    lines = sorted(str(entry) for entry in entries)
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def read_stamp(directory: Path) -> dict | None:
    """读戳；缺失 / 非 JSON / 非对象一律视为未命中（返回 None），不是错误。"""
    path = directory / STAMP_NAME
    if not path.is_file():
        return None
    try:
        return load_json(path)
    except (OSError, ValueError):
        return None


def remove_stamp(directory: Path) -> None:
    """重生成前清掉旧戳：中断留下的半成品目录不得被下次运行命中。"""
    (directory / STAMP_NAME).unlink(missing_ok=True)


def cache_hit(directory: Path, mode: str, entries: Iterable[str]) -> bool:
    """戳与当前定义/图版本/模式/输入一致即命中；产物文件完整性由调用方校验。"""
    stamp = read_stamp(directory)
    if stamp is None:
        return False
    expected = {
        "schema_version": STAMP_SCHEMA_VERSION,
        "mode": mode,
        "definition_hash": preprocess.definition_hash(),
        "graph_version": preprocess.OPSET_VERSION,
    }
    if any(stamp.get(key) != value for key, value in expected.items()):
        return False
    return stamp.get("inputs_sha256") == input_fingerprint(entries)


def write_stamp(directory: Path, mode: str, entries: Iterable[str]) -> dict:
    """整批产物写完后落戳（原子替换）；返回写入内容供日志引用。"""
    entries = list(entries)
    stamp = {
        "schema_version": STAMP_SCHEMA_VERSION,
        "mode": mode,
        "definition_hash": preprocess.definition_hash(),
        "graph_version": preprocess.OPSET_VERSION,
        "git_commit": git_commit(),
        "input_count": len(entries),
        "inputs_sha256": input_fingerprint(entries),
    }
    directory.mkdir(parents=True, exist_ok=True)
    atomic_json_dump(directory / STAMP_NAME, stamp)
    return stamp
