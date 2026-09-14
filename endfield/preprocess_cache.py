"""processed* 产物缓存戳：定义哈希 + 图版本 + 输入指纹。"""

from __future__ import annotations

import hashlib
import subprocess
from collections.abc import Iterable, Mapping
from pathlib import Path

from endfield import preprocess
from endfield.atomic_io import atomic_json_dump, load_json

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
    """重生成前清掉旧戳。"""
    (directory / STAMP_NAME).unlink(missing_ok=True)


def cache_hit(
    directory: Path,
    mode: str,
    entries: Iterable[str],
    provenance: Mapping[str, str] | None = None,
) -> bool:
    """戳与当前定义/图版本/模式/输入一致即命中；产物文件完整性由调用方校验。

    `provenance` 非 None 时还要求戳内记录逐键一致（如 ref 数据的资产根）。
    """
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
    if stamp.get("inputs_sha256") != input_fingerprint(entries):
        return False
    if provenance is None:
        return True
    return stamp.get("provenance") == dict(provenance)


def write_stamp(
    directory: Path,
    mode: str,
    entries: Iterable[str],
    provenance: Mapping[str, str] | None = None,
) -> dict:
    """整批产物写完后落戳（原子替换）；返回写入内容供日志引用。

    `provenance` 记录本批产物内容依赖的工作台事实（如资产根），随戳一起固化。
    """
    entries = list(entries)
    stamp = {
        "schema_version": STAMP_SCHEMA_VERSION,
        "mode": mode,
        "definition_hash": preprocess.definition_hash(),
        "graph_version": preprocess.OPSET_VERSION,
        "git_commit": git_commit(),
        "input_count": len(entries),
        "inputs_sha256": input_fingerprint(entries),
        "provenance": dict(provenance or {}),
    }
    directory.mkdir(parents=True, exist_ok=True)
    atomic_json_dump(directory / STAMP_NAME, stamp)
    return stamp
