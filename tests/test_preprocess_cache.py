"""processed* 缓存戳（#26）：定义哈希 + 图版本 + 输入指纹决定命中；commit 只作溯源。"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from endfield import preprocess, preprocess_cache

REPO_ROOT = Path(__file__).resolve().parents[1]


def read_stamp_file(directory: Path) -> dict:
    path = directory / preprocess_cache.STAMP_NAME
    return json.loads(path.read_text(encoding="utf-8"))


def test_write_stamp_records_definition_graph_version_commit_and_inputs(tmp_path: Path) -> None:
    stamp = preprocess_cache.write_stamp(tmp_path, "polar", ["b_r1.png", "a_r0.png"])

    assert read_stamp_file(tmp_path) == stamp
    assert stamp["schema_version"] == preprocess_cache.STAMP_SCHEMA_VERSION
    assert stamp["mode"] == "polar"
    assert stamp["definition_hash"] == preprocess.definition_hash()
    assert stamp["graph_version"] == preprocess.OPSET_VERSION
    expected_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()
    assert stamp["git_commit"] == expected_commit
    assert stamp["input_count"] == 2


def test_input_fingerprint_is_order_insensitive_and_content_sensitive() -> None:
    assert preprocess_cache.input_fingerprint(["a", "b"]) == preprocess_cache.input_fingerprint(
        ["b", "a"]
    )
    assert preprocess_cache.input_fingerprint(["a", "b"]) != preprocess_cache.input_fingerprint(
        ["a", "c"]
    )
    assert preprocess_cache.input_fingerprint(["a"]) != preprocess_cache.input_fingerprint([])


def test_cache_hit_requires_same_mode_and_inputs(tmp_path: Path) -> None:
    entries = ["a_r0.png", "b_r1.png"]
    preprocess_cache.write_stamp(tmp_path, "polar", entries)

    assert preprocess_cache.cache_hit(tmp_path, "polar", entries)
    assert not preprocess_cache.cache_hit(tmp_path, "ref", entries)
    assert not preprocess_cache.cache_hit(tmp_path, "polar", ["a_r0.png"])


def test_write_stamp_records_provenance(tmp_path: Path) -> None:
    stamp = preprocess_cache.write_stamp(
        tmp_path, "ref", ["a_r0.png"], provenance={"assets_root": "/tmp/assets"}
    )

    assert stamp["provenance"] == {"assets_root": "/tmp/assets"}
    assert read_stamp_file(tmp_path)["provenance"] == {"assets_root": "/tmp/assets"}
    assert preprocess_cache.write_stamp(tmp_path, "polar", ["a_r0.png"])["provenance"] == {}


def test_cache_hit_requires_matching_provenance_when_given(tmp_path: Path) -> None:
    entries = ["a_r0.png"]
    preprocess_cache.write_stamp(
        tmp_path, "ref", entries, provenance={"assets_root": "/tmp/assets"}
    )

    assert preprocess_cache.cache_hit(
        tmp_path, "ref", entries, provenance={"assets_root": "/tmp/assets"}
    )
    assert not preprocess_cache.cache_hit(
        tmp_path, "ref", entries, provenance={"assets_root": "/tmp/other"}
    )
    assert preprocess_cache.cache_hit(tmp_path, "ref", entries)
    assert not preprocess_cache.cache_hit(tmp_path, "ref", entries, provenance={"other": "x"})


def test_cache_hit_rejects_stale_definition_hash(tmp_path: Path) -> None:
    entries = ["a_r0.png"]
    preprocess_cache.write_stamp(tmp_path, "polar", entries)
    stamp = read_stamp_file(tmp_path)
    stamp["definition_hash"] = "0" * 64
    (tmp_path / preprocess_cache.STAMP_NAME).write_text(json.dumps(stamp), encoding="utf-8")

    assert not preprocess_cache.cache_hit(tmp_path, "polar", entries)


def test_cache_hit_rejects_stale_graph_version(tmp_path: Path) -> None:
    entries = ["a_r0.png"]
    preprocess_cache.write_stamp(tmp_path, "polar", entries)
    stamp = read_stamp_file(tmp_path)
    stamp["graph_version"] = preprocess.OPSET_VERSION + 1
    (tmp_path / preprocess_cache.STAMP_NAME).write_text(json.dumps(stamp), encoding="utf-8")

    assert not preprocess_cache.cache_hit(tmp_path, "polar", entries)


def test_cache_hit_ignores_git_commit_change(tmp_path: Path) -> None:
    """commit 是溯源自证：文档提交不该触发数据重算。"""
    entries = ["a_r0.png"]
    preprocess_cache.write_stamp(tmp_path, "polar", entries)
    stamp = read_stamp_file(tmp_path)
    stamp["git_commit"] = "0" * 40
    (tmp_path / preprocess_cache.STAMP_NAME).write_text(json.dumps(stamp), encoding="utf-8")

    assert preprocess_cache.cache_hit(tmp_path, "polar", entries)


def test_cache_hit_treats_missing_or_corrupt_stamp_as_miss(tmp_path: Path) -> None:
    assert not preprocess_cache.cache_hit(tmp_path, "polar", ["a_r0.png"])

    (tmp_path / preprocess_cache.STAMP_NAME).write_text("not json", encoding="utf-8")
    assert not preprocess_cache.cache_hit(tmp_path, "polar", ["a_r0.png"])
