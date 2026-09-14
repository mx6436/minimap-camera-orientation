"""定位产物（`locate.jsonl`）的读写与集合操作测试：解析、续跑、合并、分片。"""

from __future__ import annotations

import json
from pathlib import Path

from placement.records import (
    load_records,
    merge_records,
    parse_record,
    pending_names,
    split_shards,
    write_jsonl,
)
from tests._locate_records import OK, failed


def test_parse_record_requires_name_and_status() -> None:
    record = parse_record(json.dumps(OK))
    assert record["name"] == OK["name"]
    assert record["status"] == 0


def test_write_load_merge_is_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "locate.jsonl"
    b = {**OK, "name": "map01_lv001_x157.2_y486.6_r342.png"}
    write_jsonl(path, [OK, failed(b["name"], 1, "Global search failed.")])
    done = load_records(path)
    assert sorted(done) == sorted([OK["name"], b["name"]])

    merged = merge_records(
        done.values(),
        [
            failed(b["name"], 0, "Global Search Success"),
            {**OK, "name": "ValleyIV_Base_x1_y2_r3.png"},
        ],
    )
    assert [r["name"] for r in merged] == sorted(
        [OK["name"], b["name"], "ValleyIV_Base_x1_y2_r3.png"]
    )
    assert next(r for r in merged if r["name"] == b["name"])["status"] == 0
    write_jsonl(path, merged)
    write_jsonl(path, load_records(path).values())
    assert [json.loads(line) for line in path.read_text().splitlines()] == merged


def test_pending_names_skips_success_but_retries_failures() -> None:
    done = {OK["name"]: OK, "b.png": failed("b.png", 1, "Global search failed.")}
    names = [OK["name"], "b.png", "c.png"]
    assert pending_names(names, done) == ["b.png", "c.png"]
    assert pending_names(names, done, retry_failed=False) == ["c.png"]


def test_split_shards_keeps_order_and_balances() -> None:
    names = [f"{i}.png" for i in range(7)]
    shards = split_shards(names, 3)
    assert [len(s) for s in shards] == [3, 2, 2]
    assert [n for s in shards for n in s] == names
