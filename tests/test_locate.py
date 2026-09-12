"""批量定位产物逻辑与 CLI 进程封装的测试：解析、续跑、合并、失败分类、汇总、分片、
zone 资产映射、流式定位。

这些是 `locate_dataset.py`（调用 local/maplocator/bin/map-locate 的进程编排在脚本里）
消费/产出的纯逻辑；CLI 本身另有集成验证（见 data/locator 的核对产物）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from endfield.locate import (
    LocalizerStream,
    accept,
    classify,
    load_records,
    merge_records,
    parse_record,
    pending_names,
    run_cli,
    split_shards,
    summarize,
    write_jsonl,
    zone_asset_path,
)
from endfield.polar import load_source_bgr

OK = {
    "name": "Wuling_Base_x1000.0_y1403.0_r346.1.png",
    "status": 0,
    "message": "Global Search Success",
    "zone": "Wuling_Base",
    "x": 1.0,
    "y": 2.0,
    "rot": 3.0,
    "locConf": 0.9,
    "isHeld": False,
    "latencyMs": 10,
    "attempts": 1,
    "elapsedMs": 20,
}


def failed(name: str, status: int, message: str) -> dict:
    return {
        **OK,
        "name": name,
        "status": status,
        "message": message,
        "zone": "",
        "locConf": 0.0,
        "attempts": 3,
    }


def test_parse_record_requires_name_and_status() -> None:
    record = parse_record(json.dumps(OK))
    assert record["name"] == OK["name"]
    assert record["status"] == 0


def test_classify_maps_status_and_message() -> None:
    assert classify(OK) == "ok"
    assert classify(failed("a", -1, "imread failed")) == "read_failed"
    assert classify(failed("a", -2, "minimap ROI out of bounds")) == "roi_failed"
    assert classify(failed("a", 1, "Global search failed.")) == "global_search_failed"
    assert classify(failed("a", 1, "Cold-start collecting.")) == "cold_start_unsettled"
    assert classify(failed("a", 1, "Far-jump rejected.")) == "far_jump_rejected"
    assert classify(failed("a", 2, "Screen Blocked")) == "screen_blocked"
    assert classify(failed("a", 4, "YOLO failed")) == "yolo_failed"
    assert classify(failed("a", 5, "NotInitialized")) == "not_initialized"
    assert classify(failed("a", 9, "?")) == "unknown_9"


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


def test_accept_gates_held_and_low_conf() -> None:
    assert accept(OK) == (True, "ok")
    assert accept({**OK, "isHeld": True, "locConf": 0.2}) == (False, "held")
    assert accept({**OK, "locConf": 0.5}) == (False, "below_loc_threshold")
    assert accept(failed("x.png", 1, "Global search failed.")) == (False, "global_search_failed")


def test_summarize_counts_and_distributions() -> None:
    records = [
        OK,
        {**OK, "name": "Wuling_Base_b.png", "attempts": 3, "locConf": 0.6},
        {**OK, "name": "Wuling_Base_c.png", "isHeld": True, "locConf": 0.4},
        failed("map01_lv001_x157.2_y486.6_r342.png", 1, "Cold-start collecting."),
    ]
    summary = summarize(records)
    assert summary["total"] == 4
    assert summary["ok"] == 3
    assert summary["failed"] == 1
    assert summary["accepted"] == 2
    assert summary["excluded_by_reason"]["held"] == 1
    assert summary["excluded_by_reason"]["cold_start_unsettled"] == 1
    assert summary["by_category"]["cold_start_unsettled"] == 1
    assert summary["attempts"] == {1: 2, 3: 2}
    assert summary["locconf"]["below_high_conf"] == 2
    assert summary["locconf"]["below_loc_threshold"] == 1
    assert summary["held"] == 1
    assert summary["families"]["Wuling"] == {"total": 3, "ok": 3, "accepted": 2}
    assert summary["families"]["map01"] == {"total": 1, "ok": 0, "accepted": 0}


def test_zone_asset_path_resolves_naming_schemes(tmp_path: Path) -> None:
    (tmp_path / "ValleyIV").mkdir()
    (tmp_path / "ValleyIV" / "Base.png").touch()
    (tmp_path / "ValleyIV" / "Lv006Tier109.png").touch()
    (tmp_path / "OMVBase").mkdir()
    (tmp_path / "OMVBase" / "OMVBase01.png").touch()

    assert zone_asset_path("ValleyIV_Base", tmp_path) == tmp_path / "ValleyIV" / "Base.png"
    assert (
        zone_asset_path("ValleyIV_L6_109", tmp_path) == tmp_path / "ValleyIV" / "Lv006Tier109.png"
    )
    assert zone_asset_path("OMVBase01", tmp_path) == tmp_path / "OMVBase" / "OMVBase01.png"
    assert zone_asset_path("Nope_Base", tmp_path) is None
    assert zone_asset_path("Nope01", tmp_path) is None


def test_run_cli_streams_paths_and_writes_jsonl(tmp_path: Path) -> None:
    fake = tmp_path / "fake_cli.py"
    fake.write_text(
        "import json, sys\n"
        "for line in sys.stdin:\n"
        "    name = line.strip()\n"
        "    if name:\n"
        "        print(json.dumps({'name': name, 'status': 0, 'attempts': 1}))\n"
    )
    out = tmp_path / "part.jsonl"
    names = ["x.png", "y.png"]
    assert run_cli([sys.executable, str(fake)], tmp_path, names, out) == 2
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert [r["name"] for r in rows] == names


FAKE_STREAM_CLI = (
    "import json, sys\n"
    "from pathlib import Path\n"
    "args = sys.argv[1:]\n"
    "assert '--stream' in args and '--resource-dir' in args, args\n"
    "for line in sys.stdin:\n"
    "    path = line.strip()\n"
    "    if not path:\n"
    "        continue\n"
    "    p = Path(path)\n"
    "    assert p.is_file() and p.stat().st_size > 0, path\n"
    "    print(json.dumps({'name': p.name, 'status': 0, 'message': 'Tracking Success',"
    " 'zone': 'Fake_Base', 'x': 1.0, 'y': 2.0, 'rot': 0.0, 'locConf': 0.9,"
    " 'isHeld': False, 'latencyMs': 1, 'attempts': 1, 'elapsedMs': 2}), flush=True)\n"
)


def test_localizer_stream_roundtrips_one_record_per_frame(tmp_path: Path) -> None:
    fake = tmp_path / "fake_stream_cli.py"
    fake.write_text(FAKE_STREAM_CLI, encoding="utf-8")
    stream = LocalizerStream([sys.executable, str(fake)], tmp_path / "resource", tmp_path / "work")
    stream.start()
    try:
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        first = stream.locate(frame)
        second = stream.locate(frame)
    finally:
        stream.close()
    assert first == second
    assert first["name"] == "frame.bmp"
    assert first["status"] == 0


def test_localizer_stream_reports_cli_death(tmp_path: Path) -> None:
    fake = tmp_path / "dead_cli.py"
    fake.write_text("import sys; sys.exit(3)\n", encoding="utf-8")
    stream = LocalizerStream([sys.executable, str(fake)], tmp_path, tmp_path / "work")
    stream.start()
    try:
        with pytest.raises(RuntimeError, match="map-locate"):
            stream.locate(np.zeros((720, 1280, 3), dtype=np.uint8))
    finally:
        stream.close()


REPO_ROOT = Path(__file__).resolve().parents[1]
STREAM_CLI_PATH = REPO_ROOT / "local" / "maplocator" / "bin" / "map-locate"
STREAM_RESOURCE_DIR = REPO_ROOT / "local" / "maplocator" / "resource"
STREAM_RAW_DIR = REPO_ROOT / "data" / "raw"
STREAM_LOCATE_PATH = REPO_ROOT / "data" / "locator" / "locate.jsonl"


@pytest.mark.skipif(
    not STREAM_CLI_PATH.exists()
    or not STREAM_RESOURCE_DIR.is_dir()
    or not STREAM_LOCATE_PATH.exists(),
    reason="local map-locate binary / resource / locate.jsonl not available",
)
def test_localizer_stream_keeps_tracking_without_reset(tmp_path: Path) -> None:
    """连续喂同一帧：若逐帧 reset，每帧都停在全局搜索/冷启动；stream 模式跨帧保留
    追踪状态，末帧应为 Tracking Success 且每帧只调一次 locate（attempts=1）。"""
    from concurrent.futures import ThreadPoolExecutor
    from concurrent.futures import TimeoutError as FutureTimeout

    records = load_records(STREAM_LOCATE_PATH)
    name = next(
        name
        for name, record in sorted(records.items())
        if accept(record)[0] and (STREAM_RAW_DIR / name).exists()
    )
    frame = load_source_bgr(STREAM_RAW_DIR / name)
    stream = LocalizerStream([str(STREAM_CLI_PATH)], STREAM_RESOURCE_DIR, tmp_path)
    stream.start()
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(lambda: [stream.locate(frame) for _ in range(6)])
            try:
                results = future.result(timeout=180)
            except FutureTimeout:
                stream.close()
                pytest.fail("locate() 未在超时前返回：流式 CLI 疑似读 stdin 到 EOF")
        assert all(record["attempts"] == 1 for record in results)
        assert results[0]["message"] != "Tracking Success"
        assert results[-1]["status"] == 0
        assert results[-1]["message"] == "Tracking Success"
    finally:
        stream.close()
