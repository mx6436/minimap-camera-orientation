"""`map-locate` 进程驱动的测试：批量一轮与 `--stream` 常驻进程。

批量分片与进度的编排在 `cli/locate_dataset.py`，其集成测试见
tests/test_locate_dataset.py。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from endfield.polar import load_source_frame
from placement.locator import LocalizerStream, run_cli
from placement.placement import accept
from placement.records import load_records


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
STREAM_RAW_DIRS = (REPO_ROOT / "data" / "train_raw", REPO_ROOT / "data" / "val_raw")
STREAM_LOCATE_PATH = REPO_ROOT / "data" / "locator" / "locate.jsonl"


def stream_raw_path(name: str) -> Path | None:
    for directory in STREAM_RAW_DIRS:
        path = directory / name
        if path.is_file():
            return path
    return None


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
    for name, record in sorted(records.items()):
        raw_path = stream_raw_path(name)
        if accept(record)[0] and raw_path is not None:
            break
    else:
        pytest.skip("no accepted real sample with a raw png")
    frame = load_source_frame(raw_path)
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
