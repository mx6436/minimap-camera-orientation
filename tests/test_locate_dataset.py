"""locate-dataset 入口行为：两侧原始目录并集、逐样本路径解析、产物与汇总落盘。

用可执行假 CLI（同 tests/test_locate.py 的 run_cli 模式）走完整编排，不依赖
local/maplocator/；CLI 本身另有集成验证（真实重跑见 data/locator 的核对产物）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from cli import locate_dataset
from endfield import dataset

FAKE_CLI = (
    "import json, sys\n"
    "from pathlib import Path\n"
    "args = sys.argv[1:]\n"
    "if not args:\n"
    "    print('usage: map-locate --resource-dir <dir> [--stream]', file=sys.stderr)\n"
    "    raise SystemExit(2)\n"
    "assert '--resource-dir' in args, args\n"
    "for line in sys.stdin:\n"
    "    path = Path(line.strip())\n"
    "    if not path.name:\n"
    "        continue\n"
    "    assert path.is_file(), path\n"
    "    print(json.dumps({'name': path.name, 'path': str(path), 'status': 0,"
    " 'message': 'Global Search Success', 'zone': 'Fake_Base', 'x': 1.0, 'y': 2.0,"
    " 'rot': 0.0, 'scale': 1.0, 'locConf': 0.9, 'isHeld': False, 'latencyMs': 1,"
    " 'attempts': 1, 'elapsedMs': 2}))\n"
)


def write_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake png")


def make_workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "maplocator"
    (workspace / "resource").mkdir(parents=True)
    cli = workspace / "bin" / "map-locate"
    cli.parent.mkdir(parents=True)
    cli.write_text(f"#!{sys.executable}\n" + FAKE_CLI, encoding="utf-8")
    cli.chmod(0o755)
    return workspace


def run_main(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *argv: str) -> Path:
    out = tmp_path / "locator" / "locate.jsonl"
    monkeypatch.setattr(locate_dataset, "OUT_PATH", out)
    monkeypatch.setattr(locate_dataset, "SUMMARY_PATH", out.parent / "summary.json")
    workspace = make_workspace(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["locate-dataset", "--maplocator-root", str(workspace), *argv],
    )
    locate_dataset.main()
    return out


def read_records(path: Path) -> dict[str, dict]:
    return {json.loads(line)["name"]: json.loads(line) for line in path.read_text().splitlines()}


def test_main_locates_union_of_both_raw_dirs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    train_raw, val_raw = tmp_path / "train_raw", tmp_path / "val_raw"
    write_png(train_raw / "a_r0.png")
    write_png(val_raw / "b_r90.png")
    monkeypatch.setattr(locate_dataset, "RAW_DIRS", (train_raw, val_raw))

    out = run_main(tmp_path, monkeypatch, "--jobs", "2")

    records = read_records(out)
    assert sorted(records) == ["a_r0.png", "b_r90.png"]
    assert Path(records["a_r0.png"]["path"]) == train_raw / "a_r0.png"
    assert Path(records["b_r90.png"]["path"]) == val_raw / "b_r90.png"
    summary = json.loads((out.parent / "summary.json").read_text(encoding="utf-8"))
    assert summary["total"] == 2
    assert summary["accepted"] == 2


def test_default_raw_dirs_cover_every_maintained_raw_directory() -> None:
    """定位输入 = 全部人工维护目录：漏掉 hard_raw 会让困难样本静默进 skipped。"""
    assert locate_dataset.RAW_DIRS == (*dataset.TRAIN_RAW_DIRS, dataset.VAL_RAW_DIR)
    assert dataset.HARD_RAW_DIR in locate_dataset.RAW_DIRS


def test_main_rejects_name_present_in_both_raw_dirs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    train_raw, val_raw = tmp_path / "train_raw", tmp_path / "val_raw"
    write_png(train_raw / "a_r0.png")
    write_png(val_raw / "a_r0.png")
    monkeypatch.setattr(locate_dataset, "RAW_DIRS", (train_raw, val_raw))

    with pytest.raises(SystemExit, match="both"):
        run_main(tmp_path, monkeypatch)
