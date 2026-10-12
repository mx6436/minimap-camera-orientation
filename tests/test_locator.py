"""`map-locate` 进程驱动的测试：批量一轮。

批量分片与进度的编排在 `cli/locate_dataset.py`，其集成测试见
tests/test_locate_dataset.py。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from placement.locator import run_cli


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
