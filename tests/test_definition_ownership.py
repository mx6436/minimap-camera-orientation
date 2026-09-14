"""前处理定义的 ownership 守卫：字节关键实现只允许落在 endfield/preprocess.py。

CONTEXT.md「前处理定义」要求全仓不出现第二份展开或合成实现；本文件把该约束变成
可执行断言，防止帧到 ROI 的几何或条带入口再次长回 polar.py / ref.py 里。
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

import endfield.polar
import endfield.preprocess
import endfield.ref

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFINITION_FILE = Path(inspect.getsourcefile(endfield.preprocess))

DEFINITION_ENTRIES = (
    "observed_roi",
    "observed_strip",
    "strip_pair",
    "strip_pair_prepared",
)
FORBIDDEN_NAMES = ("ROI_CENTER", *DEFINITION_ENTRIES)


def test_definition_entries_resolve_to_the_definition_file() -> None:
    for name in (*DEFINITION_ENTRIES, "definition_hash"):
        entry = getattr(endfield.preprocess, name)
        assert Path(inspect.getsourcefile(entry)) == DEFINITION_FILE


def test_polar_and_ref_do_not_define_or_reexport_definition_geometry() -> None:
    for module in (endfield.polar, endfield.ref):
        for name in FORBIDDEN_NAMES:
            assert not hasattr(module, name), f"{module.__name__} must not expose {name}"


def test_no_second_definition_outside_the_definition_file() -> None:
    patterns = (
        re.compile(r"^def observed_roi\b", re.M),
        re.compile(r"^def observed_strip\b", re.M),
        re.compile(r"^def strip_pair\b", re.M),
        re.compile(r"^ROI_CENTER\s*=", re.M),
    )
    offenders: set[Path] = set()
    for path in (*REPO_ROOT.glob("*.py"), *REPO_ROOT.glob("endfield/**/*.py")):
        if path.resolve() == DEFINITION_FILE.resolve():
            continue
        text = path.read_text(encoding="utf-8")
        if any(pattern.search(text) for pattern in patterns):
            offenders.add(path)
    assert not offenders, f"definition geometry must live only in {DEFINITION_FILE}: {offenders}"
