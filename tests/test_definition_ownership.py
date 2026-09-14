"""前处理定义的 ownership 守卫：字节关键实现只允许落在 endfield/preprocess.py。

CONTEXT.md「前处理定义」要求全仓不出现第二份展开或合成实现；本文件把该约束变成
可执行断言，防止帧到 ROI 的几何或条带入口再次长回 polar.py、placement/sample.py
或别处。定义文件的家写死在这里：搬迁必须显式改这条测试，不允许静默转移。
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

import endfield.polar
import endfield.preprocess
import placement.sample

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFINITION_PATH = REPO_ROOT / "endfield" / "preprocess.py"
# 扫描面：仓库根一层的脚本 + 各顶层包（新增顶层包时一并加进来）
SCANNED_PACKAGES = ("endfield", "placement", "cli")

DEFINITION_ENTRIES = (
    "observed_roi",
    "observed_strip",
    "strip_pair",
    "strip_pair_prepared",
)
FORBIDDEN_NAMES = ("ROI_CENTER", *DEFINITION_ENTRIES)


def scanned_paths() -> list[Path]:
    paths = list(REPO_ROOT.glob("*.py"))
    for package in SCANNED_PACKAGES:
        paths.extend((REPO_ROOT / package).glob("**/*.py"))
    return paths


def test_definition_file_is_where_we_think_it_is() -> None:
    assert Path(inspect.getsourcefile(endfield.preprocess)).resolve() == DEFINITION_PATH


def test_definition_entries_resolve_to_the_definition_file() -> None:
    for name in (*DEFINITION_ENTRIES, "definition_hash"):
        entry = getattr(endfield.preprocess, name)
        assert Path(inspect.getsourcefile(entry)).resolve() == DEFINITION_PATH


def test_consumers_do_not_define_or_reexport_definition_geometry() -> None:
    for module in (endfield.polar, placement.sample):
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
    for path in scanned_paths():
        if path.resolve() == DEFINITION_PATH:
            continue
        text = path.read_text(encoding="utf-8")
        if any(pattern.search(text) for pattern in patterns):
            offenders.add(path)
    assert not offenders, f"definition geometry must live only in {DEFINITION_PATH}: {offenders}"
