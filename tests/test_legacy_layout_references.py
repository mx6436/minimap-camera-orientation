"""数据换代（#44）：源码与文档不得再引用旧布局 `data/raw` / `val_manifest`。

扫描仓库内的文档/源码文件；gitignored 的本地数据与产物目录（`data/`、`runs/`、
`local/`、`.venv/`、`prototype/out/` 等）除外——历史只保留在 git 历史与本地产物里。
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LEGACY_NEEDLES = ("data/raw", "val_manifest")
SCAN_SUFFIXES = {".py", ".md", ".toml"}
SKIP_DIRS = {
    ".git",
    ".venv",
    ".ruff_cache",
    ".pi-subagents",
    "__pycache__",
    "data",
    "runs",
    "local",
    "out",
    "drafts",
}


def _scannable_files() -> list[Path]:
    files: list[Path] = []
    for path in REPO_ROOT.rglob("*"):
        if not path.is_file() or path.suffix not in SCAN_SUFFIXES:
            continue
        if path == Path(__file__).resolve():
            continue  # 本测试以 needle 为字面量
        relative = path.relative_to(REPO_ROOT)
        if any(part in SKIP_DIRS for part in relative.parts[:-1]):
            continue
        files.append(relative)
    return files


def test_source_and_docs_have_no_legacy_layout_references() -> None:
    hits = [
        f"{relative}:{needle}"
        for relative in _scannable_files()
        for needle in LEGACY_NEEDLES
        if needle in (REPO_ROOT / relative).read_text(encoding="utf-8", errors="replace")
    ]
    assert hits == []
