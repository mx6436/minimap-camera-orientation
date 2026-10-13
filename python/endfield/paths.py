"""仓库路径基准：`python/..` 是 `data/` 与 `runs/` 的根。"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def repo_path(value: Path) -> Path:
    """相对路径按仓库根解析；绝对路径原样返回。"""
    return value if value.is_absolute() else REPO_ROOT / value
