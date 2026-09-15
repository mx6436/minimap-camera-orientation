"""续训恢复态（`last.pt`）：载荷往返与缺失时的拒绝。"""

from __future__ import annotations

from pathlib import Path

import pytest

from endfield.model import ARCH_VERSION
from endfield.train.artifacts import load_resume_state, save_resume_state


def test_resume_state_round_trip(tmp_path: Path) -> None:
    save_resume_state(tmp_path / "last.pt", {"epoch": 7, "best_val": 3.5})
    state = load_resume_state(tmp_path / "last.pt")

    assert (state["arch"], state["epoch"], state["best_val"]) == (ARCH_VERSION, 7, 3.5)


def test_load_resume_state_rejects_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="last.pt"):
        load_resume_state(tmp_path / "last.pt")
