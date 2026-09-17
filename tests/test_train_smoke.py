"""`train` 端到端冒烟：合成小数据集跑通一个 epoch，覆盖有 / 无困难样本两种情形。

不依赖真实 `data/processed*`（真实首跑见 05）：把划分目录与 `hard_raw` 重定向到 tmp 后
走 `cli.train.main()` 的完整路径——装配 hard 名单、训练、评估、落盘档案与产物。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from cli import train
from endfield.preprocess import IMG_H, IMG_W
from endfield.run_record import RECORD_NAME

TRAIN_NAMES = ("a_r0.png", "b_r90.png", "h_r45.png")
VAL_NAMES = ("c_r180.png",)
HARD_NAME = "h_r45.png"


def write_processed(directory: Path, names: tuple[str, ...]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    frame = np.full((IMG_H, IMG_W, 3), 128, dtype=np.uint8)
    for name in names:
        assert cv2.imwrite(str(directory / name), frame)


def run_smoke(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    hard: tuple[str, ...],
    resume: bool = False,
    extra_config: str = "",
) -> Path:
    """跑一次 `--smoke` 训练，返回 run 目录。"""
    train_dir, val_dir = tmp_path / "train", tmp_path / "val"
    hard_raw = tmp_path / "hard_raw"
    write_processed(train_dir, TRAIN_NAMES)
    write_processed(val_dir, VAL_NAMES)
    write_processed(hard_raw, hard)
    config_path = tmp_path / "train.toml"
    config_path.write_text(
        'input_mode = "polar"\nbatch_size = 2\nepochs = 1\nprecision = "fp32"\n' + extra_config,
        encoding="utf-8",
    )
    out = tmp_path / "run"
    monkeypatch.setattr(train, "split_dirs", lambda _mode: (train_dir, val_dir))
    monkeypatch.setattr(train, "HARD_RAW_DIR", hard_raw)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train",
            "--config",
            str(config_path),
            "--run-dir",
            str(out),
            "--device",
            "cpu",
            "--threads",
            "1",
            "--no-compile",
            "--smoke",
            *(["--resume"] if resume else []),
        ],
    )
    train.main()
    return out


def read_record(run_path: Path) -> dict:
    return json.loads((run_path / RECORD_NAME).read_text(encoding="utf-8"))


def test_smoke_train_without_hard_samples(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out = run_smoke(tmp_path, monkeypatch, hard=())

    record = read_record(out)
    assert record["hard_weight"] == 5.0
    assert record["hard_count"] == 0
    assert (out / "best.pt").is_file()
    assert (out / "summary.json").is_file()


def test_smoke_train_with_hard_sample_in_the_split(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = run_smoke(tmp_path, monkeypatch, hard=(HARD_NAME,))

    record = read_record(out)
    assert record["hard_count"] == 1
    assert (out / "best.pt").is_file()


def test_smoke_train_warns_when_hard_sample_is_not_in_the_split(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """hard_raw 有样本但全被剔除：权重无处可落，显式警示且档案 hard_count = 0。"""
    out = run_smoke(tmp_path, monkeypatch, hard=("missing_r0.png",))

    printed = capsys.readouterr().out
    assert "has no effect" in printed
    assert read_record(out)["hard_count"] == 0


def test_resume_follows_the_hard_premise(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """同一 hard 前提可续训；改 hard_weight 即按既有口径拒绝。"""
    run_smoke(tmp_path, monkeypatch, hard=(HARD_NAME,))

    run_smoke(tmp_path, monkeypatch, hard=(HARD_NAME,), resume=True)

    with pytest.raises(SystemExit, match="不一致"):
        run_smoke(
            tmp_path,
            monkeypatch,
            hard=(HARD_NAME,),
            resume=True,
            extra_config="hard_weight = 2.5\n",
        )
