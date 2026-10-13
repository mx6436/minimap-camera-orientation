"""数据集布局：`data/` 根、raw 名单划分与 C++ 预处理产物的取址。

`data/{train_raw,hard_raw,val_raw}` 是仅有的三个人工维护目录：训练侧 =
`train_raw ∪ hard_raw`（样本落在哪个训练目录不改变并集），val 侧只有 `val_raw`；划分
在内存里按文件名做，不依赖链接视图。条带产物由 MaaEnd C++ 数据准备工具写入
`data/processed_ref/<name>.png`（观测 3 通道）与 `data/processed_ref/ref/<name>.png`
（参考 4 通道）。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from endfield.data_utils import png_names
from endfield.paths import REPO_ROOT

DEFAULT_DATA_ROOT = REPO_ROOT / "data"
# 参考流在 processed_ref 中的并行子目录
REF_SUBDIR = "ref"


@dataclass(frozen=True)
class DataLayout:
    """一个 `data/` 根的布局：三处 raw 名单目录 + processed_ref 产物目录。"""

    root: Path

    @property
    def train_raw(self) -> Path:
        return self.root / "train_raw"

    @property
    def hard_raw(self) -> Path:
        return self.root / "hard_raw"

    @property
    def val_raw(self) -> Path:
        return self.root / "val_raw"

    @property
    def processed_ref(self) -> Path:
        return self.root / "processed_ref"

    def observed_path(self, name: str) -> Path:
        return self.processed_ref / name

    def reference_path(self, name: str) -> Path:
        return self.processed_ref / REF_SUBDIR / name

    def available(self, name: str) -> bool:
        """观测与参考两份条带都在：数据准备工具跳过（定位失败/低分等）的样本不可用。"""
        return self.observed_path(name).is_file() and self.reference_path(name).is_file()


@dataclass(frozen=True)
class SampleSplit:
    """raw 目录的名单划分：训练侧 = `train ∪ hard`，`hard` 是训练损失计权名单。"""

    train: list[str]
    val: list[str]
    hard: list[str]


def split_names(layout: DataLayout) -> SampleSplit:
    """按 raw 目录名单在内存划分；同名跨目录即报错（名单必须无歧义）。"""
    names = {
        "train_raw": png_names(layout.train_raw),
        "hard_raw": png_names(layout.hard_raw),
        "val_raw": png_names(layout.val_raw),
    }
    seen: dict[str, str] = {}
    for directory, listed in names.items():
        for name in listed:
            if name in seen:
                raise SystemExit(f"sample name in both {seen[name]} and {directory}: {name}")
            seen[name] = directory
    return SampleSplit(
        train=sorted(names["train_raw"] + names["hard_raw"]),
        val=names["val_raw"],
        hard=names["hard_raw"],
    )
