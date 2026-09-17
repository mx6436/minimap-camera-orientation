"""数据集布局：原始样本、前处理产物与划分视图的目录，以及数据准备的产物布局。

数据集生成与训练读取共用这一份布局；`data/{train_raw,hard_raw,val_raw}` 是仅有的三个
人工维护目录：训练侧 = `train_raw ∪ hard_raw`（样本落在哪个训练目录不改变并集），
val 侧只有 `val_raw`。`DatasetLayout` 把一个输入模式的产物根、划分视图与并行子树收成一个值，
数据准备与训练读取不再各持一份目录常量。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from endfield.data_utils import union_png_samples

REPO_ROOT = Path(__file__).resolve().parents[1]

TRAIN_RAW_DIR = REPO_ROOT / "data" / "train_raw"
HARD_RAW_DIR = REPO_ROOT / "data" / "hard_raw"
VAL_RAW_DIR = REPO_ROOT / "data" / "val_raw"

# 训练侧原始目录：口径只此一处；样本移入 hard_raw 不改变训练侧并集
TRAIN_RAW_DIRS = (TRAIN_RAW_DIR, HARD_RAW_DIR)

# 参考流在数据根中的并行子目录（processed_ref/ref、train_ref/ref 等）
REF_SUBDIR = "ref"


@dataclass(frozen=True)
class DatasetLayout:
    """一个输入模式的数据集布局：产物根、划分视图目录与并行子树名。"""

    processed_dir: Path
    train_dir: Path
    val_dir: Path
    subdirs: tuple[str, ...] = ()


POLAR_LAYOUT = DatasetLayout(
    processed_dir=REPO_ROOT / "data" / "processed",
    train_dir=REPO_ROOT / "data" / "train",
    val_dir=REPO_ROOT / "data" / "val",
)
REF_LAYOUT = DatasetLayout(
    processed_dir=REPO_ROOT / "data" / "processed_ref",
    train_dir=REPO_ROOT / "data" / "train_ref",
    val_dir=REPO_ROOT / "data" / "val_ref",
    subdirs=(REF_SUBDIR,),
)

PROCESSED_DIR = POLAR_LAYOUT.processed_dir
TRAIN_DIR = POLAR_LAYOUT.train_dir
VAL_DIR = POLAR_LAYOUT.val_dir
PROCESSED_REF_DIR = REF_LAYOUT.processed_dir
TRAIN_REF_DIR = REF_LAYOUT.train_dir
VAL_REF_DIR = REF_LAYOUT.val_dir

LOCATOR_DIR = REPO_ROOT / "data" / "locator"
LOCATE_PATH = LOCATOR_DIR / "locate.jsonl"
LOCATE_SUMMARY_PATH = LOCATOR_DIR / "summary.json"


def raw_samples(
    train_raw_dirs: Sequence[Path] = TRAIN_RAW_DIRS, val_raw_dir: Path = VAL_RAW_DIR
) -> dict[str, Path]:
    """训练侧目录 + 验证目录并集 -> {样本名: 源文件}；同名跨目录硬报错。"""
    return union_png_samples((*train_raw_dirs, val_raw_dir))


def directory_split(
    train_names: Sequence[str], val_names: Sequence[str]
) -> tuple[list[str], list[str]]:
    """目录即划分：两侧名单排序返回；任一侧为空即报错。"""
    train, val = sorted(train_names), sorted(val_names)
    if not train:
        raise SystemExit("train split is empty: no usable samples on the train side")
    if not val:
        raise SystemExit("val split is empty: no usable samples on the val side")
    return train, val
