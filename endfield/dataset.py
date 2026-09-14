"""数据集布局：原始样本、前处理产物与划分视图的目录，以及参考流的并行子树名。

数据集生成器与训练读取共用这一份布局；`data/{train_raw,val_raw}` 是仅有的两个
人工维护目录。
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

TRAIN_RAW_DIR = REPO_ROOT / "data" / "train_raw"
VAL_RAW_DIR = REPO_ROOT / "data" / "val_raw"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
PROCESSED_REF_DIR = REPO_ROOT / "data" / "processed_ref"
TRAIN_DIR = REPO_ROOT / "data" / "train"
VAL_DIR = REPO_ROOT / "data" / "val"
TRAIN_REF_DIR = REPO_ROOT / "data" / "train_ref"
VAL_REF_DIR = REPO_ROOT / "data" / "val_ref"
LOCATOR_DIR = REPO_ROOT / "data" / "locator"
LOCATE_PATH = LOCATOR_DIR / "locate.jsonl"
LOCATE_SUMMARY_PATH = LOCATOR_DIR / "summary.json"

# 参考流在数据根中的并行子目录（processed_ref/ref、train_ref/ref 等）
REF_SUBDIR = "ref"
