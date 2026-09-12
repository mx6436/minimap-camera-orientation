"""数据集与加载器：文件名角度标注 → (BGR 张量, 角度标量)。"""

from __future__ import annotations

import hashlib
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from endfield.data_utils import load_bgr, load_bgra, parse_angle
from endfield.ref import REF_CHANNELS, REF_SUBDIR, ref_tensor, reference_gap_fraction

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_DIR = REPO_ROOT / "data" / "train"
VAL_DIR = REPO_ROOT / "data" / "val"
TRAIN_REF_DIR = REPO_ROOT / "data" / "train_ref"
VAL_REF_DIR = REPO_ROOT / "data" / "val_ref"

SPLIT_DIRS: dict[str, tuple[Path, Path]] = {
    "polar": (TRAIN_DIR, VAL_DIR),
    "ref": (TRAIN_REF_DIR, VAL_REF_DIR),
}

# 输入模式的通道数：ref 为 [obs.BGR, ref.BGR, ref.A]（见 endfield/ref.py）
INPUT_CHANNELS: dict[str, int] = {"polar": 3, "ref": REF_CHANNELS}


def split_dirs(input_mode: str) -> tuple[Path, Path]:
    """输入模式 -> (训练目录, 验证目录)。"""
    try:
        return SPLIT_DIRS[input_mode]
    except KeyError:
        raise ValueError(f"unknown input_mode: {input_mode!r}") from None


def input_channels(input_mode: str) -> int:
    """输入模式 -> AzimuthNet 首层通道数。"""
    try:
        return INPUT_CHANNELS[input_mode]
    except KeyError:
        raise ValueError(f"unknown input_mode: {input_mode!r}") from None


def filter_reference_gap(names: list[str], directory: Path, max_missing: float) -> list[str]:
    """按参考条带的环内缺失占比过滤训练样本：**严格大于**阈值即排除（等于保留）。

    缺失占比 = 42x360 条带中 `ref.A < 255` 的像素比例（读 `ref/` 流，与
    `prepare_data.py --mode ref` 落盘一致，各半径等权）。只选样本，不改磁盘数据。
    """
    kept: list[str] = []
    for name in names:
        reference = load_bgra(directory / REF_SUBDIR / name)
        if reference_gap_fraction(reference) <= max_missing:
            kept.append(name)
    return kept


class AngleDataset(Dataset):
    def __init__(
        self,
        directory: Path,
        names: list[str],
        noise_augment: bool = False,
        roll_augment: bool = False,
        input_mode: str = "polar",
    ) -> None:
        self.directory = directory
        self.names = names
        self.noise_augment = noise_augment
        self.roll_augment = roll_augment
        self.input_mode = input_mode
        self.channels = input_channels(input_mode)

    def __len__(self) -> int:
        return len(self.names)

    def _load(self, name: str) -> np.ndarray:
        if self.input_mode == "ref":
            observed = load_bgr(self.directory / name)
            reference = load_bgra(self.directory / REF_SUBDIR / name)
            return ref_tensor(observed, reference)
        return load_bgr(self.directory / name)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        name = self.names[index]
        angle = parse_angle(Path(name))
        array = self._load(name).astype(np.float32) / 255.0
        if array.shape[2] != self.channels:
            raise ValueError(
                f"{name}: expected {self.channels} channels for input_mode "
                f"{self.input_mode!r}, got {array.shape[2]}"
            )
        if self.roll_augment:
            # 架构对角向平移精确等变，滚动后的样本严格有效；随机 δ 同时
            # 平衡各 bin 的有效样本量，不受标注角度分布影响
            delta = random.randrange(360)
            array = np.roll(array, delta, axis=1)
            angle = (angle + delta) % 360.0
        if self.noise_augment:
            if random.random() < 0.5:
                array = array + np.random.normal(0.0, 0.02, array.shape).astype(np.float32)
            array = np.clip(array, 0.0, 1.0)
        tensor = torch.from_numpy(array.transpose(2, 0, 1)).contiguous()
        return tensor, torch.tensor(angle, dtype=torch.float32)


def names_fingerprint(names: list[str]) -> str:
    """跨 run 校验数据集一致性的摘要：排序、\n 连接后取 sha256。"""
    return hashlib.sha256("\n".join(sorted(names)).encode("utf-8")).hexdigest()


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> DataLoader:
    if generator is None:
        generator = torch.Generator()
        generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        generator=generator,
        pin_memory=device.type == "cuda",
    )
