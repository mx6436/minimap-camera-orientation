"""数据集与加载器：文件名角度标注 → (BGR 张量, 角度标量, 权重标量)。"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Set
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from endfield.data_utils import load_bgr, load_bgra, parse_angle
from endfield.dataset import (
    REF_SUBDIR,
    TRAIN_DIR,
    TRAIN_REF_DIR,
    VAL_DIR,
    VAL_REF_DIR,
)
from endfield.input_encoding import assemble_ref_pair, reference_gap_fraction
from endfield.run_record import InputMode, input_channels

SPLIT_DIRS: dict[InputMode, tuple[Path, Path]] = {
    InputMode.POLAR: (TRAIN_DIR, VAL_DIR),
    InputMode.REF: (TRAIN_REF_DIR, VAL_REF_DIR),
}


def split_dirs(input_mode: InputMode | str) -> tuple[Path, Path]:
    """输入模式 -> (训练目录, 验证目录)。"""
    return SPLIT_DIRS[InputMode.parse(input_mode)]


def filter_reference_gap(names: list[str], directory: Path, max_missing: float) -> list[str]:
    """按参考条带的环内缺失占比过滤训练样本：**严格大于**阈值即排除（等于保留）。

    缺失占比 = 42x360 条带中 `ref.A < 255` 的像素比例（读 `ref/` 流，与
    `prepare-data --mode ref` 落盘一致，各半径等权）。只选样本，不改磁盘数据。
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
        input_mode: InputMode | str = InputMode.POLAR,
        hard_names: Set[str] = frozenset(),
        hard_weight: float = 1.0,
    ) -> None:
        self.directory = directory
        self.names = names
        self.noise_augment = noise_augment
        self.roll_augment = roll_augment
        self.input_mode = InputMode.parse(input_mode)
        self.channels = input_channels(self.input_mode)
        # 困难样本权重：名单由调用方给出（本模块不认识 data/hard_raw），增广与权重正交
        self.hard_names = hard_names
        self.hard_weight = hard_weight

    def __len__(self) -> int:
        return len(self.names)

    def _load(self, name: str) -> np.ndarray:
        if self.input_mode is InputMode.REF:
            observed = load_bgr(self.directory / name)
            reference = load_bgra(self.directory / REF_SUBDIR / name)
            return assemble_ref_pair(observed, reference)
        return load_bgr(self.directory / name)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        name = self.names[index]
        angle = parse_angle(Path(name))
        array = self._load(name).astype(np.float32) / 255.0
        if array.shape[2] != self.channels:
            raise ValueError(
                f"{name}: expected {self.channels} channels for input_mode "
                f"{self.input_mode.value!r}, got {array.shape[2]}"
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
        weight = self.hard_weight if name in self.hard_names else 1.0
        return (
            tensor,
            torch.tensor(angle, dtype=torch.float32),
            torch.tensor(weight, dtype=torch.float32),
        )


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
