"""数据集与加载器：文件名角度标注 → (RGB 张量, sin/cos 目标)。"""

from __future__ import annotations

import hashlib
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from endfield.data_utils import angle_target, load_rgb, parse_angle

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_DIR = REPO_ROOT / "data" / "train"
VAL_DIR = REPO_ROOT / "data" / "val"


class AngleDataset(Dataset):
    def __init__(
        self,
        directory: Path,
        names: list[str],
        augment: bool = False,
        rotate: bool = True,
    ) -> None:
        self.directory = directory
        self.names = names
        self.augment = augment
        self.rotate = rotate

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        name = self.names[index]
        angle = parse_angle(Path(name))
        array = load_rgb(self.directory / name).astype(np.float32) / 255.0
        if self.augment:
            if self.rotate and random.random() < 0.5:
                delta = 15 * random.randint(1, 23)
                # 1°/列角度轴：内容顺时针转 delta 度 == 列右移 delta（严格无损）。
                array = np.roll(array, delta, axis=1)
                angle = (angle + delta) % 360
            if random.random() < 0.5:
                array = array + np.random.normal(0.0, 0.02, array.shape).astype(np.float32)
            array = np.clip(array, 0.0, 1.0)
        tensor = torch.from_numpy(array.transpose(2, 0, 1)).contiguous()
        target = torch.from_numpy(angle_target(angle))
        return tensor, target


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
