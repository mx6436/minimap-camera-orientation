"""训练数据集：输入模式 -> 通道数与样本张量。"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from endfield.polar import IMG_H, IMG_W
from endfield.ref import REF_SUBDIR
from endfield.train.data import AngleDataset, input_channels


def write_ref_sample(
    directory: Path, name: str, observed: np.ndarray, reference: np.ndarray
) -> None:
    (directory / REF_SUBDIR).mkdir(parents=True)
    assert cv2.imwrite(str(directory / name), observed)
    assert cv2.imwrite(str(directory / REF_SUBDIR / name), reference)


def test_input_channels_per_mode() -> None:
    assert input_channels("polar") == 3
    assert input_channels("ref") == 7
    with pytest.raises(ValueError, match="unknown input_mode"):
        input_channels("bogus")


def test_ref_dataset_concatenates_observed_reference_and_alpha(tmp_path: Path) -> None:
    observed = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
    observed[...] = (10, 20, 30)
    reference = np.zeros((IMG_H, IMG_W, 4), dtype=np.uint8)
    reference[...] = (40, 50, 60, 70)
    write_ref_sample(tmp_path, "sample_r90.5.png", observed, reference)

    tensor, angle = AngleDataset(tmp_path, ["sample_r90.5.png"], input_mode="ref")[0]

    assert angle.item() == pytest.approx(90.5)
    expected = np.concatenate([observed, reference], axis=2).astype(np.float32) / 255.0
    assert tensor.shape == (7, IMG_H, IMG_W)
    assert torch.allclose(tensor, torch.from_numpy(expected.transpose(2, 0, 1)))


def test_ref_dataset_roll_augment_shifts_all_channels(tmp_path: Path) -> None:
    observed = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
    observed[:, 100] = (255, 0, 0)
    reference = np.zeros((IMG_H, IMG_W, 4), dtype=np.uint8)
    reference[:, 100] = (0, 255, 0, 128)
    write_ref_sample(tmp_path, "sample_r0.png", observed, reference)

    tensor, angle = AngleDataset(tmp_path, ["sample_r0.png"], roll_augment=True, input_mode="ref")[
        0
    ]

    base, _ = AngleDataset(tmp_path, ["sample_r0.png"], input_mode="ref")[0]
    delta = int(round(angle.item())) % IMG_W
    assert tensor.shape == (7, IMG_H, IMG_W)
    assert torch.allclose(tensor, torch.roll(base, delta, dims=2))
