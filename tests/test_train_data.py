"""训练数据集：输入模式 -> 通道数、样本张量与样本权重。"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from endfield.dataset import REF_SUBDIR
from endfield.preprocess import IMG_H, IMG_W
from endfield.run_record import input_channels
from endfield.train.data import AngleDataset, filter_reference_gap


def write_ref_sample(
    directory: Path, name: str, observed: np.ndarray, reference: np.ndarray
) -> None:
    (directory / REF_SUBDIR).mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(directory / name), observed)
    assert cv2.imwrite(str(directory / REF_SUBDIR / name), reference)


def write_sample(directory: Path, name: str, value: int = 200) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    frame = np.full((IMG_H, IMG_W, 3), value, dtype=np.uint8)
    assert cv2.imwrite(str(directory / name), frame)


def test_input_channels_per_mode() -> None:
    assert input_channels("polar") == 3
    assert input_channels("ref") == 7
    with pytest.raises(ValueError, match="unknown input_mode"):
        input_channels("bogus")


def test_filter_reference_gap_excludes_strictly_over_threshold(tmp_path: Path) -> None:
    observed = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
    kept_ref = np.full((IMG_H, IMG_W, 4), 255, dtype=np.uint8)
    kept_ref[:, :108, 3] = 0  # 缺口占比 = 108/360 = 0.3，等于阈值应保留
    over_ref = kept_ref.copy()
    over_ref[:, 108, 3] = 0  # 缺口占比 = 109/360 > 0.3，排除
    write_ref_sample(tmp_path, "keep_r0.png", observed, kept_ref)
    write_ref_sample(tmp_path, "drop_r0.png", observed, over_ref)

    names = ["keep_r0.png", "drop_r0.png"]
    assert filter_reference_gap(names, tmp_path, 0.3) == ["keep_r0.png"]
    assert filter_reference_gap(names, tmp_path, 0.31) == names


def test_ref_dataset_concatenates_observed_reference_and_alpha(tmp_path: Path) -> None:
    observed = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
    observed[...] = (10, 20, 30)
    reference = np.zeros((IMG_H, IMG_W, 4), dtype=np.uint8)
    reference[...] = (40, 50, 60, 70)
    write_ref_sample(tmp_path, "sample_r90.5.png", observed, reference)

    tensor, angle, _ = AngleDataset(tmp_path, ["sample_r90.5.png"], input_mode="ref")[0]

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

    tensor, angle, weight = AngleDataset(
        tmp_path, ["sample_r0.png"], roll_augment=True, input_mode="ref"
    )[0]

    base, _, _ = AngleDataset(tmp_path, ["sample_r0.png"], input_mode="ref")[0]
    delta = int(round(angle.item())) % IMG_W
    assert tensor.shape == (7, IMG_H, IMG_W)
    assert weight.item() == pytest.approx(1.0)
    assert torch.allclose(tensor, torch.roll(base, delta, dims=2))


def test_dataset_weights_hard_names_and_leaves_others_at_one(tmp_path: Path) -> None:
    write_sample(tmp_path, "hard_r10.png")
    write_sample(tmp_path, "plain_r20.png")
    dataset = AngleDataset(
        tmp_path,
        ["hard_r10.png", "plain_r20.png"],
        hard_names={"hard_r10.png"},
        hard_weight=5.0,
    )

    hard_tensor, hard_angle, hard = dataset[0]
    plain_tensor, plain_angle, plain = dataset[1]

    assert hard_tensor.shape == (3, IMG_H, IMG_W)
    assert hard_angle.item() == pytest.approx(10.0)
    assert plain_angle.item() == pytest.approx(20.0)
    assert hard.dtype == torch.float32 and hard.item() == pytest.approx(5.0)
    assert plain.dtype == torch.float32 and plain.item() == pytest.approx(1.0)
    assert plain_tensor.shape == (3, IMG_H, IMG_W)


def test_dataset_weights_are_one_without_hard_names(tmp_path: Path) -> None:
    """名单为空（或名字不在名单里）时权重恒为 1，与 hard_weight 取值无关。"""
    write_sample(tmp_path, "plain_r20.png")
    dataset = AngleDataset(tmp_path, ["plain_r20.png"], hard_names=frozenset(), hard_weight=5.0)

    _, _, weight = dataset[0]

    assert weight.item() == pytest.approx(1.0)


def test_dataset_weight_survives_augmentation(tmp_path: Path) -> None:
    write_sample(tmp_path, "hard_r0.png")
    dataset = AngleDataset(
        tmp_path,
        ["hard_r0.png"],
        roll_augment=True,
        noise_augment=True,
        hard_names={"hard_r0.png"},
        hard_weight=5.0,
    )

    tensor, angle, weight = dataset[0]

    assert tensor.shape == (3, IMG_H, IMG_W)
    assert 0.0 <= angle.item() < 360.0
    assert weight.item() == pytest.approx(5.0)
