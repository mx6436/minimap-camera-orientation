"""AzimuthNet 的构造保证与配套解码/目标编码。"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from endfield.model import (
    ARCH_VERSION,
    EXPECTED_PARAMETER_COUNT,
    AzimuthNet,
    count_trainable_parameters,
    decode_logits,
    load_model,
    smoothed_targets,
    target_angles,
)
from endfield.polar import IMG_H, IMG_W


def test_parameter_count_matches_expected() -> None:
    model = AzimuthNet()
    assert count_trainable_parameters(model) == EXPECTED_PARAMETER_COUNT


def test_forward_shape() -> None:
    model = AzimuthNet().eval()
    with torch.no_grad():
        logits = model(torch.zeros(2, 3, IMG_H, IMG_W))
    assert logits.shape == (2, 360)


def test_shift_equivariance() -> None:
    """输入沿方位角轴平移 δ° 时 logits 与解码角都精确平移 δ°。"""
    model = AzimuthNet().eval()
    x = torch.rand(2, 3, IMG_H, IMG_W)
    with torch.no_grad():
        base = model(x)
        rolled = model(torch.roll(x, 10, dims=3))
    logit_shift_error = (rolled - torch.roll(base, 10, dims=1)).abs().max().item()
    assert logit_shift_error < 1e-4
    base_angles, _ = decode_logits(base)
    rolled_angles, _ = decode_logits(rolled)
    error = (rolled_angles - base_angles - 10.0) % 360.0
    error = np.minimum(error, 360.0 - error)
    assert np.all(error < 1e-3)


def test_decode_logits_peaks_at_argmax() -> None:
    logits = torch.full((3, 360), -10.0)
    logits[0, 0] = 10.0
    logits[1, 359] = 10.0
    logits[2, 180] = 10.0
    angles, confidence = decode_logits(logits)
    assert np.allclose(angles, [0.0, 359.0, 180.0], atol=1e-6)
    assert np.all(confidence > 0.99)


def test_confidence_is_resultant_length_times_alignment() -> None:
    """均匀分布趋 0，集中分布趋 1；完美学到的 σ=2° 软标签后验虽宽带但仍应接近 1。"""
    uniform = torch.zeros(2, 360)
    broad_probs = smoothed_targets(np.array([90.0, 270.0]), sigma=2.0)
    broad = torch.log(broad_probs.clamp_min(1e-12))  # softmax(log p) = p
    _, confidence = decode_logits(torch.cat([uniform, broad]))
    assert np.all(confidence[:2] < 1e-6)
    assert np.all(confidence[2:] > 0.99)


def test_confidence_penalizes_decoded_resultant_misalignment() -> None:
    """0.6@10° + 0.4@100°：合成模长 ~0.72、合成方向 ~43.7°、解码 10°，
    置信度折减至 ~0.60；200° 单峰的方向差恰跨 360° 周期，仍应接近 1。"""
    skewed = torch.zeros(2, 360)
    skewed[0, 10] = 0.6
    skewed[0, 100] = 0.4
    skewed[1, 200] = 1.0
    _, confidence = decode_logits(torch.log(skewed.clamp_min(1e-12)))
    assert 0.55 < confidence[0] < 0.65
    assert confidence[1] > 0.99


def test_smoothed_targets_roundtrip() -> None:
    angles = np.array([0.0, 179.5, 359.9])
    targets = smoothed_targets(angles)
    assert torch.allclose(targets.sum(dim=1), torch.ones(3), atol=1e-5)
    peaks = targets.argmax(dim=1).numpy()
    # 179.5° 在 bin 179/180 正中，平局由浮点打破
    assert np.allclose(peaks, [0, 179, 0])
    recovered = target_angles(np.stack([np.sin(np.deg2rad(angles)), np.cos(np.deg2rad(angles))], 1))
    assert np.allclose(recovered, angles % 360.0, atol=1e-4)


def test_load_model_rejects_invalid_checkpoint(tmp_path) -> None:
    path = tmp_path / "bad.pt"
    torch.save({"weights": {}}, path)
    with pytest.raises(ValueError, match="invalid checkpoint"):
        load_model(path)


def test_load_model_rejects_legacy_checkpoint(tmp_path) -> None:
    path = tmp_path / "legacy.pt"
    torch.save(
        {
            "model": {"stale": torch.zeros(1)},
            "arch": ARCH_VERSION,
            "config": {"architecture": "cone"},
        },
        path,
    )
    with pytest.raises(ValueError, match="incompatible checkpoint weights"):
        load_model(path)


def test_outermost_radius_rows_reach_output() -> None:
    """回归：最外圈半径行的证据不得被径向池化丢弃。"""
    model = AzimuthNet().eval()
    base = torch.zeros(1, 3, IMG_H, IMG_W)
    perturbed = base.clone()
    perturbed[:, :, IMG_H - 1, :] = 1.0
    with torch.no_grad():
        difference = (model(perturbed) - model(base)).abs().max().item()
    assert difference > 1e-4
