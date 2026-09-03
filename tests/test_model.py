"""ConeCNN 的构造保证与配套解码/目标编码。"""

from __future__ import annotations

import numpy as np
import torch

from endfield.model import (
    EXPECTED_CONE_PARAMETER_COUNT,
    ConeCNN,
    build_model,
    count_trainable_parameters,
    decode_logits,
    smoothed_targets,
    target_angles,
)


def test_parameter_count_matches_expected() -> None:
    model = ConeCNN()
    assert count_trainable_parameters(model) == EXPECTED_CONE_PARAMETER_COUNT


def test_forward_shape() -> None:
    model = ConeCNN().eval()
    with torch.no_grad():
        logits = model(torch.zeros(2, 3, 44, 360))
    assert logits.shape == (2, 360)


def test_shift_equivariance() -> None:
    """输入沿方位角轴平移 δ° 时 logits 与解码角都精确平移 δ°。"""
    model = ConeCNN().eval()
    x = torch.rand(2, 3, 44, 360)
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


def test_decode_logits_interpolates_sub_degree() -> None:
    """峰旁邻 bin 抬高时解码角落在两 bin 之间，且随 logit 变化连续而非 1° 跳变。"""
    base = torch.full((1, 360), -10.0)
    base[0, 180] = 10.0
    sharper = base.clone()
    sharper[0, 181] = 10.0
    even = sharper.clone()
    even[0, 181] = 10.6
    angles, _ = decode_logits(torch.cat([base, sharper, even]))
    assert angles[0] == 180.0
    assert 180.0 < angles[1] < 181.0
    assert angles[1] < angles[2] < 181.0


def test_smoothed_targets_roundtrip() -> None:
    angles = np.array([0.0, 179.5, 359.9])
    targets = smoothed_targets(angles)
    assert torch.allclose(targets.sum(dim=1), torch.ones(3), atol=1e-5)
    peaks = targets.argmax(dim=1).numpy()
    # 179.5° 在 bin 179/180 正中，平局由浮点打破
    assert np.allclose(peaks, [0, 179, 0])
    recovered = target_angles(np.stack([np.sin(np.deg2rad(angles)), np.cos(np.deg2rad(angles))], 1))
    assert np.allclose(recovered, angles % 360.0, atol=1e-4)


def test_build_model_dispatch() -> None:
    assert isinstance(build_model("cone", {}), ConeCNN)
    assert not isinstance(build_model("angle_cnn", {}), ConeCNN)
