"""前处理定义模块（#25）的行为规格：几何、采样、参考合成与越界。

断言通过公开入口（`observed_strip` / `strips`）观察行为，期望值是手算的规格
（方位零点是正北、顺时针为正；半径自上而下由内向外），不复用实现内部公式。
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from endfield import preprocess

WHITE = 255


def _roi() -> np.ndarray:
    return np.zeros((preprocess.ROI_H, preprocess.ROI_W, 3), dtype=np.uint8)


def _mark(roi: np.ndarray, xs: tuple[int, ...], ys: tuple[int, ...]) -> None:
    for y in ys:
        for x in xs:
            roi[y, x] = WHITE


@pytest.mark.parametrize(
    ("column", "xs", "ys"),
    [
        (0, (59,), (19, 20)),  # 正北，半径 40.5
        (90, (99, 100), (60,)),  # 正东
        (180, (59,), (100, 101)),  # 正南
        (270, (18, 19), (60,)),  # 正西
    ],
)
def test_azimuth_is_north_zero_and_clockwise(
    column: int, xs: tuple[int, ...], ys: tuple[int, ...]
) -> None:
    roi = _roi()
    _mark(roi, xs, ys)

    observed = preprocess.observed_strip(roi)

    assert observed.shape == (42, 360, 3)
    assert observed.dtype == np.uint8
    assert observed[..., 0].sum(axis=0).argmax() == column


def test_radius_rows_run_inward_to_outward() -> None:
    roi = _roi()
    _mark(roi, (59,), (19, 20))  # 半径 40.5 -> 行索引 (40.5 - 12) - 0.5 = 28

    observed = preprocess.observed_strip(roi)

    assert observed[..., 0].sum(axis=1).argmax() == 28


def _bgra(width: int, height: int, rgb: tuple[int, int, int], alpha: int) -> np.ndarray:
    asset = np.zeros((height, width, 4), dtype=np.uint8)
    asset[..., :3] = rgb
    asset[..., 3] = alpha
    return asset


@pytest.mark.parametrize("scale", [1.0, 15.0 / 16.0])
def test_reference_opaque_constant_asset_passes_through(scale: float) -> None:
    observed = np.full((preprocess.ROI_H, preprocess.ROI_W, 3), 200, dtype=np.uint8)
    asset = _bgra(160, 140, (10, 20, 30), 255)

    obs, ref = preprocess.strips(observed, asset, 80.0, 70.0, scale)

    assert obs.shape == (42, 360, 3) and ref.shape == (42, 360, 4)
    assert obs.dtype == np.uint8 and ref.dtype == np.uint8
    assert np.all(ref[..., :3] == (10, 20, 30))
    assert np.all(ref[..., 3] == 255)
    assert np.all(obs == 200)


def test_reference_fully_transparent_asset_copies_observed() -> None:
    rng = np.random.default_rng(4)
    observed = rng.integers(0, 256, (preprocess.ROI_H, preprocess.ROI_W, 3), dtype=np.uint8)
    asset = _bgra(160, 140, (10, 20, 30), 0)

    obs, ref = preprocess.strips(observed, asset, 80.0, 70.0, 1.0)

    assert np.array_equal(ref[..., :3], obs)
    assert np.all(ref[..., 3] == 0)


def test_reference_composition_uses_exact_alpha_weight() -> None:
    observed = np.full((preprocess.ROI_H, preprocess.ROI_W, 3), 200, dtype=np.uint8)
    asset = _bgra(160, 140, (100, 100, 100), 128)

    _, ref = preprocess.strips(observed, asset, 80.0, 70.0, 1.0)

    # 100*(128/255) + 200*(1 - 128/255) = 149.8039... -> 150
    assert np.all(ref[..., :3] == 150)
    assert np.all(ref[..., 3] == 128)


def test_normalize_asset_pads_three_channel_with_opaque_alpha() -> None:
    rgb = np.full((4, 5, 3), 7, dtype=np.uint8)

    padded = preprocess.normalize_asset(rgb)

    assert padded.shape == (4, 5, 4)
    assert np.all(padded[..., 3] == 255)
    assert np.array_equal(padded[..., :3], rgb)
    assert preprocess.normalize_asset(padded) is padded


def test_prepared_asset_matches_strips_on_the_same_input() -> None:
    """预转资产 API 与逐样本入口逐字节等价（采样语义仍只有一份）。"""
    rng = np.random.default_rng(7)
    observed = rng.integers(0, 256, (preprocess.ROI_H, preprocess.ROI_W, 3), dtype=np.uint8)
    asset = rng.integers(0, 256, (140, 160, 4), dtype=np.uint8)

    prepared = preprocess.prepare_asset(asset)
    obs_a, ref_a = preprocess.strips(observed, asset, 80.0, 70.0, 15.0 / 16.0)
    obs_b, ref_b = preprocess.strips_prepared(observed, prepared, 80.0, 70.0, 15.0 / 16.0)

    assert prepared.dtype == torch.float32
    assert prepared.shape == (1, 4, 140, 160)
    assert np.array_equal(obs_a, obs_b)
    assert np.array_equal(ref_a, ref_b)


def test_prepare_asset_normalizes_three_channel_entry() -> None:
    rgb = np.full((4, 5, 3), 9, dtype=np.uint8)

    prepared = preprocess.prepare_asset(rgb)

    assert prepared.shape == (1, 4, 4, 5)
    assert torch.all(prepared[:, :3] == 9.0)
    assert torch.all(prepared[:, 3] == 255.0)


def test_strips_prepared_rejects_non_prepared_asset() -> None:
    with pytest.raises(ValueError, match="prepared asset"):
        preprocess.strips_prepared(_roi(), torch.zeros(1, 3, 4, 4), 0.0, 0.0, 1.0)


def test_strips_rejects_non_bgra_asset() -> None:
    roi = _roi()
    with pytest.raises(ValueError, match="BGRA"):
        preprocess.strips(roi, np.zeros((10, 10, 3), dtype=np.uint8), 5.0, 5.0, 1.0)


def _ramp_asset(width: int = 200, height: int = 140) -> np.ndarray:
    asset = np.zeros((height, width, 4), dtype=np.uint8)
    ramp = np.arange(width, dtype=np.uint8)
    asset[..., 0] = ramp[None, :]
    asset[..., 3] = 255
    return asset


def test_reference_uses_exact_subpixel_center() -> None:
    observed = np.zeros((preprocess.ROI_H, preprocess.ROI_W, 3), dtype=np.uint8)
    asset = _ramp_asset()
    # 列 0（正北）的 u 恰为极点 u=59；资产坐标 au = x，线性 ramp 的采样值 = au
    _, low = preprocess.strips(observed, asset, 80.3, 70.0, 1.0)
    _, high = preprocess.strips(observed, asset, 80.7, 70.0, 1.0)

    assert low[0, 0, 0] == 80
    assert high[0, 0, 0] == 81


def test_reference_scale_shrinks_asset_coordinates_exactly() -> None:
    observed = np.zeros((preprocess.ROI_H, preprocess.ROI_W, 3), dtype=np.uint8)
    asset = _ramp_asset()
    # 行 28（r = 40.5）的列 90（正东）：u = 99.5；au = 80 + 40.5 * scale
    _, one_to_one = preprocess.strips(observed, asset, 80.0, 70.0, 1.0)
    _, scaled = preprocess.strips(observed, asset, 80.0, 70.0, 15.0 / 16.0)

    assert one_to_one[28, 90, 0] == 120  # 120.5 半偶舍入
    assert scaled[28, 90, 0] == 118  # 117.96875


def test_reference_out_of_bounds_reads_as_missing() -> None:
    rng = np.random.default_rng(5)
    observed = rng.integers(0, 256, (preprocess.ROI_H, preprocess.ROI_W, 3), dtype=np.uint8)
    asset = _bgra(40, 40, (10, 20, 30), 255)

    obs, ref = preprocess.strips(observed, asset, 20.0, 20.0, 1.0)

    alpha = ref[..., 3]
    assert alpha.min() == 0 and np.any(alpha == 255)
    assert np.all(ref[..., :3][alpha == 255] == (10, 20, 30))
    assert np.array_equal(ref[..., :3][alpha == 0], obs[alpha == 0])
