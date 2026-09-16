"""参考输入的适配层：资产 I/O、条带采样、7 通道编码与参考缺失占比。

定义模块（`endfield/preprocess.py`）的语义行为由 `tests/test_preprocess.py` 按规格断言；
本文件验证适配层的机械动作，期望值取自定义模块的公开输出。数据准备与 ref 输入侧解析
分别在 `tests/test_prepare.py` 与 `tests/test_ref_inputs.py`。
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from endfield import preprocess
from endfield.input_encoding import assemble_ref_pair, reference_gap_fraction
from endfield.polar import load_source_frame
from endfield.preprocess import IMG_H, IMG_W, observed_roi
from endfield.run_record import InputMode, input_channels
from placement import workspace
from placement.placement import Placement, accept
from placement.records import load_records
from placement.sample import ReferenceSampler, load_reference_image

REF_CHANNELS = input_channels(InputMode.REF)


def test_load_reference_image_reads_bgra_png(tmp_path: Path) -> None:
    asset = np.arange(2 * 3 * 4, dtype=np.uint8).reshape(2, 3, 4)
    path = tmp_path / "asset.png"
    assert cv2.imwrite(str(path), asset)
    assert np.array_equal(load_reference_image(path), asset)


def test_reference_gap_fraction_counts_any_alpha_below_255() -> None:
    reference = np.full((1, 4, 4), 255, dtype=np.uint8)
    reference[..., 3] = [[255, 254, 0, 128]]
    assert reference_gap_fraction(reference) == pytest.approx(3 / 4)


def test_sampler_treats_three_channel_asset_as_fully_opaque(tmp_path: Path) -> None:
    """3 通道资产在采样入口按完全不透明处理（补 255 alpha）。"""
    (tmp_path / "Test").mkdir()
    asset = np.full((200, 200, 3), 9, dtype=np.uint8)
    assert cv2.imwrite(str(tmp_path / "Test" / "Base.png"), asset)
    observed = np.zeros((preprocess.ROI_H, preprocess.ROI_W, 3), dtype=np.uint8)

    observed_strip, reference = ReferenceSampler(tmp_path).strips(
        observed, Placement(zone="Test_Base", x=100.0, y=100.0, scale=1.0)
    )

    assert observed_strip.shape == (IMG_H, IMG_W, 3)
    assert reference.shape == (IMG_H, IMG_W, 4)
    assert np.all(reference[..., 3] == 255)


def test_assemble_ref_pair_concatenates_channels_in_spec_order() -> None:
    observed = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
    reference = np.zeros((IMG_H, IMG_W, 4), dtype=np.uint8)
    observed[...] = (1, 2, 3)
    reference[...] = (4, 5, 6, 7)
    ref = assemble_ref_pair(observed, reference)
    assert ref.shape == (IMG_H, IMG_W, REF_CHANNELS)
    assert ref.dtype == np.uint8
    for channel, value in enumerate((1, 2, 3, 4, 5, 6, 7)):
        assert np.all(ref[..., channel] == value)


def test_assemble_ref_pair_rejects_wrong_stream_shapes() -> None:
    observed = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
    reference = np.zeros((IMG_H, IMG_W, 4), dtype=np.uint8)
    with pytest.raises(ValueError, match="observed"):
        assemble_ref_pair(np.zeros((IMG_H, IMG_W, 4), dtype=np.uint8), reference)
    with pytest.raises(ValueError, match="reference"):
        assemble_ref_pair(observed, np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8))


REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_LOCATE_PATH = REPO_ROOT / "data" / "locator" / "locate.jsonl"
REAL_RAW_DIRS = (REPO_ROOT / "data" / "train_raw", REPO_ROOT / "data" / "val_raw")


def real_raw_path(name: str) -> Path | None:
    for directory in REAL_RAW_DIRS:
        path = directory / name
        if path.is_file():
            return path
    return None


@pytest.mark.skipif(
    not REAL_LOCATE_PATH.exists() or not workspace.assets_root().is_dir(),
    reason="real MapLocator locate.jsonl / assets not available",
)
def test_real_accepted_sample_builds_ref_pair() -> None:
    records = load_records(REAL_LOCATE_PATH)
    for name, record in sorted(records.items()):
        raw_path = real_raw_path(name)
        if accept(record)[0] and "_L" in str(record.get("zone", "")) and raw_path is not None:
            break
    else:
        pytest.skip("no accepted real sample with a raw png and _L zone")
    placement = Placement.from_record(record)
    assert placement.asset_path(workspace.assets_root()) is not None
    observed = observed_roi(load_source_frame(raw_path))

    observed_strip, reference_strip = ReferenceSampler(workspace.assets_root()).strips(
        observed, placement
    )
    ref = assemble_ref_pair(observed_strip, reference_strip)
    alpha = ref[..., 6]

    assert ref.shape == (IMG_H, IMG_W, REF_CHANNELS)
    assert ref.dtype == np.uint8
    # 参考缺失处 ref.BGR 为白底（alpha==0 不变量）
    assert np.all(ref[..., 3:6][alpha == 0] == 255)
