"""live.py 实机输入侧的测试：run record 选模式、720p 基准缩放、条带与训练产物一致。

`endfield/live.py` 是实机推理与训练前处理共用的输入侧；本地缺 MapLocator 资产或
真实定位产物时，依赖它们的用例跳过。
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from endfield.live import (
    MissingZoneAsset,
    load_run_config,
    ref_strip_at,
    to_base_frame,
)
from endfield.locate import accept, load_records, write_jsonl, zone_asset_path
from endfield.polar import BASE_SIZE, IMG_H, IMG_W, imread_png, load_source_frame
from endfield.preprocess import observed_roi
from endfield.ref import REF_CHANNELS, ref_strip

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_RAW_DIRS = (REPO_ROOT / "data" / "train_raw", REPO_ROOT / "data" / "val_raw")
REAL_LOCATE_PATH = REPO_ROOT / "data" / "locator" / "locate.jsonl"
REAL_ASSETS_ROOT = REPO_ROOT / "local" / "maplocator" / "resource" / "image" / "MapLocator"


def real_raw_path(name: str) -> Path | None:
    for directory in REAL_RAW_DIRS:
        path = directory / name
        if path.is_file():
            return path
    return None


def write_record(run_dir: Path, record: dict) -> None:
    (run_dir / "record.json").write_text(json.dumps(record), encoding="utf-8")


def test_load_run_config_reads_polar_mode_without_assets(tmp_path: Path) -> None:
    write_record(tmp_path, {"input_mode": "polar"})
    config = load_run_config(tmp_path)
    assert config.input_mode == "polar"
    assert config.assets_root is None


def test_load_run_config_defaults_missing_input_mode_to_polar(tmp_path: Path) -> None:
    # version 27 及以前的 polar run 没有 input_mode 字段
    write_record(tmp_path, {"version": 27})
    config = load_run_config(tmp_path)
    assert config.input_mode == "polar"
    assert config.assets_root is None


def test_load_run_config_resolves_relative_ref_assets_against_repo(tmp_path: Path) -> None:
    write_record(
        tmp_path,
        {
            "input_mode": "ref",
            "ref_reference_assets_root": "local/maplocator/resource/image/MapLocator",
        },
    )
    config = load_run_config(tmp_path)
    assert config.input_mode == "ref"
    assert (
        config.assets_root
        == REPO_ROOT / "local" / "maplocator" / "resource" / "image" / "MapLocator"
    )


def test_load_run_config_keeps_absolute_ref_assets(tmp_path: Path) -> None:
    assets = tmp_path / "assets"
    write_record(tmp_path, {"input_mode": "ref", "ref_reference_assets_root": str(assets)})
    assert load_run_config(tmp_path).assets_root == assets


def test_load_run_config_rejects_ref_without_assets_root(tmp_path: Path) -> None:
    write_record(tmp_path, {"input_mode": "ref"})
    with pytest.raises(ValueError, match="ref_reference_assets_root"):
        load_run_config(tmp_path)


def test_load_run_config_rejects_removed_pair_mode(tmp_path: Path) -> None:
    write_record(
        tmp_path,
        {
            "input_mode": "pair",
            "pair_reference_assets_root": str(tmp_path / "assets"),
            "pair_encoding": 2,
        },
    )
    with pytest.raises(ValueError, match="unsupported input_mode"):
        load_run_config(tmp_path)


def test_load_run_config_rejects_unknown_mode(tmp_path: Path) -> None:
    write_record(tmp_path, {"input_mode": "bogus"})
    with pytest.raises(ValueError, match="input_mode"):
        load_run_config(tmp_path)


def test_load_run_config_rejects_removed_residual_mode(tmp_path: Path) -> None:
    write_record(
        tmp_path,
        {"input_mode": "residual", "residual_reference_assets_root": "local/assets"},
    )
    with pytest.raises(ValueError, match="input_mode"):
        load_run_config(tmp_path)


def test_to_base_frame_scales_non_base_frames_to_720p() -> None:
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    scaled = to_base_frame(frame)
    assert scaled.shape == (BASE_SIZE[1], BASE_SIZE[0], 3)
    assert scaled.dtype == np.uint8


def test_ref_strip_at_rejects_missing_zone_asset(tmp_path: Path) -> None:
    frame = np.zeros((BASE_SIZE[1], BASE_SIZE[0], 3), dtype=np.uint8)
    record = {"zone": "Nowhere_Base", "x": 1.0, "y": 2.0, "scale": 1.0}
    with pytest.raises(MissingZoneAsset, match="Nowhere_Base"):
        ref_strip_at(frame, record, tmp_path)


def test_ref_strip_at_rejects_record_without_xy(tmp_path: Path) -> None:
    assets = tmp_path / "assets"
    (assets / "Test").mkdir(parents=True)
    asset = np.zeros((120, 120, 4), dtype=np.uint8)
    asset[..., 3] = 255
    assert cv2.imwrite(str(assets / "Test" / "Base.png"), asset)
    frame = np.zeros((BASE_SIZE[1], BASE_SIZE[0], 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="lacks x/y"):
        ref_strip_at(frame, {"zone": "Test_Base", "scale": 1.0}, assets)


def test_ref_strip_at_rejects_record_without_scale(tmp_path: Path) -> None:
    """旧 CLI 产物（无 scale）时直接报错。"""
    assets = tmp_path / "assets"
    (assets / "Test").mkdir(parents=True)
    asset = np.zeros((120, 120, 4), dtype=np.uint8)
    asset[..., 3] = 255
    assert cv2.imwrite(str(assets / "Test" / "Base.png"), asset)
    frame = np.zeros((BASE_SIZE[1], BASE_SIZE[0], 3), dtype=np.uint8)
    with pytest.raises(KeyError, match="scale"):
        ref_strip_at(frame, {"zone": "Test_Base", "x": 1.0, "y": 2.0}, assets)


def test_ref_strip_at_reads_raw_bgra_asset_alpha(tmp_path: Path) -> None:
    """ref.A 保留原始 alpha，而非黑底合成后的 255。"""
    assets = tmp_path / "assets"
    (assets / "Test").mkdir(parents=True)
    rng = np.random.default_rng(3)
    asset = rng.integers(0, 256, (240, 240, 4), dtype=np.uint8)
    asset[..., 3] = 7  # 中心 118x120 裁剪窗口内 alpha 恒为 7
    assert cv2.imwrite(str(assets / "Test" / "Base.png"), asset)
    frame = rng.integers(0, 256, (BASE_SIZE[1], BASE_SIZE[0], 3), dtype=np.uint8)
    record = {"zone": "Test_Base", "x": 120.0, "y": 120.0, "scale": 1.0}

    strip = ref_strip_at(frame, record, assets)

    assert strip.shape == (IMG_H, IMG_W, REF_CHANNELS)
    assert strip.dtype == np.uint8
    assert np.all(strip[..., 6] == 7)
    # 观测流 = 训练基准帧的 ROI 展开
    assert np.array_equal(
        strip[..., :3],
        ref_strip(observed_roi(frame), asset, 120.0, 120.0)[..., :3],
    )


def test_ref_strip_at_copies_observed_where_reference_is_missing(tmp_path: Path) -> None:
    assets = tmp_path / "assets"
    (assets / "Test").mkdir(parents=True)
    rng = np.random.default_rng(10)
    asset = rng.integers(0, 256, (240, 240, 4), dtype=np.uint8)
    asset[:, 110:130, 3] = 0  # 中心 (120,120) 裁剪窗口内一条成片透明带
    assert cv2.imwrite(str(assets / "Test" / "Base.png"), asset)
    frame = rng.integers(0, 256, (BASE_SIZE[1], BASE_SIZE[0], 3), dtype=np.uint8)
    record = {"zone": "Test_Base", "x": 120.0, "y": 120.0, "scale": 1.0}

    strip = ref_strip_at(frame, record, assets)
    alpha = strip[..., 6]

    assert np.any(alpha == 0)
    assert np.array_equal(strip[..., 3:6][alpha == 0], strip[..., :3][alpha == 0])


@pytest.mark.skipif(
    not REAL_LOCATE_PATH.exists() or not REAL_ASSETS_ROOT.is_dir(),
    reason="real locate.jsonl / assets not available",
)
def test_live_ref_strip_matches_regenerated_training_artifacts(tmp_path: Path) -> None:
    """同帧同坐标：live 路径与 prepare_data --mode ref 的两路产物逐字节一致（含尺度）。

    用同一批真实样本现场重跑数据管线（不读 data/processed_ref，避免拿旧定义产物对拍）。
    """
    import prepare_data

    records = load_records(REAL_LOCATE_PATH)
    samples = [
        (name, record, real_raw_path(name))
        for name, record in sorted(records.items())
        if accept(record)[0]
        and real_raw_path(name) is not None
        and zone_asset_path(str(record.get("zone", "")), REAL_ASSETS_ROOT) is not None
    ]
    assert samples, "no accepted real sample with a zone asset"
    scaled = [item for item in samples if item[1].get("zone") == "ValleyIV_Base"]
    chosen = (samples[:8] + scaled[:2]) if scaled else samples[:10]

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    for name, _, path in chosen:
        assert path is not None
        (raw_dir / name).symlink_to(path)
    locate_path = tmp_path / "locate.jsonl"
    write_jsonl(locate_path, [{**record, "name": name} for name, record, _ in chosen])
    processed = tmp_path / "processed_ref"
    prepare_data.generate_processed_ref(
        {name: raw_dir / name for name, _, _ in chosen}, locate_path, REAL_ASSETS_ROOT, processed
    )

    for name, record, path in chosen:
        assert path is not None
        frame = load_source_frame(path)
        strip = ref_strip_at(frame, record, REAL_ASSETS_ROOT)
        observed = imread_png(processed / name)
        reference = imread_png(processed / "ref" / name)
        assert strip.shape == (IMG_H, IMG_W, REF_CHANNELS)
        assert np.array_equal(strip[..., :3], observed), f"observed mismatch for {name}"
        assert np.array_equal(strip[..., 3:], reference), f"reference mismatch for {name}"
