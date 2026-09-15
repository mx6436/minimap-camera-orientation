"""参考条带采样的测试：条带对产出、底图资产复用、缺资产与参考缺失的域语义。

`ReferenceSampler.strips` 是实机与训练数据生成共用的唯一采样入口；末尾一条端到端
用例验证它与数据管线的落盘产物逐字节同源（需真实定位产物与资产，缺件时跳过）。
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from endfield.input_encoding import assemble_ref_pair
from endfield.polar import BASE_SIZE, load_source_frame
from endfield.preprocess import IMG_H, IMG_W, observed_roi
from placement import sample as sample_module
from placement import workspace
from placement.placement import Placement, accept
from placement.records import load_records, write_jsonl
from placement.sample import MissingZoneAsset, ReferenceSampler

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


def write_asset(assets_root: Path, region: str, asset: np.ndarray) -> None:
    (assets_root / region).mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(assets_root / region / "Base.png"), asset)


def base_frame(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (BASE_SIZE[1], BASE_SIZE[0], 3), dtype=np.uint8)


def test_strips_produce_observed_and_reference_bands(tmp_path: Path) -> None:
    """条带对形状/通道正确，ref.A 保留资产原始 alpha（不是黑底合成后的 255）。"""
    asset = np.zeros((240, 240, 4), dtype=np.uint8)
    asset[..., :3] = 50
    asset[..., 3] = 7
    write_asset(tmp_path, "Test", asset)
    placement = Placement(zone="Test_Base", x=120.0, y=120.0, scale=1.0)

    observed, reference = ReferenceSampler(tmp_path).strips(observed_roi(base_frame(3)), placement)

    assert observed.shape == (IMG_H, IMG_W, 3)
    assert observed.dtype == np.uint8
    assert reference.shape == (IMG_H, IMG_W, 4)
    assert reference.dtype == np.uint8
    assert np.all(reference[..., 3] == 7)
    # 观测流与参考流的 BGR 都来自同一次采样：配对张量前三通道逐像素等于观测条带
    assert np.array_equal(assemble_ref_pair(observed, reference)[..., :3], observed)


def test_reference_gap_falls_back_to_observation(tmp_path: Path) -> None:
    """参考缺失处（ref.A == 0）的 ref.BGR 逐像素等于观测像素。"""
    rng = np.random.default_rng(10)
    asset = rng.integers(0, 256, (240, 240, 4), dtype=np.uint8)
    asset[:, 110:130, 3] = 0  # 中心 (120,120) 裁剪窗口内一条成片透明带
    write_asset(tmp_path, "Test", asset)
    placement = Placement(zone="Test_Base", x=120.0, y=120.0, scale=1.0)

    observed, reference = ReferenceSampler(tmp_path).strips(observed_roi(base_frame(10)), placement)

    alpha = reference[..., 3]
    assert np.any(alpha == 0)
    assert np.array_equal(reference[..., :3][alpha == 0], observed[alpha == 0])


def test_asset_is_read_once_per_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """同一底图跨样本复用：资产只读一次、只转一次 float32。"""
    asset = np.zeros((240, 240, 4), dtype=np.uint8)
    asset[..., 3] = 255
    write_asset(tmp_path, "Test", asset)
    reads: list[Path] = []
    load = sample_module.load_reference_image

    def counting(path: Path) -> np.ndarray:
        reads.append(path)
        return load(path)

    monkeypatch.setattr(sample_module, "load_reference_image", counting)
    sampler = ReferenceSampler(tmp_path)
    placements = [
        Placement(zone="Test_Base", x=120.0, y=120.0, scale=scale) for scale in (1.0, 1.0, 2.0)
    ]

    for index, placement in enumerate(placements):
        sampler.strips(observed_roi(base_frame(index)), placement)

    assert reads == [tmp_path / "Test" / "Base.png"]


def test_missing_zone_asset_raises(tmp_path: Path) -> None:
    sampler = ReferenceSampler(tmp_path)
    with pytest.raises(MissingZoneAsset, match="Nowhere_Base"):
        sampler.strips(
            observed_roi(base_frame(1)), Placement(zone="Nowhere_Base", x=1.0, y=2.0, scale=1.0)
        )
    # 定位失败记录的空 zone 同样拿不到资产
    with pytest.raises(MissingZoneAsset):
        sampler.strips(observed_roi(base_frame(1)), Placement(zone="", x=0.0, y=0.0, scale=1.0))


def test_strips_reject_non_roi_observation(tmp_path: Path) -> None:
    """观测输入的形状契约由定义模块把关，采样器不另立一份校验。"""
    asset = np.zeros((240, 240, 4), dtype=np.uint8)
    asset[..., 3] = 255
    write_asset(tmp_path, "Test", asset)
    with pytest.raises(ValueError, match="minimap ROI"):
        ReferenceSampler(tmp_path).strips(
            np.zeros((10, 10, 3), dtype=np.uint8),
            Placement(zone="Test_Base", x=120.0, y=120.0, scale=1.0),
        )


@pytest.mark.skipif(
    not REAL_LOCATE_PATH.exists() or not REAL_ASSETS_ROOT.is_dir(),
    reason="real locate.jsonl / assets not available",
)
def test_strips_match_regenerated_training_artifacts(tmp_path: Path) -> None:
    """同帧同定位：采样入口的条带对与 prepare-data --mode ref 的两路产物逐字节一致。

    用同一批真实样本现场重跑数据准备（不读 data/processed_ref，避免拿旧定义产物对拍）。
    """
    from endfield import dataset, prepare
    from endfield.run_record import InputMode
    from placement import ref_inputs

    records = load_records(REAL_LOCATE_PATH)
    samples = [
        (name, record, real_raw_path(name))
        for name, record in sorted(records.items())
        if accept(record)[0]
        and real_raw_path(name) is not None
        and Placement.from_record(record).asset_path(REAL_ASSETS_ROOT) is not None
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
    ref = ref_inputs.resolve(
        {name: raw_dir / name for name, _, _ in chosen},
        locate_path=locate_path,
        assets_root=REAL_ASSETS_ROOT,
        zmdmap_root=workspace.zmdmap_root(),
    )
    names = [name for name, _ in ref.inputs.sources]
    assert len(names) > 1, "need at least one sample on each side of the split view"
    prepare.prepare(
        InputMode.REF,
        ref.inputs,
        ref.renderer(ReferenceSampler(ref.assets_root)),
        layout=dataset.DatasetLayout(
            processed_dir=processed,
            train_dir=tmp_path / "train_ref",
            val_dir=tmp_path / "val_ref",
            subdirs=(dataset.REF_SUBDIR,),
        ),
        train_side=names[:-1],
        val_side=names[-1:],
        workers=1,
    )

    sampler = ReferenceSampler(REAL_ASSETS_ROOT)
    for name, record, path in chosen:
        assert path is not None
        frame = load_source_frame(path)
        observed_strip, reference_strip = sampler.strips(
            observed_roi(frame), Placement.from_record(record)
        )
        strip = assemble_ref_pair(observed_strip, reference_strip)
        observed = cv2.imread(str(processed / name), cv2.IMREAD_UNCHANGED)
        reference = cv2.imread(str(processed / "ref" / name), cv2.IMREAD_UNCHANGED)
        assert strip.shape == (IMG_H, IMG_W, 7)
        assert np.array_equal(strip[..., :3], observed), f"observed mismatch for {name}"
        assert np.array_equal(strip[..., 3:], reference), f"reference mismatch for {name}"
