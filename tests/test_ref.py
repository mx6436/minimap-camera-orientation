"""参考输入（ref）编码的回归测试：定义模块适配、7 通道编码与 ref 数据管线。

定义模块（`endfield/preprocess.py`）的语义行为由 `tests/test_preprocess.py` 按
规格断言；本文件验证适配层的机械动作（资产 I/O、拼接、缺口占比）与数据管线，
期望值取自定义模块的公开输出。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pytest

import prepare_data
from endfield import maplocator, preprocess, preprocess_cache
from endfield.locate import accept, load_records, record_scale, write_jsonl, zone_asset_path
from endfield.polar import IMG_H, IMG_W, imread_png, load_source_frame
from endfield.preprocess import observed_roi
from endfield.ref import (
    REF_CHANNELS,
    REF_SUBDIR,
    assemble_ref_pair,
    load_reference_image,
    ref_pair,
    reference_gap_fraction,
)


def test_load_reference_image_reads_bgra_png(tmp_path: Path) -> None:
    asset = np.arange(2 * 3 * 4, dtype=np.uint8).reshape(2, 3, 4)
    path = tmp_path / "asset.png"
    assert cv2.imwrite(str(path), asset)
    assert np.array_equal(load_reference_image(path), asset)


def test_reference_gap_fraction_counts_any_alpha_below_255() -> None:
    reference = np.full((1, 4, 4), 255, dtype=np.uint8)
    reference[..., 3] = [[255, 254, 0, 128]]
    assert reference_gap_fraction(reference) == pytest.approx(3 / 4)


def test_ref_pair_normalizes_three_channel_asset_at_the_entry() -> None:
    observed = np.zeros((preprocess.ROI_H, preprocess.ROI_W, 3), dtype=np.uint8)
    asset = np.full((200, 200, 3), 9, dtype=np.uint8)

    ref = ref_pair(observed, asset, 100.0, 100.0)

    assert ref.shape == (IMG_H, IMG_W, REF_CHANNELS)
    assert np.all(ref[..., 6] == 255)  # 3 通道资产按完全不透明处理


def test_ref_pair_copies_observed_channel_where_reference_is_missing() -> None:
    rng = np.random.default_rng(8)
    asset = rng.integers(0, 256, (200, 200, 4), dtype=np.uint8)
    asset[:, 90:110, 3] = 0
    observed = rng.integers(0, 256, (preprocess.ROI_H, preprocess.ROI_W, 3), dtype=np.uint8)

    ref = ref_pair(observed, asset, 100.0, 100.0)
    alpha = ref[..., 6]

    assert np.any(alpha == 0)
    # 参考缺失处 ref.BGR 逐像素等于 obs.BGR
    assert np.array_equal(ref[..., 3:6][alpha == 0], ref[..., :3][alpha == 0])


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


def write_frame(path: Path, value: int = 200) -> None:
    frame = np.full((200, 200, 3), 7, dtype=np.uint8)
    frame[51:171, 49:167] = value
    assert cv2.imwrite(str(path), frame)


def locate_record(name: str, **overrides: object) -> dict:
    record = {
        "name": name,
        "status": 0,
        "message": "Global Search Success",
        "zone": "Test_Base",
        "x": 100.0,
        "y": 100.0,
        "rot": 0.0,
        "scale": 1.0,
        "locConf": 0.9,
        "isHeld": False,
        "attempts": 1,
    }
    record.update(overrides)
    return record


@dataclass
class RefFixture:
    train_raw: Path
    val_raw: Path
    assets_root: Path
    locate_path: Path
    names: dict[str, str]

    @property
    def samples(self) -> dict[str, Path]:
        return prepare_data.raw_samples(self.train_raw, self.val_raw)


def ref_fixture(tmp_path: Path) -> RefFixture:
    """两侧各一张 accepted 样本（Test_Base，底图与观测一致）+ 三类不可用记录。

    ok/held 在 val_raw，ok2/fail/noasset 在 train_raw；不可用样本用于验证从各自
    一侧剔除并计数。
    """
    train_raw, val_raw = tmp_path / "train_raw", tmp_path / "val_raw"
    assets_root = tmp_path / "assets"
    train_raw.mkdir()
    val_raw.mkdir()
    (assets_root / "Test").mkdir(parents=True)
    names = {
        "ok": "Test_Base_x100.0_y100.0_r0.0.png",
        "ok2": "Test_Base_x100.0_y100.0_r1.0.png",
        "held": "Test_Base_x100.0_y100.0_r2.0.png",
        "fail": "Test_Base_x100.0_y100.0_r3.0.png",
        "noasset": "Nowhere_Base_x100.0_y100.0_r4.0.png",
    }
    val_keys = ("ok", "held")
    for key in ("ok", "ok2", "held", "fail", "noasset"):
        write_frame((val_raw if key in val_keys else train_raw) / names[key])
    # 底图裁剪窗口（中心 (100,100) 的 118x120）内与观测一致，参考 BGR = 观测
    asset = np.full((200, 200, 4), 200, dtype=np.uint8)
    asset[..., 3] = 255
    assert cv2.imwrite(str(assets_root / "Test" / "Base.png"), asset)

    locate_path = tmp_path / "locate.jsonl"
    write_jsonl(
        locate_path,
        [
            locate_record(names["ok"]),
            locate_record(names["ok2"]),
            locate_record(names["held"], isHeld=True),
            locate_record(names["fail"], status=1, message="Global search failed."),
            locate_record(names["noasset"], zone="Nowhere_Base"),
        ],
    )
    return RefFixture(train_raw, val_raw, assets_root, locate_path, names)


def test_generate_processed_ref_writes_both_streams_and_skips_unaccepted(
    tmp_path: Path,
) -> None:
    fx = ref_fixture(tmp_path)
    processed_dir = tmp_path / "processed_ref"

    processed_names, skipped = prepare_data.generate_processed_ref(
        fx.samples, fx.locate_path, fx.assets_root, processed_dir
    )

    assert processed_names == [fx.names["ok"], fx.names["ok2"]]
    assert skipped == {
        fx.names["held"]: "held",
        fx.names["fail"]: "global_search_failed",
        fx.names["noasset"]: "asset_missing",
    }
    observed = imread_png(processed_dir / fx.names["ok"])
    reference = imread_png(processed_dir / REF_SUBDIR / fx.names["ok"])
    assert observed.shape == (IMG_H, IMG_W, 3)
    assert reference.shape == (IMG_H, IMG_W, 4)
    assert observed.dtype == np.uint8 and reference.dtype == np.uint8
    # 底图与观测一致：观测流 = 参考 BGR 流，参考 alpha 全 255（无参考缺失）
    assert np.all(observed == 200)
    assert np.array_equal(observed, reference[..., :3])
    assert np.all(reference[..., 3] == 255)


def test_generate_processed_ref_marks_out_of_bounds_and_copies_observed(
    tmp_path: Path,
) -> None:
    fx = ref_fixture(tmp_path)
    # 底图小于参考窗口：越界处无内容 -> alpha 0，BGR 取观测（即 copy 观测）
    small = np.full((120, 120, 4), 200, dtype=np.uint8)
    small[..., 3] = 255
    assert cv2.imwrite(str(fx.assets_root / "Test" / "Base.png"), small)
    processed_dir = tmp_path / "processed_ref"

    prepare_data.generate_processed_ref(fx.samples, fx.locate_path, fx.assets_root, processed_dir)

    observed = imread_png(processed_dir / fx.names["ok"])
    reference = imread_png(processed_dir / REF_SUBDIR / fx.names["ok"])
    alpha = reference[..., 3]
    assert alpha.min() == 0 and alpha.max() == 255
    assert np.array_equal(reference[..., :3][alpha == 0], observed[alpha == 0])


def test_generate_processed_ref_is_deterministic_on_rerun(tmp_path: Path) -> None:
    fx = ref_fixture(tmp_path)
    processed_dir = tmp_path / "processed_ref"

    def digests() -> dict[str, str]:
        return {
            str(path.relative_to(processed_dir)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(processed_dir.rglob("*.png"))
        }

    prepare_data.generate_processed_ref(fx.samples, fx.locate_path, fx.assets_root, processed_dir)
    first = digests()
    prepare_data.generate_processed_ref(fx.samples, fx.locate_path, fx.assets_root, processed_dir)
    assert first == digests()
    assert set(first) == {
        fx.names["ok"],
        fx.names["ok2"],
        f"{REF_SUBDIR}/{fx.names['ok']}",
        f"{REF_SUBDIR}/{fx.names['ok2']}",
    }


def test_generate_processed_ref_skips_regeneration_on_cache_hit(tmp_path: Path) -> None:
    fx = ref_fixture(tmp_path)
    processed_dir = tmp_path / "processed_ref"
    prepare_data.generate_processed_ref(fx.samples, fx.locate_path, fx.assets_root, processed_dir)
    observed_path = processed_dir / fx.names["ok"]
    reference_path = processed_dir / REF_SUBDIR / fx.names["ok"]
    assert cv2.imwrite(str(observed_path), np.full((IMG_H, IMG_W, 3), 123, np.uint8))
    assert cv2.imwrite(str(reference_path), np.full((IMG_H, IMG_W, 4), 123, np.uint8))

    processed_names, skipped = prepare_data.generate_processed_ref(
        fx.samples, fx.locate_path, fx.assets_root, processed_dir
    )

    assert processed_names == [fx.names["ok"], fx.names["ok2"]]
    assert skipped == {
        fx.names["held"]: "held",
        fx.names["fail"]: "global_search_failed",
        fx.names["noasset"]: "asset_missing",
    }
    assert np.all(imread_png(observed_path) == 123)
    assert np.all(imread_png(reference_path) == 123)
    stamp = json.loads((processed_dir / preprocess_cache.STAMP_NAME).read_text(encoding="utf-8"))
    assert stamp["mode"] == "ref"
    assert stamp["definition_hash"] == preprocess.definition_hash()
    assert stamp["provenance"] == {"assets_root": str(fx.assets_root)}


def test_generate_processed_ref_regenerates_when_assets_root_provenance_differs(
    tmp_path: Path,
) -> None:
    fx = ref_fixture(tmp_path)
    processed_dir = tmp_path / "processed_ref"
    prepare_data.generate_processed_ref(fx.samples, fx.locate_path, fx.assets_root, processed_dir)
    observed_path = processed_dir / fx.names["ok"]
    assert cv2.imwrite(str(observed_path), np.full((IMG_H, IMG_W, 3), 123, np.uint8))
    stamp_path = processed_dir / preprocess_cache.STAMP_NAME
    stamp = json.loads(stamp_path.read_text(encoding="utf-8"))
    stamp["provenance"]["assets_root"] = str(tmp_path / "other_assets")
    stamp_path.write_text(json.dumps(stamp), encoding="utf-8")

    prepare_data.generate_processed_ref(fx.samples, fx.locate_path, fx.assets_root, processed_dir)

    assert np.all(imread_png(observed_path) == 200)
    assert json.loads(stamp_path.read_text(encoding="utf-8"))["provenance"] == {
        "assets_root": str(fx.assets_root)
    }


def test_generate_processed_ref_regenerates_when_locate_fields_change(tmp_path: Path) -> None:
    fx = ref_fixture(tmp_path)
    processed_dir = tmp_path / "processed_ref"
    prepare_data.generate_processed_ref(fx.samples, fx.locate_path, fx.assets_root, processed_dir)
    assert cv2.imwrite(
        str(processed_dir / fx.names["ok"]), np.full((IMG_H, IMG_W, 3), 123, np.uint8)
    )

    # 同名单、不同坐标：定位记录字段属于前处理输入，必须触发失效
    write_jsonl(
        fx.locate_path,
        [locate_record(fx.names["ok"], x=101.0), locate_record(fx.names["ok2"])],
    )
    prepare_data.generate_processed_ref(fx.samples, fx.locate_path, fx.assets_root, processed_dir)

    assert np.all(imread_png(processed_dir / fx.names["ok"]) == 200)


def test_generate_processed_ref_regenerates_when_stamp_definition_differs(tmp_path: Path) -> None:
    fx = ref_fixture(tmp_path)
    processed_dir = tmp_path / "processed_ref"
    prepare_data.generate_processed_ref(fx.samples, fx.locate_path, fx.assets_root, processed_dir)
    assert cv2.imwrite(
        str(processed_dir / fx.names["ok"]), np.full((IMG_H, IMG_W, 3), 123, np.uint8)
    )
    stamp_path = processed_dir / preprocess_cache.STAMP_NAME
    stamp = json.loads(stamp_path.read_text(encoding="utf-8"))
    stamp["definition_hash"] = "0" * 64
    stamp_path.write_text(json.dumps(stamp), encoding="utf-8")

    prepare_data.generate_processed_ref(fx.samples, fx.locate_path, fx.assets_root, processed_dir)

    assert np.all(imread_png(processed_dir / fx.names["ok"]) == 200)
    assert json.loads(stamp_path.read_text(encoding="utf-8"))["definition_hash"] == (
        preprocess.definition_hash()
    )


def test_generate_processed_ref_matches_across_worker_counts(tmp_path: Path) -> None:
    """并行解码只加速 I/O：workers=4 与 workers=1 的两路产物须逐字节一致。"""
    fx = ref_fixture(tmp_path)
    serial_dir, parallel_dir = tmp_path / "serial_ref", tmp_path / "parallel_ref"

    serial = prepare_data.generate_processed_ref(
        fx.samples, fx.locate_path, fx.assets_root, serial_dir, workers=1
    )
    parallel = prepare_data.generate_processed_ref(
        fx.samples, fx.locate_path, fx.assets_root, parallel_dir, workers=4
    )
    assert serial == parallel

    def digests(directory: Path) -> dict[str, str]:
        return {
            str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(directory.rglob("*.png"))
        }

    assert digests(serial_dir) == digests(parallel_dir)


def test_generate_processed_ref_without_usable_assets_returns_empty(tmp_path: Path) -> None:
    """accepted 记录全部缺资产：无产物可生成，不建并行池也不报错。"""
    fx = ref_fixture(tmp_path)
    write_jsonl(fx.locate_path, [locate_record(fx.names["noasset"], zone="Nowhere_Base")])
    processed_dir = tmp_path / "processed_ref"

    processed_names, skipped = prepare_data.generate_processed_ref(
        fx.samples, fx.locate_path, fx.assets_root, processed_dir
    )

    assert processed_names == []
    assert skipped[fx.names["noasset"]] == "asset_missing"


def test_generate_processed_ref_counts_samples_without_locate_record(tmp_path: Path) -> None:
    """目录内无定位记录的样本从产物剔除并计数（#39 重跑定位前的真实状态）。"""
    fx = ref_fixture(tmp_path)
    orphan = "Test_Base_x100.0_y100.0_r9.0.png"
    write_frame(fx.train_raw / orphan)
    processed_dir = tmp_path / "processed_ref"

    processed_names, skipped = prepare_data.generate_processed_ref(
        fx.samples, fx.locate_path, fx.assets_root, processed_dir
    )

    assert orphan not in processed_names
    assert skipped[orphan] == "no_locate_record"


def test_generate_processed_ref_cache_ignores_directory_membership(tmp_path: Path) -> None:
    """输入指纹取两侧并集：样本换侧只换划分，不触发重算。"""
    fx = ref_fixture(tmp_path)
    processed_dir = tmp_path / "processed_ref"
    prepare_data.generate_processed_ref(fx.samples, fx.locate_path, fx.assets_root, processed_dir)
    assert cv2.imwrite(
        str(processed_dir / fx.names["ok"]), np.full((IMG_H, IMG_W, 3), 123, np.uint8)
    )

    (fx.val_raw / fx.names["ok"]).rename(fx.train_raw / fx.names["ok"])
    processed_names, _ = prepare_data.generate_processed_ref(
        fx.samples, fx.locate_path, fx.assets_root, processed_dir
    )

    assert processed_names == [fx.names["ok"], fx.names["ok2"]]
    assert np.all(imread_png(processed_dir / fx.names["ok"]) == 123)


def test_run_ref_force_regenerates(tmp_path: Path) -> None:
    fx = ref_fixture(tmp_path)
    processed_dir, train_dir, val_dir = (
        tmp_path / "processed_ref",
        tmp_path / "train_ref",
        tmp_path / "val_ref",
    )
    prepare_data.run_ref(
        fx.assets_root,
        fx.train_raw,
        fx.val_raw,
        fx.locate_path,
        processed_dir,
        train_dir,
        val_dir,
    )
    assert cv2.imwrite(
        str(processed_dir / fx.names["ok"]), np.full((IMG_H, IMG_W, 3), 123, np.uint8)
    )

    prepare_data.run_ref(
        fx.assets_root,
        fx.train_raw,
        fx.val_raw,
        fx.locate_path,
        processed_dir,
        train_dir,
        val_dir,
        force=True,
    )

    assert np.all(imread_png(processed_dir / fx.names["ok"]) == 200)


def test_run_ref_links_each_directory_to_its_side_and_drops_unusable(tmp_path: Path) -> None:
    fx = ref_fixture(tmp_path)
    processed_dir, train_dir, val_dir = (
        tmp_path / "processed_ref",
        tmp_path / "train_ref",
        tmp_path / "val_ref",
    )

    prepare_data.run_ref(
        fx.assets_root,
        fx.train_raw,
        fx.val_raw,
        fx.locate_path,
        processed_dir,
        train_dir,
        val_dir,
    )

    assert (val_dir / fx.names["ok"]).is_file()
    assert (val_dir / REF_SUBDIR / fx.names["ok"]).is_file()
    assert (train_dir / fx.names["ok2"]).is_file()
    assert (train_dir / REF_SUBDIR / fx.names["ok2"]).is_file()
    for name in (fx.names["held"], fx.names["fail"], fx.names["noasset"]):
        assert not (train_dir / name).exists()
        assert not (val_dir / name).exists()
    assert not (train_dir / fx.names["ok"]).exists()
    assert not (val_dir / fx.names["ok2"]).exists()


def test_run_ref_rejects_side_without_usable_samples(tmp_path: Path) -> None:
    fx = ref_fixture(tmp_path)
    (fx.val_raw / fx.names["ok"]).rename(fx.train_raw / fx.names["ok"])
    processed_dir, train_dir, val_dir = (
        tmp_path / "processed_ref",
        tmp_path / "train_ref",
        tmp_path / "val_ref",
    )

    with pytest.raises(SystemExit, match="val split is empty"):
        prepare_data.run_ref(
            fx.assets_root,
            fx.train_raw,
            fx.val_raw,
            fx.locate_path,
            processed_dir,
            train_dir,
            val_dir,
        )


def test_generate_processed_ref_rejects_records_without_scale(tmp_path: Path) -> None:
    """旧 CLI 产物（无 scale）直接报错。"""
    fx = ref_fixture(tmp_path)
    record = locate_record(fx.names["ok"])
    record.pop("scale")
    write_jsonl(fx.locate_path, [record])
    with pytest.raises(KeyError, match="scale"):
        prepare_data.generate_processed_ref(
            fx.samples, fx.locate_path, fx.assets_root, tmp_path / "processed_ref"
        )


def test_generate_processed_ref_writes_observed_backdrop_reference(tmp_path: Path) -> None:
    fx = ref_fixture(tmp_path)
    # 底图中心开一条 alpha=0 带：参考 BGR 应 copy 观测而非黑底
    asset = np.full((200, 200, 4), 200, dtype=np.uint8)
    asset[..., 3] = 255
    asset[:, 90:110, 3] = 0
    assert cv2.imwrite(str(fx.assets_root / "Test" / "Base.png"), asset)
    processed_dir = tmp_path / "processed_ref"

    processed_names, _ = prepare_data.generate_processed_ref(
        fx.samples, fx.locate_path, fx.assets_root, processed_dir
    )

    assert processed_names == [fx.names["ok"], fx.names["ok2"]]
    observed = imread_png(processed_dir / fx.names["ok"])
    reference = imread_png(processed_dir / REF_SUBDIR / fx.names["ok"])
    alpha = reference[..., 3]
    assert np.any(alpha == 0) and np.any(alpha == 255)
    # 参考缺失处逐像素等于观测
    assert np.array_equal(reference[..., :3][alpha == 0], observed[alpha == 0])
    assert np.any(reference[..., :3][alpha == 0] != 0)


def test_link_split_without_subdirs_links_processed_view(tmp_path: Path) -> None:
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    for name in ("a.png", "b.png"):
        assert cv2.imwrite(str(processed_dir / name), np.zeros((IMG_H, IMG_W, 3), np.uint8))

    prepare_data.link_split(
        ["a.png"], ["b.png"], tmp_path / "train", tmp_path / "val", processed_dir
    )

    assert (tmp_path / "train" / "a.png").is_file()
    assert (tmp_path / "val" / "b.png").is_file()
    assert not (tmp_path / "train" / "b.png").exists()


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
    not REAL_LOCATE_PATH.exists() or not maplocator.assets_root().is_dir(),
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
    zone = str(record["zone"])
    asset_path = zone_asset_path(zone, maplocator.assets_root())
    assert asset_path is not None
    x, y = float(record["x"]), float(record["y"])
    observed = observed_roi(load_source_frame(raw_path))

    ref = ref_pair(observed, load_reference_image(asset_path), x, y, record_scale(record))
    alpha = ref[..., 6]

    assert ref.shape == (IMG_H, IMG_W, REF_CHANNELS)
    assert ref.dtype == np.uint8
    # 参考缺失处 ref.BGR 逐像素等于 obs.BGR（alpha==0 不变量）
    assert np.array_equal(ref[..., 3:6][alpha == 0], ref[..., :3][alpha == 0])
