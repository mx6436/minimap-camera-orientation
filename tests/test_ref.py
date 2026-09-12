"""参考输入（ref）前处理的回归测试：参考两路裁剪、7 通道拼接与 ref 数据管线。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

import prepare_data
from endfield.locate import accept, load_records, write_jsonl, zone_asset_path
from endfield.polar import IMG_H, IMG_W, INNER_R, OUTER_R, imread_png, load_source_bgr, unwrap
from endfield.ref import (
    MAP_ASSETS_ROOT,
    REF_CHANNELS,
    REF_SUBDIR,
    ROI_POLE,
    compose_observed_backdrop,
    composite_on_black,
    load_reference_image,
    observed_roi,
    ref_strip,
    ref_tensor,
    reference_alpha_plane,
    reference_crop,
    reference_planes,
    reference_strip,
    zone_scale,
)


def test_reference_alpha_plane_reads_fourth_channel() -> None:
    bgra = np.arange(2 * 3 * 4, dtype=np.uint8).reshape(2, 3, 4)
    assert reference_alpha_plane(bgra).tolist() == bgra[..., 3].tolist()


def test_reference_alpha_plane_treats_three_channel_asset_as_opaque() -> None:
    bgr = np.full((2, 3, 3), 9, dtype=np.uint8)
    alpha = reference_alpha_plane(bgr)
    assert alpha.shape == (2, 3)
    assert np.all(alpha == 255)


def test_reference_planes_bgr_matches_black_composite_crop() -> None:
    rng = np.random.default_rng(0)
    asset = rng.integers(0, 256, (200, 200, 4), dtype=np.uint8)
    bgr, alpha = reference_planes(asset, 100.0, 100.0, 1.0)
    assert bgr.tolist() == reference_crop(composite_on_black(asset), 100.0, 100.0, 1.0).tolist()
    assert alpha.tolist() == reference_crop(asset[..., 3], 100.0, 100.0, 1.0).tolist()


def test_reference_planes_alpha_fills_out_of_bounds_with_zero() -> None:
    asset = np.full((120, 120, 4), 255, dtype=np.uint8)
    _, alpha = reference_planes(asset, 20.0, 20.0, 1.0)
    # 裁剪窗口（中心 (20,20) 的 118x120）超出 120x120 资产：外侧无内容 -> alpha 0
    assert alpha.shape == (120, 118)
    assert np.all(alpha[:40] == 0) and np.all(alpha[:, :39] == 0)
    assert np.all(alpha[60:80, 60:100] == 255)


def test_reference_strip_unwraps_composed_bgr_and_alpha_into_bgra() -> None:
    rng = np.random.default_rng(1)
    asset = rng.integers(0, 256, (200, 200, 4), dtype=np.uint8)
    observed = rng.integers(0, 256, (120, 118, 3), dtype=np.uint8)
    strip = reference_strip(observed, asset, 100.0, 100.0)
    assert strip.shape == (IMG_H, IMG_W, 4)
    assert strip.dtype == np.uint8
    black_ref, alpha = reference_planes(asset, 100.0, 100.0)
    composed = compose_observed_backdrop(black_ref, alpha, observed)
    assert np.array_equal(strip[..., :3], unwrap(composed, *ROI_POLE, INNER_R, OUTER_R))
    assert np.array_equal(strip[..., 3], unwrap(alpha, *ROI_POLE, INNER_R, OUTER_R))


def test_ref_tensor_concatenates_channels_in_spec_order() -> None:
    observed = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
    reference = np.zeros((IMG_H, IMG_W, 4), dtype=np.uint8)
    observed[...] = (1, 2, 3)
    reference[...] = (4, 5, 6, 7)
    ref = ref_tensor(observed, reference)
    assert ref.shape == (IMG_H, IMG_W, REF_CHANNELS)
    assert ref.dtype == np.uint8
    for channel, value in enumerate((1, 2, 3, 4, 5, 6, 7)):
        assert np.all(ref[..., channel] == value)


def test_ref_tensor_rejects_wrong_stream_shapes() -> None:
    observed = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
    reference = np.zeros((IMG_H, IMG_W, 4), dtype=np.uint8)
    with pytest.raises(ValueError, match="observed"):
        ref_tensor(np.zeros((IMG_H, IMG_W, 4), dtype=np.uint8), reference)
    with pytest.raises(ValueError, match="reference"):
        ref_tensor(observed, np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8))


def test_compose_observed_backdrop_blends_and_saturates() -> None:
    black = np.array([[[10, 20, 30]]], dtype=np.uint8)
    alpha = np.zeros((1, 1), dtype=np.uint8)
    observed = np.array([[[200, 100, 50]]], dtype=np.uint8)
    # α=0：ref.BGR = black_ref + obs（越界/成片缺失处 black_ref 为 0，即 copy 观测）
    assert compose_observed_backdrop(black, alpha, observed).tolist() == [[[210, 120, 80]]]
    # 合成值超 255 必须饱和，不能以 uint8 回绕（455 回绕会变成 199）
    saturating = compose_observed_backdrop(
        np.full((1, 1, 3), 255, dtype=np.uint8),
        np.zeros((1, 1), dtype=np.uint8),
        np.full((1, 1, 3), 200, dtype=np.uint8),
    )
    assert saturating.tolist() == [[[255, 255, 255]]]


def test_compose_observed_backdrop_copies_observed_at_alpha_zero() -> None:
    rng = np.random.default_rng(7)
    asset = rng.integers(0, 256, (200, 200, 4), dtype=np.uint8)
    asset[:, 90:110, 3] = 0  # (100,100) 裁剪窗口内一条成片透明带
    observed = rng.integers(0, 256, (120, 118, 3), dtype=np.uint8)

    black_ref, alpha = reference_planes(asset, 100.0, 100.0, 1.0)
    composed = compose_observed_backdrop(black_ref, alpha, observed)

    assert np.any(alpha == 0) and np.any(alpha == 255)
    # α==0 处逐像素等于观测（观测背底合成）
    assert np.array_equal(composed[alpha == 0], observed[alpha == 0])
    # α==255 处逐像素与黑底合成一致
    assert np.array_equal(composed[alpha == 255], black_ref[alpha == 255])


def test_ref_strip_copies_observed_channel_where_reference_is_missing() -> None:
    rng = np.random.default_rng(8)
    asset = rng.integers(0, 256, (200, 200, 4), dtype=np.uint8)
    asset[:, 90:110, 3] = 0
    observed = rng.integers(0, 256, (120, 118, 3), dtype=np.uint8)

    ref = ref_strip(observed, asset, 100.0, 100.0)
    alpha = ref[..., 6]

    assert ref.shape == (IMG_H, IMG_W, REF_CHANNELS)
    assert np.any(alpha == 0)
    # 展开后仍逐像素成立：参考缺失处 ref.BGR == obs.BGR
    assert np.array_equal(ref[..., 3:6][alpha == 0], ref[..., :3][alpha == 0])


def test_ref_strip_matches_black_composite_where_asset_is_opaque() -> None:
    rng = np.random.default_rng(9)
    asset = rng.integers(0, 256, (200, 200, 4), dtype=np.uint8)
    asset[..., 3] = 255
    observed = rng.integers(0, 256, (120, 118, 3), dtype=np.uint8)

    ref = ref_strip(observed, asset, 100.0, 100.0)
    black_ref, _ = reference_planes(asset, 100.0, 100.0)

    assert np.array_equal(ref[..., 3:6], unwrap(black_ref, *ROI_POLE, INNER_R, OUTER_R))


@pytest.mark.parametrize("scale", [1.0, 15.0 / 16.0])
def test_ref_strip_matches_reference_crop_on_same_frame(tmp_path: Path, scale: float) -> None:
    rng = np.random.default_rng(2)
    asset = rng.integers(0, 256, (200, 200, 4), dtype=np.uint8)
    asset_path = tmp_path / "Base.png"
    assert cv2.imwrite(str(asset_path), asset)
    observed = rng.integers(0, 256, (120, 118, 3), dtype=np.uint8)

    ref = ref_strip(observed, load_reference_image(asset_path), 100.0, 100.0, scale)
    assert ref.shape == (IMG_H, IMG_W, REF_CHANNELS)
    assert ref.dtype == np.uint8
    # 观测流 = polar 展开
    assert np.array_equal(ref[..., :3], unwrap(observed, *ROI_POLE, INNER_R, OUTER_R))
    # 参考 BGR 流 = 黑底合成裁剪 + 观测背底合成后展开
    black_ref, alpha = reference_planes(asset, 100.0, 100.0, scale)
    composed = compose_observed_backdrop(black_ref, alpha, observed)
    assert np.array_equal(ref[..., 3:6], unwrap(composed, *ROI_POLE, INNER_R, OUTER_R))
    # 参考 alpha 流 = 原始 alpha 的同几何展开
    assert np.array_equal(ref[..., 6], unwrap(alpha, *ROI_POLE, INNER_R, OUTER_R))


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
        "locConf": 0.9,
        "isHeld": False,
        "attempts": 1,
    }
    record.update(overrides)
    return record


def ref_fixture(tmp_path: Path) -> tuple[Path, Path, Path, dict[str, str]]:
    """两张 accepted 样本（Test_Base，底图与观测一致）+ 三类不可用记录。"""
    raw_dir, assets_root = tmp_path / "raw", tmp_path / "assets"
    raw_dir.mkdir()
    (assets_root / "Test").mkdir(parents=True)
    names = {
        "ok": "Test_Base_x100.0_y100.0_r0.0.png",
        "ok2": "Test_Base_x100.0_y100.0_r1.0.png",
        "held": "Test_Base_x100.0_y100.0_r2.0.png",
        "fail": "Test_Base_x100.0_y100.0_r3.0.png",
        "noasset": "Nowhere_Base_x100.0_y100.0_r4.0.png",
    }
    for key in ("ok", "ok2", "held", "fail"):
        write_frame(raw_dir / names[key])
    # 底图裁剪窗口（中心 (100,100) 的 118x120）内与观测一致，黑底合成后参考 BGR = 观测
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
    return raw_dir, assets_root, locate_path, names


def write_manifest(path: Path, files: list[str]) -> None:
    path.write_text(json.dumps({"version": 1, "files": files}), encoding="utf-8")


def test_generate_processed_ref_writes_both_streams_and_skips_unaccepted(
    tmp_path: Path,
) -> None:
    raw_dir, assets_root, locate_path, names = ref_fixture(tmp_path)
    processed_dir = tmp_path / "processed_ref"

    processed_names, skipped = prepare_data.generate_processed_ref(
        raw_dir, locate_path, assets_root, processed_dir
    )

    assert processed_names == [names["ok"], names["ok2"]]
    assert skipped == {
        names["held"]: "held",
        names["fail"]: "global_search_failed",
        names["noasset"]: "asset_missing",
    }
    observed = imread_png(processed_dir / names["ok"])
    reference = imread_png(processed_dir / REF_SUBDIR / names["ok"])
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
    raw_dir, assets_root, locate_path, names = ref_fixture(tmp_path)
    # 底图小于参考窗口：越界处无内容 -> alpha 0，BGR 取观测背底（即 copy 观测）
    small = np.full((120, 120, 4), 200, dtype=np.uint8)
    small[..., 3] = 255
    assert cv2.imwrite(str(assets_root / "Test" / "Base.png"), small)
    processed_dir = tmp_path / "processed_ref"

    prepare_data.generate_processed_ref(raw_dir, locate_path, assets_root, processed_dir)

    observed = imread_png(processed_dir / names["ok"])
    reference = imread_png(processed_dir / REF_SUBDIR / names["ok"])
    alpha = reference[..., 3]
    assert alpha.min() == 0 and alpha.max() == 255
    assert np.array_equal(reference[..., :3][alpha == 0], observed[alpha == 0])


def test_generate_processed_ref_is_deterministic_on_rerun(tmp_path: Path) -> None:
    raw_dir, assets_root, locate_path, names = ref_fixture(tmp_path)
    processed_dir = tmp_path / "processed_ref"

    def digests() -> dict[str, str]:
        return {
            str(path.relative_to(processed_dir)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(processed_dir.rglob("*.png"))
        }

    prepare_data.generate_processed_ref(raw_dir, locate_path, assets_root, processed_dir)
    first = digests()
    prepare_data.generate_processed_ref(raw_dir, locate_path, assets_root, processed_dir)
    assert first == digests()
    assert set(first) == {names["ok"], names["ok2"], f"ref/{names['ok']}", f"ref/{names['ok2']}"}


def test_run_ref_splits_both_streams_and_skips_manifest_entries_without_output(
    tmp_path: Path,
) -> None:
    raw_dir, assets_root, locate_path, names = ref_fixture(tmp_path)
    manifest = tmp_path / "val_manifest.json"
    write_manifest(manifest, [names["ok"], names["held"]])
    processed_dir, train_dir, val_dir = (
        tmp_path / "processed_ref",
        tmp_path / "train_ref",
        tmp_path / "val_ref",
    )

    prepare_data.run_ref(
        assets_root, raw_dir, locate_path, manifest, processed_dir, train_dir, val_dir
    )

    assert (val_dir / names["ok"]).is_file()
    assert (val_dir / REF_SUBDIR / names["ok"]).is_file()
    assert (train_dir / names["ok2"]).is_file()
    assert (train_dir / REF_SUBDIR / names["ok2"]).is_file()
    assert not (train_dir / names["ok"]).exists()
    assert not (val_dir / names["held"]).exists()
    assert not (train_dir / names["held"]).exists()


def test_generate_processed_ref_writes_observed_backdrop_reference(tmp_path: Path) -> None:
    raw_dir, assets_root, locate_path, names = ref_fixture(tmp_path)
    # 底图中心开一条 alpha=0 带：参考 BGR 应 copy 观测而非黑底
    asset = np.full((200, 200, 4), 200, dtype=np.uint8)
    asset[..., 3] = 255
    asset[:, 90:110, 3] = 0
    assert cv2.imwrite(str(assets_root / "Test" / "Base.png"), asset)
    processed_dir = tmp_path / "processed_ref"

    processed_names, _ = prepare_data.generate_processed_ref(
        raw_dir, locate_path, assets_root, processed_dir
    )

    assert processed_names == [names["ok"], names["ok2"]]
    observed = imread_png(processed_dir / names["ok"])
    reference = imread_png(processed_dir / REF_SUBDIR / names["ok"])
    alpha = reference[..., 3]
    assert np.any(alpha == 0) and np.any(alpha == 255)
    # 参考缺失处逐像素等于观测（黑底合成在此处为 0）
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
REAL_RAW_DIR = REPO_ROOT / "data" / "raw"


@pytest.mark.skipif(
    not REAL_LOCATE_PATH.exists() or not MAP_ASSETS_ROOT.is_dir(),
    reason="real MapLocator locate.jsonl / assets not available",
)
def test_real_accepted_sample_builds_ref_strip() -> None:
    records = load_records(REAL_LOCATE_PATH)
    name, record = next(
        (name, record)
        for name, record in sorted(records.items())
        if accept(record)[0] and record.get("zone", "").find("_L") != -1
    )
    zone = str(record["zone"])
    asset_path = zone_asset_path(zone, MAP_ASSETS_ROOT)
    assert asset_path is not None
    x, y = float(record["x"]), float(record["y"])
    observed = observed_roi(load_source_bgr(REAL_RAW_DIR / name))

    ref = ref_strip(observed, load_reference_image(asset_path), x, y, zone_scale(zone))
    assert ref.shape == (IMG_H, IMG_W, REF_CHANNELS)
    assert ref.dtype == np.uint8
    # 参考 BGR 流 = 黑底裁剪 + 观测背底合成；alpha 流与 BGR 同几何
    black_ref, alpha = reference_planes(load_reference_image(asset_path), x, y, zone_scale(zone))
    composed = compose_observed_backdrop(black_ref, alpha, observed)
    assert np.array_equal(ref[..., 3:6], unwrap(composed, *ROI_POLE, INNER_R, OUTER_R))
    assert np.array_equal(ref[..., 6], unwrap(alpha, *ROI_POLE, INNER_R, OUTER_R))
