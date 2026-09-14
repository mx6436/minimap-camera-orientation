"""polar 数据管线：目录划分输入（train_raw / val_raw）、落盘与缓存戳命中/失效。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

import prepare_data
from endfield import preprocess, preprocess_cache
from endfield.data_utils import png_names
from endfield.polar import imread_png

TRAIN_RAWS = ("a_r0.png", "b_r90.png")
VAL_RAWS = ("c_r180.png",)
FRAME_VALUE = 7
SENTINEL = 123


def write_raw(directory: Path, names: tuple[str, ...], value: int = FRAME_VALUE) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    frame = np.full((200, 200, 3), value, dtype=np.uint8)
    for name in names:
        assert cv2.imwrite(str(directory / name), frame)


def write_sentinel(path: Path) -> None:
    """把一个产物改成合法但值不同的 PNG：缓存命中时它必须原样留下。"""
    sentinel = np.full((preprocess.IMG_H, preprocess.IMG_W, 3), SENTINEL, dtype=np.uint8)
    assert cv2.imwrite(str(path), sentinel)


def test_raw_samples_merges_both_directories(tmp_path: Path) -> None:
    train_raw, val_raw = tmp_path / "train_raw", tmp_path / "val_raw"
    write_raw(train_raw, TRAIN_RAWS)
    write_raw(val_raw, VAL_RAWS)

    samples = prepare_data.raw_samples(train_raw, val_raw)

    assert samples == {
        name: directory / name
        for directory, names in ((train_raw, TRAIN_RAWS), (val_raw, VAL_RAWS))
        for name in names
    }


def test_raw_samples_rejects_name_present_in_both_directories(tmp_path: Path) -> None:
    """跨侧同名会在 train/val 之间泄漏，必须硬报错而不是静默合并。"""
    train_raw, val_raw = tmp_path / "train_raw", tmp_path / "val_raw"
    write_raw(train_raw, ("a_r0.png",))
    write_raw(val_raw, ("a_r0.png",))

    with pytest.raises(SystemExit, match="both"):
        prepare_data.raw_samples(train_raw, val_raw)


def test_directory_split_returns_each_side_sorted() -> None:
    assert prepare_data.directory_split(["b", "a"], ["d", "c"]) == (["a", "b"], ["c", "d"])


@pytest.mark.parametrize(
    ("train_names", "val_names", "match"),
    [([], ["v"], "train split is empty"), (["t"], [], "val split is empty")],
)
def test_directory_split_rejects_empty_side(
    train_names: list[str], val_names: list[str], match: str
) -> None:
    with pytest.raises(SystemExit, match=match):
        prepare_data.directory_split(train_names, val_names)


def test_generate_processed_writes_pngs_and_stamp(tmp_path: Path) -> None:
    train_raw, val_raw = tmp_path / "train_raw", tmp_path / "val_raw"
    processed_dir = tmp_path / "processed"
    write_raw(train_raw, TRAIN_RAWS)
    write_raw(val_raw, VAL_RAWS)

    names = prepare_data.generate_processed(
        prepare_data.raw_samples(train_raw, val_raw), processed_dir
    )

    assert names == [*TRAIN_RAWS, *VAL_RAWS]
    assert png_names(processed_dir) == [*TRAIN_RAWS, *VAL_RAWS]
    assert np.all(imread_png(processed_dir / TRAIN_RAWS[0]) == FRAME_VALUE)
    stamp = json.loads((processed_dir / preprocess_cache.STAMP_NAME).read_text(encoding="utf-8"))
    assert stamp["mode"] == "polar"
    assert stamp["definition_hash"] == preprocess.definition_hash()
    assert stamp["input_count"] == len(TRAIN_RAWS) + len(VAL_RAWS)


def test_generate_processed_skips_regeneration_on_cache_hit(tmp_path: Path) -> None:
    train_raw, val_raw = tmp_path / "train_raw", tmp_path / "val_raw"
    processed_dir = tmp_path / "processed"
    write_raw(train_raw, TRAIN_RAWS)
    write_raw(val_raw, VAL_RAWS)
    samples = prepare_data.raw_samples(train_raw, val_raw)
    prepare_data.generate_processed(samples, processed_dir)
    write_sentinel(processed_dir / TRAIN_RAWS[0])

    names = prepare_data.generate_processed(samples, processed_dir)

    assert names == [*TRAIN_RAWS, *VAL_RAWS]
    assert np.all(imread_png(processed_dir / TRAIN_RAWS[0]) == SENTINEL)


def test_generate_processed_regenerates_when_stamp_definition_differs(tmp_path: Path) -> None:
    train_raw, val_raw = tmp_path / "train_raw", tmp_path / "val_raw"
    processed_dir = tmp_path / "processed"
    write_raw(train_raw, TRAIN_RAWS)
    write_raw(val_raw, VAL_RAWS)
    samples = prepare_data.raw_samples(train_raw, val_raw)
    prepare_data.generate_processed(samples, processed_dir)
    write_sentinel(processed_dir / TRAIN_RAWS[0])
    stamp_path = processed_dir / preprocess_cache.STAMP_NAME
    stamp = json.loads(stamp_path.read_text(encoding="utf-8"))
    stamp["definition_hash"] = "0" * 64
    stamp_path.write_text(json.dumps(stamp), encoding="utf-8")

    prepare_data.generate_processed(samples, processed_dir)

    assert np.all(imread_png(processed_dir / TRAIN_RAWS[0]) == FRAME_VALUE)
    rewritten = json.loads(stamp_path.read_text(encoding="utf-8"))
    assert rewritten["definition_hash"] == preprocess.definition_hash()


def test_generate_processed_regenerates_when_raw_set_changes(tmp_path: Path) -> None:
    train_raw, val_raw = tmp_path / "train_raw", tmp_path / "val_raw"
    processed_dir = tmp_path / "processed"
    write_raw(train_raw, TRAIN_RAWS)
    write_raw(val_raw, VAL_RAWS)
    prepare_data.generate_processed(prepare_data.raw_samples(train_raw, val_raw), processed_dir)
    write_sentinel(processed_dir / TRAIN_RAWS[0])

    write_raw(val_raw, ("d_r270.png",), value=9)
    names = prepare_data.generate_processed(
        prepare_data.raw_samples(train_raw, val_raw), processed_dir
    )

    assert names == [*TRAIN_RAWS, *VAL_RAWS, "d_r270.png"]
    assert np.all(imread_png(processed_dir / TRAIN_RAWS[0]) == FRAME_VALUE)
    assert np.all(imread_png(processed_dir / "d_r270.png") == 9)


def test_generate_processed_regenerates_when_output_missing(tmp_path: Path) -> None:
    train_raw, val_raw = tmp_path / "train_raw", tmp_path / "val_raw"
    processed_dir = tmp_path / "processed"
    write_raw(train_raw, TRAIN_RAWS)
    write_raw(val_raw, VAL_RAWS)
    samples = prepare_data.raw_samples(train_raw, val_raw)
    prepare_data.generate_processed(samples, processed_dir)
    (processed_dir / TRAIN_RAWS[0]).unlink()

    names = prepare_data.generate_processed(samples, processed_dir)

    assert names == [*TRAIN_RAWS, *VAL_RAWS]
    assert np.all(imread_png(processed_dir / TRAIN_RAWS[0]) == FRAME_VALUE)


def test_generate_processed_force_ignores_cache_hit(tmp_path: Path) -> None:
    train_raw, val_raw = tmp_path / "train_raw", tmp_path / "val_raw"
    processed_dir = tmp_path / "processed"
    write_raw(train_raw, TRAIN_RAWS)
    write_raw(val_raw, VAL_RAWS)
    samples = prepare_data.raw_samples(train_raw, val_raw)
    prepare_data.generate_processed(samples, processed_dir)
    write_sentinel(processed_dir / TRAIN_RAWS[0])

    prepare_data.generate_processed(samples, processed_dir, force=True)

    assert np.all(imread_png(processed_dir / TRAIN_RAWS[0]) == FRAME_VALUE)


def test_generate_processed_failure_does_not_leave_cache_stamp(tmp_path: Path) -> None:
    """强制重生成中途失败时旧戳必须清掉：半成品目录不得被下次运行命中。"""
    train_raw, val_raw = tmp_path / "train_raw", tmp_path / "val_raw"
    processed_dir = tmp_path / "processed"
    write_raw(train_raw, TRAIN_RAWS)
    write_raw(val_raw, VAL_RAWS)
    samples = prepare_data.raw_samples(train_raw, val_raw)
    prepare_data.generate_processed(samples, processed_dir)
    (train_raw / TRAIN_RAWS[0]).write_bytes(b"not a png")

    with pytest.raises(ValueError, match="PNG"):
        prepare_data.generate_processed(samples, processed_dir, force=True)

    assert not (processed_dir / preprocess_cache.STAMP_NAME).exists()
    write_raw(train_raw, (TRAIN_RAWS[0],), value=11)
    prepare_data.generate_processed(prepare_data.raw_samples(train_raw, val_raw), processed_dir)
    assert np.all(imread_png(processed_dir / TRAIN_RAWS[0]) == 11)


def test_generate_processed_matches_across_worker_counts(tmp_path: Path) -> None:
    """并行解码只加速 I/O：workers=4 与 workers=1 的产物须逐字节一致。"""
    train_raw, val_raw = tmp_path / "train_raw", tmp_path / "val_raw"
    write_raw(train_raw, TRAIN_RAWS)
    write_raw(val_raw, VAL_RAWS)
    samples = prepare_data.raw_samples(train_raw, val_raw)
    serial_dir, parallel_dir = tmp_path / "serial", tmp_path / "parallel"

    assert prepare_data.generate_processed(samples, serial_dir, workers=1) == (
        prepare_data.generate_processed(samples, parallel_dir, workers=4)
    )

    def digests(directory: Path) -> dict[str, str]:
        return {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(directory.glob("*.png"))
        }

    assert digests(serial_dir) == digests(parallel_dir)


def test_run_polar_links_each_directory_to_its_side(tmp_path: Path) -> None:
    """目录即划分：train_raw 全量进 train 视图，val_raw 全量进 val 视图。"""
    train_raw, val_raw = tmp_path / "train_raw", tmp_path / "val_raw"
    processed_dir, train_dir, val_dir = tmp_path / "processed", tmp_path / "train", tmp_path / "val"
    write_raw(train_raw, TRAIN_RAWS)
    write_raw(val_raw, VAL_RAWS)

    prepare_data.run_polar(train_raw, val_raw, processed_dir, train_dir, val_dir)

    assert png_names(train_dir) == sorted(TRAIN_RAWS)
    assert png_names(val_dir) == sorted(VAL_RAWS)
    assert png_names(processed_dir) == sorted([*TRAIN_RAWS, *VAL_RAWS])


def test_run_polar_keeps_cache_when_sample_moves_between_directories(tmp_path: Path) -> None:
    """输入指纹取两侧并集：样本换侧只换视图，不重算 processed。"""
    train_raw, val_raw = tmp_path / "train_raw", tmp_path / "val_raw"
    processed_dir, train_dir, val_dir = tmp_path / "processed", tmp_path / "train", tmp_path / "val"
    write_raw(train_raw, TRAIN_RAWS)
    write_raw(val_raw, VAL_RAWS)
    prepare_data.run_polar(train_raw, val_raw, processed_dir, train_dir, val_dir)
    write_sentinel(processed_dir / "b_r90.png")

    (train_raw / "b_r90.png").rename(val_raw / "b_r90.png")
    prepare_data.run_polar(train_raw, val_raw, processed_dir, train_dir, val_dir)

    assert np.all(imread_png(processed_dir / "b_r90.png") == SENTINEL)
    assert png_names(train_dir) == ["a_r0.png"]
    assert png_names(val_dir) == ["b_r90.png", *VAL_RAWS]


def test_run_polar_rejects_empty_side(tmp_path: Path) -> None:
    train_raw, val_raw = tmp_path / "train_raw", tmp_path / "val_raw"
    processed_dir, train_dir, val_dir = tmp_path / "processed", tmp_path / "train", tmp_path / "val"
    write_raw(train_raw, TRAIN_RAWS)

    with pytest.raises(SystemExit, match="val split is empty"):
        prepare_data.run_polar(train_raw, val_raw, processed_dir, train_dir, val_dir)


def test_pipeline_sources_no_longer_reference_manifest() -> None:
    """清单机制整体删除：管线源码不得残留引用（#38 的删除验收）。"""
    repo_root = Path(__file__).resolve().parents[1]
    for relative in ("prepare_data.py", "endfield/data_utils.py"):
        source = (repo_root / relative).read_text(encoding="utf-8")
        assert "manifest" not in source.lower(), relative
