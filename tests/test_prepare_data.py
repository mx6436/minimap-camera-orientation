"""polar 数据管线（#26）：落盘、缓存戳命中/失效与 --force。"""

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

RAWS = ("a_r0.png", "b_r90.png")
FRAME_VALUE = 7
SENTINEL = 123


def write_raw(directory: Path, names: tuple[str, ...] = RAWS, value: int = FRAME_VALUE) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    frame = np.full((200, 200, 3), value, dtype=np.uint8)
    for name in names:
        assert cv2.imwrite(str(directory / name), frame)


def write_sentinel(path: Path) -> None:
    """把一个产物改成合法但值不同的 PNG：缓存命中时它必须原样留下。"""
    sentinel = np.full((preprocess.IMG_H, preprocess.IMG_W, 3), SENTINEL, dtype=np.uint8)
    assert cv2.imwrite(str(path), sentinel)


def write_manifest(path: Path, files: list[str]) -> None:
    path.write_text(json.dumps({"version": 1, "files": files}), encoding="utf-8")


def test_generate_processed_writes_pngs_and_stamp(tmp_path: Path) -> None:
    raw_dir, processed_dir = tmp_path / "raw", tmp_path / "processed"
    write_raw(raw_dir)

    names = prepare_data.generate_processed(raw_dir, processed_dir)

    assert names == list(RAWS)
    assert png_names(processed_dir) == list(RAWS)
    assert np.all(imread_png(processed_dir / RAWS[0]) == FRAME_VALUE)
    stamp = json.loads((processed_dir / preprocess_cache.STAMP_NAME).read_text(encoding="utf-8"))
    assert stamp["mode"] == "polar"
    assert stamp["definition_hash"] == preprocess.definition_hash()
    assert stamp["input_count"] == len(RAWS)


def test_generate_processed_skips_regeneration_on_cache_hit(tmp_path: Path) -> None:
    raw_dir, processed_dir = tmp_path / "raw", tmp_path / "processed"
    write_raw(raw_dir)
    prepare_data.generate_processed(raw_dir, processed_dir)
    write_sentinel(processed_dir / RAWS[0])

    names = prepare_data.generate_processed(raw_dir, processed_dir)

    assert names == list(RAWS)
    assert np.all(imread_png(processed_dir / RAWS[0]) == SENTINEL)


def test_generate_processed_regenerates_when_stamp_definition_differs(tmp_path: Path) -> None:
    raw_dir, processed_dir = tmp_path / "raw", tmp_path / "processed"
    write_raw(raw_dir)
    prepare_data.generate_processed(raw_dir, processed_dir)
    write_sentinel(processed_dir / RAWS[0])
    stamp_path = processed_dir / preprocess_cache.STAMP_NAME
    stamp = json.loads(stamp_path.read_text(encoding="utf-8"))
    stamp["definition_hash"] = "0" * 64
    stamp_path.write_text(json.dumps(stamp), encoding="utf-8")

    prepare_data.generate_processed(raw_dir, processed_dir)

    assert np.all(imread_png(processed_dir / RAWS[0]) == FRAME_VALUE)
    rewritten = json.loads(stamp_path.read_text(encoding="utf-8"))
    assert rewritten["definition_hash"] == preprocess.definition_hash()


def test_generate_processed_regenerates_when_raw_set_changes(tmp_path: Path) -> None:
    raw_dir, processed_dir = tmp_path / "raw", tmp_path / "processed"
    write_raw(raw_dir)
    prepare_data.generate_processed(raw_dir, processed_dir)
    write_sentinel(processed_dir / RAWS[0])

    write_raw(raw_dir, ("c_r180.png",), value=9)
    names = prepare_data.generate_processed(raw_dir, processed_dir)

    assert names == [*RAWS, "c_r180.png"]
    assert np.all(imread_png(processed_dir / RAWS[0]) == FRAME_VALUE)
    assert np.all(imread_png(processed_dir / "c_r180.png") == 9)


def test_generate_processed_regenerates_when_output_missing(tmp_path: Path) -> None:
    raw_dir, processed_dir = tmp_path / "raw", tmp_path / "processed"
    write_raw(raw_dir)
    prepare_data.generate_processed(raw_dir, processed_dir)
    (processed_dir / RAWS[0]).unlink()

    names = prepare_data.generate_processed(raw_dir, processed_dir)

    assert names == list(RAWS)
    assert np.all(imread_png(processed_dir / RAWS[0]) == FRAME_VALUE)


def test_generate_processed_force_ignores_cache_hit(tmp_path: Path) -> None:
    raw_dir, processed_dir = tmp_path / "raw", tmp_path / "processed"
    write_raw(raw_dir)
    prepare_data.generate_processed(raw_dir, processed_dir)
    write_sentinel(processed_dir / RAWS[0])

    prepare_data.generate_processed(raw_dir, processed_dir, force=True)

    assert np.all(imread_png(processed_dir / RAWS[0]) == FRAME_VALUE)


def test_generate_processed_failure_does_not_leave_cache_stamp(tmp_path: Path) -> None:
    """强制重生成中途失败时旧戳必须清掉：半成品目录不得被下次运行命中。"""
    raw_dir, processed_dir = tmp_path / "raw", tmp_path / "processed"
    write_raw(raw_dir)
    prepare_data.generate_processed(raw_dir, processed_dir)
    (raw_dir / RAWS[0]).write_bytes(b"not a png")

    with pytest.raises(ValueError, match="PNG"):
        prepare_data.generate_processed(raw_dir, processed_dir, force=True)

    assert not (processed_dir / preprocess_cache.STAMP_NAME).exists()
    write_raw(raw_dir, (RAWS[0],), value=11)
    prepare_data.generate_processed(raw_dir, processed_dir)
    assert np.all(imread_png(processed_dir / RAWS[0]) == 11)


def test_generate_processed_matches_across_worker_counts(tmp_path: Path) -> None:
    """并行解码只加速 I/O：workers=4 与 workers=1 的产物须逐字节一致。"""
    raw_dir = tmp_path / "raw"
    write_raw(raw_dir)
    serial_dir, parallel_dir = tmp_path / "serial", tmp_path / "parallel"

    assert prepare_data.generate_processed(raw_dir, serial_dir, workers=1) == (
        prepare_data.generate_processed(raw_dir, parallel_dir, workers=4)
    )

    def digests(directory: Path) -> dict[str, str]:
        return {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(directory.glob("*.png"))
        }

    assert digests(serial_dir) == digests(parallel_dir)


def test_run_polar_applies_manifest_split(tmp_path: Path) -> None:
    raw_dir, processed_dir = tmp_path / "raw", tmp_path / "processed"
    train_dir, val_dir = tmp_path / "train", tmp_path / "val"
    manifest = tmp_path / "val_manifest.json"
    write_raw(raw_dir)
    write_manifest(manifest, [RAWS[1]])

    prepare_data.run_polar(raw_dir, manifest, processed_dir, train_dir, val_dir)

    assert (val_dir / RAWS[1]).is_file()
    assert (train_dir / RAWS[0]).is_file()
    assert not (train_dir / RAWS[1]).exists()
