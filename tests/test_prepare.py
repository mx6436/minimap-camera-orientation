"""数据准备（polar / ref）：产物、缓存戳命中/失效、划分视图与输入侧适配器。"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pytest

from endfield import dataset, prepare, preprocess, preprocess_cache
from endfield.data_utils import png_names
from endfield.dataset import REF_SUBDIR, DatasetLayout
from endfield.polar import imread_png
from endfield.preprocess import IMG_H, IMG_W
from endfield.run_record import InputMode
from placement.records import write_jsonl
from placement.sample import ReferenceSampler
from tests._ref_fixture import RefFixture, locate_record, ref_fixture

TRAIN_RAWS = ("a_r0.png", "b_r90.png")
HARD_RAWS = ("h_r45.png", "h2_r300.png")
VAL_RAWS = ("c_r180.png",)
FRAME_VALUE = 7
SENTINEL = 123


def write_raw(directory: Path, names: tuple[str, ...], value: int = FRAME_VALUE) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    frame = np.full((200, 200, 3), value, dtype=np.uint8)
    for name in names:
        assert cv2.imwrite(str(directory / name), frame)


def write_sentinel(path: Path, channels: int = 3) -> None:
    """把一个产物改成合法但值不同的 PNG。"""
    sentinel = np.full((IMG_H, IMG_W, channels), SENTINEL, dtype=np.uint8)
    assert cv2.imwrite(str(path), sentinel)


def digests(directory: Path) -> dict[str, str]:
    return {
        str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(directory.rglob("*.png"))
    }


@dataclass
class PolarCase:
    train_raw: Path
    hard_raw: Path
    val_raw: Path
    layout: DatasetLayout

    @property
    def processed(self) -> Path:
        return self.layout.processed_dir

    def prepare(self, *, force: bool = False, workers: int = prepare.IO_WORKERS):
        return prepare.prepare(
            InputMode.POLAR,
            prepare.polar_inputs(
                dataset.raw_samples((self.train_raw, self.hard_raw), self.val_raw)
            ),
            prepare.polar_renderer(),
            layout=self.layout,
            train_side=png_names(self.train_raw) + png_names(self.hard_raw),
            val_side=png_names(self.val_raw),
            force=force,
            workers=workers,
        )


def layout(tmp_path: Path, name: str, *, ref: bool = False) -> DatasetLayout:
    suffix = f"_{name}" if name else ""
    return DatasetLayout(
        processed_dir=tmp_path / f"processed{suffix}",
        train_dir=tmp_path / f"train{suffix}",
        val_dir=tmp_path / f"val{suffix}",
        subdirs=(REF_SUBDIR,) if ref else (),
    )


def polar_case(tmp_path: Path, name: str = "", hard: tuple[str, ...] = ()) -> PolarCase:
    train_raw, val_raw = tmp_path / "train_raw", tmp_path / "val_raw"
    hard_raw = tmp_path / "hard_raw"
    write_raw(train_raw, TRAIN_RAWS)
    write_raw(hard_raw, hard)
    write_raw(val_raw, VAL_RAWS)
    return PolarCase(train_raw, hard_raw, val_raw, layout(tmp_path, name))


def prepare_ref(
    fx: RefFixture,
    ref_layout: DatasetLayout,
    *,
    force: bool = False,
    workers: int = prepare.IO_WORKERS,
):
    """ref 全管线：输入侧解析 -> 生成 + 划分视图，返回（输入侧值, 报告）。"""
    ref = fx.resolve()
    report = prepare.prepare(
        InputMode.REF,
        ref.inputs,
        ref.renderer(ReferenceSampler(ref.assets_root)),
        layout=ref_layout,
        train_side=png_names(fx.train_raw),
        val_side=png_names(fx.val_raw),
        force=force,
        workers=workers,
    )
    return ref, report


# --------------------------------------------------------------------------- #
# 输入侧适配器（数据集布局）
# --------------------------------------------------------------------------- #


def test_raw_samples_merges_all_raw_directories(tmp_path: Path) -> None:
    """训练侧目录并集 + 验证目录：困难样本随 train_raw 一起进并集。"""
    train_raw, hard_raw, val_raw = (
        tmp_path / "train_raw",
        tmp_path / "hard_raw",
        tmp_path / "val_raw",
    )
    write_raw(train_raw, TRAIN_RAWS)
    write_raw(hard_raw, HARD_RAWS)
    write_raw(val_raw, VAL_RAWS)

    samples = dataset.raw_samples((train_raw, hard_raw), val_raw)

    assert samples == {
        name: directory / name
        for directory, names in (
            (train_raw, TRAIN_RAWS),
            (hard_raw, HARD_RAWS),
            (val_raw, VAL_RAWS),
        )
        for name in names
    }


def test_raw_samples_rejects_name_present_in_both_directories(tmp_path: Path) -> None:
    train_raw, val_raw = tmp_path / "train_raw", tmp_path / "val_raw"
    write_raw(train_raw, ("a_r0.png",))
    write_raw(val_raw, ("a_r0.png",))

    with pytest.raises(SystemExit, match="both"):
        dataset.raw_samples((train_raw,), val_raw)


def test_raw_samples_rejects_name_in_both_train_directories(tmp_path: Path) -> None:
    """样本在两个训练目录同样硬报错：碰撞即说明没搬干净。"""
    train_raw, hard_raw = tmp_path / "train_raw", tmp_path / "hard_raw"
    write_raw(train_raw, ("a_r0.png",))
    write_raw(hard_raw, ("a_r0.png",))

    with pytest.raises(SystemExit, match="both"):
        dataset.raw_samples((train_raw, hard_raw), tmp_path / "val_raw")


def test_directory_split_returns_each_side_sorted() -> None:
    assert dataset.directory_split(["b", "a"], ["d", "c"]) == (["a", "b"], ["c", "d"])


@pytest.mark.parametrize(
    ("train_names", "val_names", "match"),
    [([], ["v"], "train split is empty"), (["t"], [], "val split is empty")],
)
def test_directory_split_rejects_empty_side(
    train_names: list[str], val_names: list[str], match: str
) -> None:
    with pytest.raises(SystemExit, match=match):
        dataset.directory_split(train_names, val_names)


# --------------------------------------------------------------------------- #
# polar 管线
# --------------------------------------------------------------------------- #


def test_prepare_polar_writes_pngs_and_stamp(tmp_path: Path) -> None:
    case = polar_case(tmp_path)

    report = case.prepare()

    assert list(report.names) == [*TRAIN_RAWS, *VAL_RAWS]
    assert not report.cache_hit
    assert png_names(case.processed) == [*TRAIN_RAWS, *VAL_RAWS]
    assert np.all(imread_png(case.processed / TRAIN_RAWS[0]) == FRAME_VALUE)
    stamp = json.loads((case.processed / preprocess_cache.STAMP_NAME).read_text(encoding="utf-8"))
    assert stamp["mode"] == "polar"
    assert stamp["definition_hash"] == preprocess.definition_hash()
    assert stamp["input_count"] == len(TRAIN_RAWS) + len(VAL_RAWS)
    assert report.definition_hash == preprocess.definition_hash()


def test_prepare_polar_skips_regeneration_on_cache_hit(tmp_path: Path) -> None:
    case = polar_case(tmp_path)
    case.prepare()
    write_sentinel(case.processed / TRAIN_RAWS[0])

    report = case.prepare()

    assert list(report.names) == [*TRAIN_RAWS, *VAL_RAWS]
    assert report.cache_hit
    assert np.all(imread_png(case.processed / TRAIN_RAWS[0]) == SENTINEL)


def test_prepare_polar_regenerates_when_stamp_definition_differs(tmp_path: Path) -> None:
    case = polar_case(tmp_path)
    case.prepare()
    write_sentinel(case.processed / TRAIN_RAWS[0])
    stamp_path = case.processed / preprocess_cache.STAMP_NAME
    stamp = json.loads(stamp_path.read_text(encoding="utf-8"))
    stamp["definition_hash"] = "0" * 64
    stamp_path.write_text(json.dumps(stamp), encoding="utf-8")

    report = case.prepare()

    assert not report.cache_hit
    assert np.all(imread_png(case.processed / TRAIN_RAWS[0]) == FRAME_VALUE)
    rewritten = json.loads(stamp_path.read_text(encoding="utf-8"))
    assert rewritten["definition_hash"] == preprocess.definition_hash()


def test_prepare_polar_regenerates_when_raw_set_changes(tmp_path: Path) -> None:
    case = polar_case(tmp_path)
    case.prepare()
    write_sentinel(case.processed / TRAIN_RAWS[0])

    write_raw(case.val_raw, ("d_r270.png",), value=9)
    report = case.prepare()

    assert list(report.names) == [*TRAIN_RAWS, *VAL_RAWS, "d_r270.png"]
    assert np.all(imread_png(case.processed / TRAIN_RAWS[0]) == FRAME_VALUE)
    assert np.all(imread_png(case.processed / "d_r270.png") == 9)


def test_prepare_polar_regenerates_when_output_missing(tmp_path: Path) -> None:
    case = polar_case(tmp_path)
    case.prepare()
    (case.processed / TRAIN_RAWS[0]).unlink()

    report = case.prepare()

    assert not report.cache_hit
    assert np.all(imread_png(case.processed / TRAIN_RAWS[0]) == FRAME_VALUE)


def test_prepare_polar_force_ignores_cache_hit(tmp_path: Path) -> None:
    case = polar_case(tmp_path)
    case.prepare()
    write_sentinel(case.processed / TRAIN_RAWS[0])

    report = case.prepare(force=True)

    assert not report.cache_hit
    assert np.all(imread_png(case.processed / TRAIN_RAWS[0]) == FRAME_VALUE)


def test_prepare_polar_failure_does_not_leave_cache_stamp(tmp_path: Path) -> None:
    case = polar_case(tmp_path)
    case.prepare()
    (case.train_raw / TRAIN_RAWS[0]).write_bytes(b"not a png")

    with pytest.raises(ValueError, match="PNG"):
        case.prepare(force=True)

    assert not (case.processed / preprocess_cache.STAMP_NAME).exists()
    write_raw(case.train_raw, (TRAIN_RAWS[0],), value=11)
    case.prepare()
    assert np.all(imread_png(case.processed / TRAIN_RAWS[0]) == 11)


def test_prepare_polar_matches_across_worker_counts(tmp_path: Path) -> None:
    """workers=4 与 workers=1 的产物逐字节一致。"""
    serial = polar_case(tmp_path, "serial")
    parallel = polar_case(tmp_path, "parallel")

    serial_report = serial.prepare(workers=1)
    parallel_report = parallel.prepare(workers=4)

    assert list(serial_report.names) == list(parallel_report.names)
    assert digests(serial.processed) == digests(parallel.processed)


def test_prepare_polar_links_each_directory_to_its_side(tmp_path: Path) -> None:
    """train_raw 全量进 train 视图，val_raw 全量进 val 视图。"""
    case = polar_case(tmp_path)

    report = case.prepare()

    assert png_names(case.layout.train_dir) == sorted(TRAIN_RAWS)
    assert png_names(case.layout.val_dir) == sorted(VAL_RAWS)
    assert png_names(case.processed) == sorted([*TRAIN_RAWS, *VAL_RAWS])
    assert list(report.train) == sorted(TRAIN_RAWS)
    assert list(report.val) == sorted(VAL_RAWS)


def test_prepare_polar_puts_hard_samples_in_train_view_only(tmp_path: Path) -> None:
    """hard_raw 是训练侧：进 train 视图与训练划分，不进 val。"""
    case = polar_case(tmp_path, "", hard=HARD_RAWS)

    report = case.prepare()

    assert png_names(case.processed) == sorted([*TRAIN_RAWS, *HARD_RAWS, *VAL_RAWS])
    assert png_names(case.layout.train_dir) == sorted([*TRAIN_RAWS, *HARD_RAWS])
    assert png_names(case.layout.val_dir) == sorted(VAL_RAWS)
    assert list(report.train) == sorted([*TRAIN_RAWS, *HARD_RAWS])
    assert list(report.val) == sorted(VAL_RAWS)


def test_prepare_polar_keeps_cache_when_sample_moves_between_directories(tmp_path: Path) -> None:
    """样本换侧只换视图，不重算 processed。"""
    case = polar_case(tmp_path)
    case.prepare()
    write_sentinel(case.processed / "b_r90.png")

    (case.train_raw / "b_r90.png").rename(case.val_raw / "b_r90.png")
    report = case.prepare()

    assert report.cache_hit
    assert np.all(imread_png(case.processed / "b_r90.png") == SENTINEL)
    assert png_names(case.layout.train_dir) == ["a_r0.png"]
    assert png_names(case.layout.val_dir) == ["b_r90.png", *VAL_RAWS]


def test_prepare_polar_keeps_cache_when_sample_moves_to_hard_dir(tmp_path: Path) -> None:
    """移进 hard_raw 只换训练目录、不换并集：cache hit，产物与训练视图都不变。"""
    case = polar_case(tmp_path)
    case.prepare()
    write_sentinel(case.processed / TRAIN_RAWS[0])

    (case.train_raw / TRAIN_RAWS[0]).rename(case.hard_raw / TRAIN_RAWS[0])
    report = case.prepare()

    assert report.cache_hit
    assert np.all(imread_png(case.processed / TRAIN_RAWS[0]) == SENTINEL)
    assert png_names(case.layout.train_dir) == sorted(TRAIN_RAWS)
    assert png_names(case.layout.val_dir) == sorted(VAL_RAWS)


def test_prepare_polar_regenerates_when_hard_sample_disappears(tmp_path: Path) -> None:
    """硬样本从 hard_raw 消失 = 并集变小：cache miss 且产物按新并集重建。"""
    case = polar_case(tmp_path, "", hard=HARD_RAWS)
    case.prepare()

    (case.hard_raw / HARD_RAWS[0]).unlink()
    report = case.prepare()

    assert not report.cache_hit
    assert png_names(case.processed) == sorted([*TRAIN_RAWS, HARD_RAWS[1], *VAL_RAWS])
    assert png_names(case.layout.train_dir) == sorted([*TRAIN_RAWS, HARD_RAWS[1]])


def test_prepare_polar_rejects_empty_side(tmp_path: Path) -> None:
    train_raw, val_raw = tmp_path / "train_raw", tmp_path / "val_raw"
    write_raw(train_raw, TRAIN_RAWS)
    val_raw.mkdir()
    case = PolarCase(train_raw, tmp_path / "hard_raw", val_raw, layout(tmp_path, ""))

    with pytest.raises(SystemExit, match="val split is empty"):
        case.prepare()


def test_prepare_data_main_wires_the_default_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`prepare-data` 默认走 polar：默认数据集重定向后与库侧同一结果（含 hard_raw）。"""
    from cli import prepare_data

    train_raw, hard_raw, val_raw = (
        tmp_path / "train_raw",
        tmp_path / "hard_raw",
        tmp_path / "val_raw",
    )
    write_raw(train_raw, TRAIN_RAWS)
    write_raw(hard_raw, HARD_RAWS)
    write_raw(val_raw, VAL_RAWS)
    polar = layout(tmp_path, "")
    monkeypatch.setattr(dataset, "TRAIN_RAW_DIRS", (train_raw, hard_raw))
    monkeypatch.setattr(dataset, "VAL_RAW_DIR", val_raw)
    monkeypatch.setattr(dataset, "POLAR_LAYOUT", polar)
    monkeypatch.setattr(sys, "argv", ["prepare-data", "--workers", "1"])

    prepare_data.main()

    assert png_names(polar.processed_dir) == sorted([*TRAIN_RAWS, *HARD_RAWS, *VAL_RAWS])
    assert png_names(polar.train_dir) == sorted([*TRAIN_RAWS, *HARD_RAWS])
    assert png_names(polar.val_dir) == sorted(VAL_RAWS)


# --------------------------------------------------------------------------- #
# ref 管线
# --------------------------------------------------------------------------- #


def test_prepare_ref_writes_both_streams_and_skips_unaccepted(tmp_path: Path) -> None:
    fx = ref_fixture(tmp_path)
    ref_layout = layout(tmp_path, "", ref=True)

    _, report = prepare_ref(fx, ref_layout)

    assert list(report.names) == [fx.names["ok"], fx.names["ok2"]]
    assert report.skipped == {
        fx.names["held"]: "held",
        fx.names["fail"]: "global_search_failed",
        fx.names["noasset"]: "asset_missing",
    }
    observed = imread_png(ref_layout.processed_dir / fx.names["ok"])
    reference = imread_png(ref_layout.processed_dir / REF_SUBDIR / fx.names["ok"])
    assert observed.shape == (IMG_H, IMG_W, 3)
    assert reference.shape == (IMG_H, IMG_W, 4)
    assert observed.dtype == np.uint8 and reference.dtype == np.uint8
    # 底图与观测一致：观测流 = 参考 BGR 流，参考 alpha 全 255（无参考缺失）
    assert np.all(observed == 200)
    assert np.array_equal(observed, reference[..., :3])
    assert np.all(reference[..., 3] == 255)


def test_prepare_ref_marks_out_of_bounds_and_whitens(tmp_path: Path) -> None:
    fx = ref_fixture(tmp_path)
    # 底图小于参考窗口：越界处无内容 -> alpha 0，BGR 取白底
    small = np.full((120, 120, 4), 200, dtype=np.uint8)
    small[..., 3] = 255
    assert cv2.imwrite(str(fx.assets_root / "Test" / "Base.png"), small)
    ref_layout = layout(tmp_path, "", ref=True)

    prepare_ref(fx, ref_layout)

    reference = imread_png(ref_layout.processed_dir / REF_SUBDIR / fx.names["ok"])
    alpha = reference[..., 3]
    assert alpha.min() == 0 and alpha.max() == 255
    assert np.all(reference[..., :3][alpha == 0] == 255)


def test_prepare_ref_is_deterministic_on_rerun(tmp_path: Path) -> None:
    fx = ref_fixture(tmp_path)
    ref_layout = layout(tmp_path, "", ref=True)

    prepare_ref(fx, ref_layout)
    first = digests(ref_layout.processed_dir)
    _, report = prepare_ref(fx, ref_layout)

    assert report.cache_hit
    assert first == digests(ref_layout.processed_dir)
    assert set(first) == {
        fx.names["ok"],
        fx.names["ok2"],
        f"{REF_SUBDIR}/{fx.names['ok']}",
        f"{REF_SUBDIR}/{fx.names['ok2']}",
    }


def test_prepare_ref_skips_regeneration_on_cache_hit(tmp_path: Path) -> None:
    fx = ref_fixture(tmp_path)
    ref_layout = layout(tmp_path, "", ref=True)
    prepare_ref(fx, ref_layout)
    observed_path = ref_layout.processed_dir / fx.names["ok"]
    reference_path = ref_layout.processed_dir / REF_SUBDIR / fx.names["ok"]
    write_sentinel(observed_path, 3)
    write_sentinel(reference_path, 4)

    _, report = prepare_ref(fx, ref_layout)

    assert report.cache_hit
    stamp = json.loads(
        (ref_layout.processed_dir / preprocess_cache.STAMP_NAME).read_text(encoding="utf-8")
    )
    assert stamp["mode"] == "ref"
    assert stamp["definition_hash"] == preprocess.definition_hash()
    assert stamp["provenance"] == {"assets_root": str(fx.assets_root)}
    assert np.all(imread_png(observed_path) == SENTINEL)
    assert np.all(imread_png(reference_path) == SENTINEL)


def test_prepare_ref_regenerates_when_assets_root_provenance_differs(tmp_path: Path) -> None:
    fx = ref_fixture(tmp_path)
    ref_layout = layout(tmp_path, "", ref=True)
    prepare_ref(fx, ref_layout)
    observed_path = ref_layout.processed_dir / fx.names["ok"]
    write_sentinel(observed_path, 3)
    stamp_path = ref_layout.processed_dir / preprocess_cache.STAMP_NAME
    stamp = json.loads(stamp_path.read_text(encoding="utf-8"))
    stamp["provenance"]["assets_root"] = str(tmp_path / "other_assets")
    stamp_path.write_text(json.dumps(stamp), encoding="utf-8")

    _, report = prepare_ref(fx, ref_layout)

    assert not report.cache_hit
    assert np.all(imread_png(observed_path) == 200)
    assert json.loads(stamp_path.read_text(encoding="utf-8"))["provenance"] == {
        "assets_root": str(fx.assets_root)
    }


def test_prepare_ref_regenerates_when_locate_fields_change(tmp_path: Path) -> None:
    fx = ref_fixture(tmp_path)
    ref_layout = layout(tmp_path, "", ref=True)
    prepare_ref(fx, ref_layout)
    write_sentinel(ref_layout.processed_dir / fx.names["ok"], 3)

    # 同名单、不同坐标：定位记录字段属于前处理输入，必须触发失效
    write_jsonl(
        fx.locate_path,
        [locate_record(fx.names["ok"], x=101.0), locate_record(fx.names["ok2"])],
    )
    _, report = prepare_ref(fx, ref_layout)

    assert not report.cache_hit
    assert np.all(imread_png(ref_layout.processed_dir / fx.names["ok"]) == 200)


def test_prepare_ref_regenerates_when_stamp_definition_differs(tmp_path: Path) -> None:
    fx = ref_fixture(tmp_path)
    ref_layout = layout(tmp_path, "", ref=True)
    prepare_ref(fx, ref_layout)
    write_sentinel(ref_layout.processed_dir / fx.names["ok"], 3)
    stamp_path = ref_layout.processed_dir / preprocess_cache.STAMP_NAME
    stamp = json.loads(stamp_path.read_text(encoding="utf-8"))
    stamp["definition_hash"] = "0" * 64
    stamp_path.write_text(json.dumps(stamp), encoding="utf-8")

    _, report = prepare_ref(fx, ref_layout)

    assert not report.cache_hit
    assert np.all(imread_png(ref_layout.processed_dir / fx.names["ok"]) == 200)
    assert json.loads(stamp_path.read_text(encoding="utf-8"))["definition_hash"] == (
        preprocess.definition_hash()
    )


def test_prepare_ref_matches_across_worker_counts(tmp_path: Path) -> None:
    """并行解码只加速 I/O：workers=4 与 workers=1 的两路产物须逐字节一致。"""
    fx = ref_fixture(tmp_path)
    serial_layout = layout(tmp_path, "serial", ref=True)
    parallel_layout = layout(tmp_path, "parallel", ref=True)

    serial_ref, serial_report = prepare_ref(fx, serial_layout, workers=1)
    parallel_ref, parallel_report = prepare_ref(fx, parallel_layout, workers=4)

    assert list(serial_report.names) == list(parallel_report.names)
    assert digests(serial_layout.processed_dir) == digests(parallel_layout.processed_dir)
    assert serial_ref.inputs.fingerprint == parallel_ref.inputs.fingerprint


def test_prepare_ref_cache_ignores_directory_membership(tmp_path: Path) -> None:
    """输入指纹取两侧并集：样本换侧只换划分，不触发重算。"""
    fx = ref_fixture(tmp_path)
    ref_layout = layout(tmp_path, "", ref=True)
    prepare_ref(fx, ref_layout)
    write_sentinel(ref_layout.processed_dir / fx.names["ok"], 3)

    (fx.val_raw / fx.names["ok"]).rename(fx.train_raw / fx.names["ok"])
    (fx.train_raw / fx.names["ok2"]).rename(fx.val_raw / fx.names["ok2"])
    _, report = prepare_ref(fx, ref_layout)

    assert report.cache_hit
    assert list(report.names) == [fx.names["ok"], fx.names["ok2"]]
    assert np.all(imread_png(ref_layout.processed_dir / fx.names["ok"]) == SENTINEL)
    assert png_names(ref_layout.train_dir) == [fx.names["ok"]]
    assert png_names(ref_layout.val_dir) == [fx.names["ok2"]]


def test_prepare_ref_force_regenerates(tmp_path: Path) -> None:
    fx = ref_fixture(tmp_path)
    ref_layout = layout(tmp_path, "", ref=True)
    prepare_ref(fx, ref_layout)
    write_sentinel(ref_layout.processed_dir / fx.names["ok"], 3)

    _, report = prepare_ref(fx, ref_layout, force=True)

    assert not report.cache_hit
    assert np.all(imread_png(ref_layout.processed_dir / fx.names["ok"]) == 200)


def test_prepare_ref_links_each_directory_to_its_side_and_drops_unusable(tmp_path: Path) -> None:
    fx = ref_fixture(tmp_path)
    ref_layout = layout(tmp_path, "", ref=True)

    _, report = prepare_ref(fx, ref_layout)

    assert (ref_layout.val_dir / fx.names["ok"]).is_file()
    assert (ref_layout.val_dir / REF_SUBDIR / fx.names["ok"]).is_file()
    assert (ref_layout.train_dir / fx.names["ok2"]).is_file()
    assert (ref_layout.train_dir / REF_SUBDIR / fx.names["ok2"]).is_file()
    for name in (fx.names["held"], fx.names["fail"], fx.names["noasset"]):
        assert not (ref_layout.train_dir / name).exists()
        assert not (ref_layout.val_dir / name).exists()
    assert not (ref_layout.train_dir / fx.names["ok"]).exists()
    assert not (ref_layout.val_dir / fx.names["ok2"]).exists()
    assert list(report.train) == [fx.names["ok2"]]
    assert list(report.val) == [fx.names["ok"]]


def test_prepare_ref_rejects_side_without_usable_samples(tmp_path: Path) -> None:
    fx = ref_fixture(tmp_path)
    (fx.val_raw / fx.names["ok"]).rename(fx.train_raw / fx.names["ok"])
    ref_layout = layout(tmp_path, "", ref=True)

    with pytest.raises(SystemExit, match="val split is empty"):
        prepare_ref(fx, ref_layout)


def test_prepare_ref_writes_white_backdrop_reference(tmp_path: Path) -> None:
    fx = ref_fixture(tmp_path)
    # 底图中心开一条 alpha=0 带：参考 BGR 应为白底而非观测像素
    asset = np.full((200, 200, 4), 200, dtype=np.uint8)
    asset[..., 3] = 255
    asset[:, 90:110, 3] = 0
    assert cv2.imwrite(str(fx.assets_root / "Test" / "Base.png"), asset)
    ref_layout = layout(tmp_path, "", ref=True)

    _, report = prepare_ref(fx, ref_layout)

    assert list(report.names) == [fx.names["ok"], fx.names["ok2"]]
    reference = imread_png(ref_layout.processed_dir / REF_SUBDIR / fx.names["ok"])
    alpha = reference[..., 3]
    assert np.any(alpha == 0) and np.any(alpha == 255)
    # 参考缺失处为白底，不透明处为底图原始 BGR
    assert np.all(reference[..., :3][alpha == 0] == 255)
    assert np.all(reference[..., :3][alpha == 255] == 200)
