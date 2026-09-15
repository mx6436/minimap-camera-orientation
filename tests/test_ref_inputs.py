"""ref 输入侧解析：入选门、资产存在性、坐标一致性过滤与落戳指纹条目。"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from endfield import dataset
from endfield.run_record import InputMode
from placement import coord_filter, ref_inputs
from placement.placement import Placement
from placement.records import write_jsonl
from placement.sample import ReferenceSampler
from tests._ref_fixture import locate_record, ref_fixture, write_filter_data, write_frame


def test_resolve_keeps_accepted_samples_and_records_skip_reasons(tmp_path: Path) -> None:
    fx = ref_fixture(tmp_path)

    ref = fx.resolve()

    assert [name for name, _ in ref.inputs.sources] == [fx.names["ok"], fx.names["ok2"]]
    assert ref.inputs.skipped == {
        fx.names["held"]: "held",
        fx.names["fail"]: "global_search_failed",
        fx.names["noasset"]: "asset_missing",
    }
    # accepted 是过入选门的记录（缺资产在下一步剔除），供日志计数
    assert set(ref.accepted) == {fx.names["ok"], fx.names["ok2"], fx.names["noasset"]}
    assert ref.placements[fx.names["ok"]].zone == "Test_Base"
    assert ref.assets_root == fx.assets_root
    assert ref.inputs.provenance == {"assets_root": str(fx.assets_root)}
    assert len(ref.inputs.fingerprint) == 2


def test_input_entries_carry_position_but_not_run_metadata() -> None:
    """指纹条目只取样本名与底图定位四元，定位记录的计时字段不进指纹。"""
    entry = ref_inputs.input_entries([("a.png", Placement(zone="Z_Base", x=1.5, y=2.5, scale=0.5))])

    assert json.loads(entry[0]) == ["a.png", "Z_Base", 1.5, 2.5, 0.5]


def test_resolve_counts_samples_without_locate_record(tmp_path: Path) -> None:
    fx = ref_fixture(tmp_path)
    orphan = "Test_Base_x100.0_y100.0_r9.0.png"
    write_frame(fx.train_raw / orphan)

    ref = fx.resolve()

    assert orphan not in [name for name, _ in ref.inputs.sources]
    assert ref.inputs.skipped[orphan] == ref_inputs.NO_LOCATE_RECORD


def test_resolve_without_usable_assets_leaves_no_sources(tmp_path: Path) -> None:
    fx = ref_fixture(tmp_path)
    write_jsonl(fx.locate_path, [locate_record(fx.names["noasset"], zone="Nowhere_Base")])

    ref = fx.resolve()

    assert ref.inputs.sources == []
    assert ref.inputs.skipped[fx.names["noasset"]] == ref_inputs.ASSET_MISSING


def test_resolve_rejects_records_without_scale(tmp_path: Path) -> None:
    """旧 CLI 产物（无 scale）直接报错。"""
    fx = ref_fixture(tmp_path)
    record = {
        "name": fx.names["ok"],
        "status": 0,
        "isHeld": False,
        "locConf": 0.9,
        "zone": "Test_Base",
        "x": 100.0,
        "y": 100.0,
    }
    write_jsonl(fx.locate_path, [record])

    with pytest.raises(KeyError, match="scale"):
        fx.resolve()


def test_resolve_rejects_empty_sample_set(tmp_path: Path) -> None:
    fx = ref_fixture(tmp_path)

    with pytest.raises(SystemExit, match="no raw png samples"):
        ref_inputs.resolve(
            {},
            locate_path=fx.locate_path,
            assets_root=fx.assets_root,
            zmdmap_root=fx.zmdmap_root,
        )


def test_resolve_requires_reference_assets(tmp_path: Path) -> None:
    fx = ref_fixture(tmp_path)

    with pytest.raises(SystemExit, match="参考底图目录不存在"):
        ref_inputs.resolve(
            fx.samples,
            locate_path=fx.locate_path,
            assets_root=tmp_path / "missing_assets",
            zmdmap_root=fx.zmdmap_root,
        )


def test_resolve_applies_coord_filter(tmp_path: Path) -> None:
    """坐标一致性过滤的接线：不一致样本被剔除并记原因，产物只留保留项。"""
    from endfield import prepare

    raw_dir, assets = tmp_path / "raw", tmp_path / "assets"
    zmd, processed = tmp_path / "zmd", tmp_path / "processed_ref"
    write_filter_data(zmd, assets)
    raw_dir.mkdir(parents=True)
    frame = np.full((200, 200, 3), 7, dtype=np.uint8)
    train_name = "ValleyIV_Base_x100.0_y100.0_r0.0.png"
    val_name = "ValleyIV_Base_x100.0_y100.0_r1.0.png"
    dropped_name = "ValleyIV_Base_x200.0_y200.0_r0.0.png"
    for name in (train_name, val_name, dropped_name):
        assert cv2.imwrite(str(raw_dir / name), frame)
    locate_path = tmp_path / "locate.jsonl"
    record = {
        "name": "",
        "status": 0,
        "isHeld": False,
        "locConf": 0.9,
        "zone": "ValleyIV_Base",
        "x": 100.0,
        "y": 100.0,
        "scale": 1.0,
    }
    write_jsonl(
        locate_path,
        [{**record, "name": name} for name in (train_name, val_name, dropped_name)],
    )

    ref = ref_inputs.resolve(
        {name: raw_dir / name for name in (train_name, val_name, dropped_name)},
        locate_path=locate_path,
        assets_root=assets,
        zmdmap_root=zmd,
    )
    layout = dataset.DatasetLayout(
        processed_dir=processed,
        train_dir=tmp_path / "train_ref",
        val_dir=tmp_path / "val_ref",
        subdirs=(dataset.REF_SUBDIR,),
    )
    report = prepare.prepare(
        InputMode.REF,
        ref.inputs,
        ref.renderer(ReferenceSampler(ref.assets_root)),
        layout=layout,
        train_side=[train_name, dropped_name],
        val_side=[val_name],
        workers=1,
    )

    assert ref.inputs.skipped[dropped_name] == coord_filter.REASON_DELTA
    assert report.names == (train_name, val_name)
    assert (processed / train_name).is_file()
    assert not (processed / dropped_name).exists()
