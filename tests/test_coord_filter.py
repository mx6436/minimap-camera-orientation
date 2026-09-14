"""标注坐标一致性过滤（#33）：换算闭式、阈值与 ref 管线接线。"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

import prepare_data
from endfield import coord_filter


def write_filter_data(
    root: Path,
    assets: Path,
    *,
    canvas: tuple[int, int] = (9600, 9000),
    rect: tuple[float, float] = (4800.0, 0.0),
    base_size: tuple[int, int] = (1440, 1350),
) -> None:
    """最小 ZmdMap 镜像 + Base.png：map01 的 lv006 矩形 + 两个 Base。"""
    root.mkdir(parents=True, exist_ok=True)
    (root / "map01_layout.json").write_text(
        json.dumps(
            {
                "base_map": "map01",
                "canvas_width": canvas[0],
                "canvas_height": canvas[1],
                "levels": {
                    "map01_lv006": {
                        "x": rect[0],
                        "y": rect[1],
                        "width": 4200,
                        "height": 4800,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    (root / "map02_layout.json").write_text(
        json.dumps(
            {
                "base_map": "map02",
                "canvas_width": 100,
                "canvas_height": 100,
                "levels": {"map02_lv005": {"x": 0, "y": 0, "width": 100, "height": 100}},
            }
        ),
        encoding="utf-8",
    )
    for region, size in (("ValleyIV", base_size), ("Wuling", (16, 16))):
        (assets / region).mkdir(parents=True, exist_ok=True)
        blank = np.zeros((size[1], size[0], 4), dtype=np.uint8)
        assert cv2.imwrite(str(assets / region / "Base.png"), blank)


def load_data(tmp_path: Path) -> coord_filter.FilterData:
    assets = tmp_path / "assets"
    write_filter_data(tmp_path / "zmd", assets)
    return coord_filter.load_filter_data(tmp_path / "zmd", assets)


def test_parse_annotation() -> None:
    annotation = coord_filter.parse_annotation("map01_lv006_x267.2_y448.7_r324.0.png")
    assert (annotation.zone, annotation.x, annotation.y) == ("map01_lv006", 267.2, 448.7)
    with pytest.raises(ValueError, match="annotation"):
        coord_filter.parse_annotation("no_coords.png")


def test_same_zone_threshold_is_inclusive(tmp_path: Path) -> None:
    data = load_data(tmp_path)
    annotation = coord_filter.parse_annotation("Wuling_Base_x100.0_y200.0_r0.0.png")
    at_limit = {"zone": "Wuling_Base", "x": 105.0, "y": 200.0}
    over = {"zone": "Wuling_Base", "x": 105.01, "y": 200.0}

    assert coord_filter.evaluate(annotation, at_limit, data) == coord_filter.Decision(True, "", 5.0)
    decision = coord_filter.evaluate(annotation, over, data)
    assert (decision.keep, decision.reason) == (False, coord_filter.REASON_DELTA)


def test_tracker_annotation_converts_into_region_base(tmp_path: Path) -> None:
    data = load_data(tmp_path)
    assert data.scales["map01"] == pytest.approx(1440 / 9600)
    annotation = coord_filter.parse_annotation("map01_lv006_x200.0_y300.0_r0.0.png")
    # base = rect*0.15 + annot*0.15/0.1625 = (720, 0) + (184.615, 276.923)
    exact = {"zone": "ValleyIV_Base", "x": 904.615, "y": 276.923}

    assert coord_filter.evaluate(annotation, exact, data).keep
    shifted = dict(exact, x=exact["x"] + 10.0)
    decision = coord_filter.evaluate(annotation, shifted, data)
    assert (decision.keep, decision.reason) == (False, coord_filter.REASON_DELTA)


def test_tracker_annotation_with_wrong_region_is_mismatch(tmp_path: Path) -> None:
    data = load_data(tmp_path)
    annotation = coord_filter.parse_annotation("map01_lv006_x200.0_y300.0_r0.0.png")
    record = {"zone": "Wuling_Base", "x": 904.6, "y": 276.9}

    decision = coord_filter.evaluate(annotation, record, data)

    assert (decision.keep, decision.reason) == (False, coord_filter.REASON_ZONE)


def test_locator_zone_mismatch_is_mismatch(tmp_path: Path) -> None:
    data = load_data(tmp_path)
    annotation = coord_filter.parse_annotation("ValleyIV_L7_133_x159.5_y172.8_r202.6.png")
    record = {"zone": "ValleyIV_Base", "x": 974.7, "y": 305.9}

    decision = coord_filter.evaluate(annotation, record, data)

    assert (decision.keep, decision.reason) == (False, coord_filter.REASON_ZONE)


def test_unsupported_tracker_prefix(tmp_path: Path) -> None:
    data = load_data(tmp_path)
    annotation = coord_filter.parse_annotation("base01_lv003_x128.5_y195.4_r321.0.png")
    record = {"zone": "OMVBase01", "x": 128.5, "y": 195.4}

    decision = coord_filter.evaluate(annotation, record, data)

    assert (decision.keep, decision.reason) == (False, coord_filter.REASON_UNSUPPORTED)


def test_missing_layout_raises(tmp_path: Path) -> None:
    assets = tmp_path / "assets"
    write_filter_data(tmp_path / "zmd", assets)
    (tmp_path / "zmd" / "map01_layout.json").unlink()

    with pytest.raises(FileNotFoundError, match="map01_layout.json"):
        coord_filter.load_filter_data(tmp_path / "zmd", assets)


def test_generate_processed_ref_applies_coord_filter(tmp_path: Path) -> None:
    raw_dir, assets = tmp_path / "raw", tmp_path / "assets"
    zmd, processed = tmp_path / "zmd", tmp_path / "processed_ref"
    write_filter_data(zmd, assets)
    raw_dir.mkdir(parents=True)
    frame = np.full((200, 200, 3), 7, dtype=np.uint8)
    kept_name = "ValleyIV_Base_x100.0_y100.0_r0.0.png"
    dropped_name = "ValleyIV_Base_x200.0_y200.0_r0.0.png"
    for name in (kept_name, dropped_name):
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
    lines = []
    for name in (kept_name, dropped_name):
        lines.append(json.dumps({**record, "name": name}))
    locate_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    names, skipped = prepare_data.generate_processed_ref(
        prepare_data.raw_samples(raw_dir, tmp_path / "empty_raw"),
        locate_path,
        assets,
        processed,
        workers=1,
        zmdmap_root=zmd,
    )

    assert names == [kept_name]
    assert skipped[dropped_name] == coord_filter.REASON_DELTA
    assert (processed / kept_name).is_file()
    assert not (processed / dropped_name).exists()
