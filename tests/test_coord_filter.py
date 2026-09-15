"""标注坐标一致性过滤（#33）：换算闭式与阈值判据。

过滤接线到 ref 输入侧的部分在 `tests/test_ref_inputs.py`。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from placement import coord_filter
from tests._ref_fixture import placement, write_filter_data


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

    at_limit = placement("Wuling_Base", 105.0, 200.0)
    over = placement("Wuling_Base", 105.01, 200.0)

    assert coord_filter.evaluate(annotation, at_limit, data) == coord_filter.Decision(True, "", 5.0)
    decision = coord_filter.evaluate(annotation, over, data)
    assert (decision.keep, decision.reason) == (False, coord_filter.REASON_DELTA)


def test_tracker_annotation_converts_into_region_base(tmp_path: Path) -> None:
    data = load_data(tmp_path)
    assert data.scales["map01"] == pytest.approx(1440 / 9600)
    annotation = coord_filter.parse_annotation("map01_lv006_x200.0_y300.0_r0.0.png")
    # base = rect*0.15 + annot*0.15/0.1625 = (720, 0) + (184.615, 276.923)
    exact = placement("ValleyIV_Base", 904.615, 276.923)

    assert coord_filter.evaluate(annotation, exact, data).keep
    shifted = placement("ValleyIV_Base", 914.615, 276.923)
    decision = coord_filter.evaluate(annotation, shifted, data)
    assert (decision.keep, decision.reason) == (False, coord_filter.REASON_DELTA)


def test_tracker_annotation_with_wrong_region_is_mismatch(tmp_path: Path) -> None:
    data = load_data(tmp_path)
    annotation = coord_filter.parse_annotation("map01_lv006_x200.0_y300.0_r0.0.png")

    decision = coord_filter.evaluate(annotation, placement("Wuling_Base", 904.6, 276.9), data)

    assert (decision.keep, decision.reason) == (False, coord_filter.REASON_ZONE)


def test_locator_zone_mismatch_is_mismatch(tmp_path: Path) -> None:
    data = load_data(tmp_path)
    annotation = coord_filter.parse_annotation("ValleyIV_L7_133_x159.5_y172.8_r202.6.png")

    decision = coord_filter.evaluate(annotation, placement("ValleyIV_Base", 974.7, 305.9), data)

    assert (decision.keep, decision.reason) == (False, coord_filter.REASON_ZONE)


def test_unsupported_tracker_prefix(tmp_path: Path) -> None:
    data = load_data(tmp_path)
    annotation = coord_filter.parse_annotation("base01_lv003_x128.5_y195.4_r321.0.png")

    decision = coord_filter.evaluate(annotation, placement("OMVBase01", 128.5, 195.4), data)

    assert (decision.keep, decision.reason) == (False, coord_filter.REASON_UNSUPPORTED)


def test_missing_layout_raises(tmp_path: Path) -> None:
    assets = tmp_path / "assets"
    write_filter_data(tmp_path / "zmd", assets)
    (tmp_path / "zmd" / "map01_layout.json").unlink()

    with pytest.raises(FileNotFoundError, match="map01_layout.json"):
        coord_filter.load_filter_data(tmp_path / "zmd", assets)
