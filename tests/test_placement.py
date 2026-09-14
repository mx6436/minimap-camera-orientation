"""底图定位的测试：记录 -> `Placement` 的构造与严格性、入选门、失败分类、汇总、
zone -> 资产路径映射。"""

from __future__ import annotations

from pathlib import Path

import pytest

from placement.placement import (
    Placement,
    accept,
    classify,
    summarize,
    zone_asset_path,
)
from tests._locate_records import OK, failed


def test_placement_from_record_reads_all_four_fields() -> None:
    placement = Placement.from_record({**OK, "scale": 15.0 / 16.0})
    assert placement == Placement(zone="Wuling_Base", x=1.0, y=2.0, scale=15.0 / 16.0)
    assert Placement.from_record({**OK, "scale": 1}).scale == 1.0


def test_placement_from_record_rejects_missing_fields() -> None:
    """旧 CLI 产物缺 scale、或记录缺坐标时直接报错（契约破损，不静默跳过）。"""
    without_scale = {key: value for key, value in OK.items() if key != "scale"}
    with pytest.raises(KeyError, match="scale"):
        Placement.from_record(without_scale)
    without_xy = {key: value for key, value in OK.items() if key not in ("x", "y")}
    with pytest.raises(KeyError, match="lacks x"):
        Placement.from_record(without_xy)


def test_placement_from_record_rejects_invalid_types() -> None:
    with pytest.raises(ValueError, match="scale"):
        Placement.from_record({**OK, "scale": "1.0"})
    with pytest.raises(ValueError, match="x"):
        Placement.from_record({**OK, "x": None})
    with pytest.raises(ValueError, match="zone"):
        Placement.from_record({**OK, "zone": 3})


def test_failed_record_still_builds_placement_without_asset(tmp_path: Path) -> None:
    """定位失败的记录照常构造：zone 为空、坐标是占位值；拿不到资产在 asset_path 处表达。"""
    placement = Placement.from_record(failed("x.png", 1, "Global search failed."))
    assert placement.zone == ""
    assert placement.asset_path(tmp_path) is None


def test_accept_gates_held_and_low_conf() -> None:
    assert accept(OK) == (True, "ok")
    assert accept({**OK, "isHeld": True, "locConf": 0.2}) == (False, "held")
    assert accept({**OK, "locConf": 0.5}) == (False, "below_loc_threshold")
    assert accept(failed("x.png", 1, "Global search failed.")) == (False, "global_search_failed")


def test_classify_maps_status_and_message() -> None:
    assert classify(OK) == "ok"
    assert classify(failed("a", -1, "imread failed")) == "read_failed"
    assert classify(failed("a", -2, "minimap ROI out of bounds")) == "roi_failed"
    assert classify(failed("a", 1, "Global search failed.")) == "global_search_failed"
    assert classify(failed("a", 1, "Cold-start collecting.")) == "cold_start_unsettled"
    assert classify(failed("a", 1, "Far-jump rejected.")) == "far_jump_rejected"
    assert classify(failed("a", 2, "Screen Blocked")) == "screen_blocked"
    assert classify(failed("a", 4, "YOLO failed")) == "yolo_failed"
    assert classify(failed("a", 5, "NotInitialized")) == "not_initialized"
    assert classify(failed("a", 9, "?")) == "unknown_9"


def test_summarize_counts_and_distributions() -> None:
    records = [
        OK,
        {**OK, "name": "Wuling_Base_b.png", "attempts": 3, "locConf": 0.6},
        {**OK, "name": "Wuling_Base_c.png", "isHeld": True, "locConf": 0.4},
        failed("map01_lv001_x157.2_y486.6_r342.png", 1, "Cold-start collecting."),
    ]
    summary = summarize(records)
    assert summary["total"] == 4
    assert summary["ok"] == 3
    assert summary["failed"] == 1
    assert summary["accepted"] == 2
    assert summary["excluded_by_reason"]["held"] == 1
    assert summary["excluded_by_reason"]["cold_start_unsettled"] == 1
    assert summary["by_category"]["cold_start_unsettled"] == 1
    assert summary["attempts"] == {1: 2, 3: 2}
    assert summary["locconf"]["below_high_conf"] == 2
    assert summary["locconf"]["below_loc_threshold"] == 1
    assert summary["held"] == 1
    assert summary["families"]["Wuling"] == {"total": 3, "ok": 3, "accepted": 2}
    assert summary["families"]["map01"] == {"total": 1, "ok": 0, "accepted": 0}


def test_zone_asset_path_resolves_naming_schemes(tmp_path: Path) -> None:
    (tmp_path / "ValleyIV").mkdir()
    (tmp_path / "ValleyIV" / "Base.png").touch()
    (tmp_path / "ValleyIV" / "Lv006Tier109.png").touch()
    (tmp_path / "OMVBase").mkdir()
    (tmp_path / "OMVBase" / "OMVBase01.png").touch()

    assert zone_asset_path("ValleyIV_Base", tmp_path) == tmp_path / "ValleyIV" / "Base.png"
    assert (
        zone_asset_path("ValleyIV_L6_109", tmp_path) == tmp_path / "ValleyIV" / "Lv006Tier109.png"
    )
    assert zone_asset_path("OMVBase01", tmp_path) == tmp_path / "OMVBase" / "OMVBase01.png"
    assert zone_asset_path("Nope_Base", tmp_path) is None
    assert zone_asset_path("Nope01", tmp_path) is None


def test_zone_asset_path_is_reachable_through_placement(tmp_path: Path) -> None:
    (tmp_path / "Test").mkdir()
    (tmp_path / "Test" / "Base.png").touch()
    placement = Placement(zone="Test_Base", x=1.0, y=2.0, scale=1.0)
    assert placement.asset_path(tmp_path) == tmp_path / "Test" / "Base.png"
