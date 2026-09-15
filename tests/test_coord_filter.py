"""标注坐标一致性过滤：zone 判据与阈值判据。

过滤接线到 ref 输入侧的部分在 `tests/test_ref_inputs.py`。
"""

from __future__ import annotations

import pytest

from placement import coord_filter
from tests._ref_fixture import placement


def test_parse_annotation() -> None:
    annotation = coord_filter.parse_annotation("Wuling_L9_395_x578.0_y1666.9_r344.7.png")
    assert (annotation.zone, annotation.x, annotation.y) == ("Wuling_L9_395", 578.0, 1666.9)
    with pytest.raises(ValueError, match="annotation"):
        coord_filter.parse_annotation("no_coords.png")


def test_same_zone_threshold_is_inclusive() -> None:
    annotation = coord_filter.parse_annotation("Wuling_Base_x100.0_y200.0_r0.0.png")

    at_limit = placement("Wuling_Base", 105.0, 200.0)
    over = placement("Wuling_Base", 105.01, 200.0)

    assert coord_filter.evaluate(annotation, at_limit) == coord_filter.Decision(True, "", 5.0)
    decision = coord_filter.evaluate(annotation, over)
    assert (decision.keep, decision.reason) == (False, coord_filter.REASON_DELTA)


def test_zone_mismatch_is_mismatch() -> None:
    annotation = coord_filter.parse_annotation("ValleyIV_L7_133_x159.5_y172.8_r202.6.png")

    decision = coord_filter.evaluate(annotation, placement("ValleyIV_Base", 974.7, 305.9))

    assert (decision.keep, decision.reason) == (False, coord_filter.REASON_ZONE)
