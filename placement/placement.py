"""底图定位：从一条定位记录取出参考裁剪前提，并判定记录的可用性。

记录字段见 docs/maplocator-workspace.md：name/status/message/zone/x/y/rot/scale/
locConf/isHeld/latencyMs/attempts/elapsedMs。其中 `scale` 是 zone 的
`ZoneTemplateScale`（底图与观测的像素尺度比，无缩放 zone 为 1.0），由定位侧携带，
训练/实机侧据此裁剪参考底图。
"""

from __future__ import annotations

import re
import statistics
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from placement.records import (
    STATUS_READ_FAILED,
    STATUS_ROI_FAILED,
    STATUS_SUCCESS,
    Record,
)

# 分数门限：kHighConfidenceOverride（旁路冷启动共识）与 MapLocator 默认 loc_threshold。
_HIGH_CONF = 0.85
_DEFAULT_LOC_THRESHOLD = 0.55
# 缺 scale 即产物早于 output-scale 的 CLI 版本（docs/maplocator-workspace.md）
_SCALE_HINT = " (regenerate with the scale-aware CLI)"

_MESSAGE_CATEGORIES = {
    "Global search failed.": "global_search_failed",
    "Cold-start collecting.": "cold_start_unsettled",
    "Far-jump rejected.": "far_jump_rejected",
}
_STATUS_CATEGORIES = {
    2: "screen_blocked",
    3: "teleported",
    4: "yolo_failed",
    5: "not_initialized",
}

_BASE_RE = re.compile(r"^(.+)_Base$")
_TIER_RE = re.compile(r"^(.+)_L(\d+)_(\d+)$")


@dataclass(frozen=True)
class Placement:
    """底图定位：一次参考裁剪的全部前提，四项同出一条定位记录。"""

    zone: str
    x: float
    y: float
    scale: float

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> Placement:
        """定位记录 -> 底图定位；缺项或类型不符即报错（契约破损，不静默跳过）。

        定位失败的记录照常构造：其 `zone` 为空、坐标与尺度是占位值。可用性由
        `accept` 判定，拿不到资产在 `asset_path` 处表达。
        """
        zone = record.get("zone", "")
        if not isinstance(zone, str):
            raise ValueError(f"locate record zone must be a string, got {zone!r}")
        return cls(
            zone=zone,
            x=_require_number(record, "x"),
            y=_require_number(record, "y"),
            scale=_require_number(record, "scale", hint=_SCALE_HINT),
        )

    def asset_path(self, assets_root: Path) -> Path | None:
        """该 zone 的参考底图资产路径；zone 为空或资产不存在 -> None。"""
        return zone_asset_path(self.zone, assets_root)


def _require_number(record: Mapping[str, Any], key: str, hint: str = "") -> float:
    try:
        value = record[key]
    except KeyError:
        raise KeyError(f"locate record lacks {key}{hint}: {record!r}") from None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"locate record {key} must be a number, got {value!r}")
    return float(value)


def accept(record: Record) -> tuple[bool, str]:
    """产物入选门：定位失败、held、locConf 低于 loc_threshold 均不入选。

    held 是 MapLocator 对「全局搜索没有过线峰、只能放行裸峰」的显式标记；识别层
    JSON 不带该字段，直连 CLI 才有，因此这道门是精确的（held 分数均 < 0.55）。
    """
    if int(record.get("status", -1)) != STATUS_SUCCESS:
        return False, classify(record)
    if record.get("isHeld"):
        return False, "held"
    if float(record.get("locConf", 0.0)) < _DEFAULT_LOC_THRESHOLD:
        return False, "below_loc_threshold"
    return True, "ok"


def classify(record: Record) -> str:
    """把状态与 message 归到稳定的失败类别，供报告与排除决策使用。"""
    status = int(record.get("status", 0))
    if status == STATUS_SUCCESS:
        return "ok"
    if status == STATUS_READ_FAILED:
        return "read_failed"
    if status == STATUS_ROI_FAILED:
        return "roi_failed"
    if status == 1:
        return _MESSAGE_CATEGORIES.get(str(record.get("message", "")), "tracking_lost")
    return _STATUS_CATEGORIES.get(status, f"unknown_{status}")


def summarize(records: Iterable[Record]) -> dict[str, Any]:
    """全量统计：成败、失败分类、调用次数与 locConf 分布、按命名族成功率。"""
    records = list(records)
    successes = [record for record in records if record.get("status") == STATUS_SUCCESS]
    by_status = Counter(int(record.get("status", 0)) for record in records)
    by_category = Counter(classify(record) for record in records)
    attempts = Counter(int(record.get("attempts", 0)) for record in records)
    accepted = 0
    excluded_by_reason: Counter[str] = Counter()
    for record in records:
        is_accepted, reason = accept(record)
        if is_accepted:
            accepted += 1
        else:
            excluded_by_reason[reason] += 1

    locconf = None
    if successes:
        confidences = sorted(float(record.get("locConf", 0.0)) for record in successes)
        if len(confidences) >= 2:
            p25, median, p75 = statistics.quantiles(confidences, n=4)
        else:
            p25 = median = p75 = confidences[0]
        locconf = {
            "min": confidences[0],
            "p25": p25,
            "median": median,
            "p75": p75,
            "max": confidences[-1],
            "below_high_conf": sum(1 for value in confidences if value < _HIGH_CONF),
            "below_loc_threshold": sum(
                1 for value in confidences if value < _DEFAULT_LOC_THRESHOLD
            ),
        }

    families: dict[str, dict[str, int]] = {}
    for record in records:
        family = str(record.get("name", "")).split("_", 1)[0]
        entry = families.setdefault(family, {"total": 0, "ok": 0, "accepted": 0})
        entry["total"] += 1
        if record.get("status") == STATUS_SUCCESS:
            entry["ok"] += 1
        if accept(record)[0]:
            entry["accepted"] += 1

    return {
        "total": len(records),
        "ok": len(successes),
        "failed": len(records) - len(successes),
        "accepted": accepted,
        "excluded_by_reason": dict(sorted(excluded_by_reason.items())),
        "by_status": dict(sorted(by_status.items())),
        "by_category": dict(sorted(by_category.items())),
        "attempts": dict(sorted(attempts.items())),
        "locconf": locconf,
        "held": sum(1 for record in successes if record.get("isHeld")),
        "families": dict(sorted(families.items())),
    }


def zone_asset_path(zone_id: str, assets_root: Path) -> Path | None:
    """把 MapLocator 的 zone_id 映射回底图资产路径（与 zone 命名规则互逆）。

    - `<Parent>_Base`  → `<root>/<Parent>/Base.png`
    - `<Parent>_L<level>_<tier>` → `<root>/<Parent>/Lv<level:03d>Tier<tier>.png`
    - 其它（如 OMVBase01）→ 任意子目录下的 `<zone_id>.png`
    """
    base_match = _BASE_RE.fullmatch(zone_id)
    if base_match:
        candidate = assets_root / base_match.group(1) / "Base.png"
        return candidate if candidate.exists() else None

    tier_match = _TIER_RE.fullmatch(zone_id)
    if tier_match:
        region, level, tier = tier_match.group(1), int(tier_match.group(2)), tier_match.group(3)
        candidate = assets_root / region / f"Lv{level:03d}Tier{tier}.png"
        return candidate if candidate.exists() else None

    hits = sorted(assets_root.glob(f"*/{zone_id}.png"))
    return hits[0] if hits else None
