"""标注坐标一致性过滤：把文件名标注与定位记录换算到同一资产帧后比对。

数据集由两批标注合并：中文区域文件夹批次用 MapLocator zone 命名（`ValleyIV_Base`、
`Wuling_L9_395` …），`maptracker/` 批次用 MapTracker 地图名命名（`map01_lv006`、
`map02_lv005` …）。两套命名是同一地图体系的不同粒度，坐标换算关系全在上游既有约定里
（本模块不拟合任何参数）：

- region ↔ map 前缀：MaaEnd `agent/go-service/maptracker/compatible/convert.go` 的
  `compatibleRegionMapPrefix`（map01↔ValleyIV、map02↔Wuling）；
- level 图像素 → ZmdMap canvas 单位：MaaEnd `tools/map_tracker/map_generator.py` 的
  `SCALE_MAP_FACTOR = 0.1625`；
- level 矩形（canvas 单位）：MaaEnd `assets/data/ZmdMap/<prefix>_layout.json`；
- canvas → MapLocator Base.png：`Base.png` 尺寸 / canvas 尺寸（本地镜像见
  `local/maplocator/data/ZmdMap/`，来源与刷新见 docs/maplocator-workspace.md）。

判据（单次、无图像匹配）：把标注换算到定位记录所在资产帧后，`max(|Δx|, |Δy|)` 超过
`MAX_DELTA` 即判不一致；标注与定位的 zone/区域不同（除「MapTracker 名 ↔ 同区域
`<Region>_Base`」这一种可换算情形）也判不一致；命名不在支持范围内（如其他区域的
MapTracker 名）判为无法核对，同样不进入数据集。
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from endfield.polar import imread_png

# 阈值单位 = 换算后资产帧的像素（MaaEnd 上游判据为「5 个单位」）
MAX_DELTA = 5.0
# MaaEnd tools/map_tracker/map_generator.py：level 图像素 → canvas 单位
SCALE_MAP_FACTOR = 0.1625
# convert.go::compatibleRegionMapPrefix 的本数据集子集
PREFIX_REGION: dict[str, str] = {"map01": "ValleyIV", "map02": "Wuling"}

REASON_DELTA = "coord_delta"
REASON_ZONE = "coord_zone"
REASON_UNSUPPORTED = "coord_unsupported"

ANNOTATION_RE = re.compile(
    r"^(?P<zone>.+?)_x(?P<x>-?[\d.]+)_y(?P<y>-?[\d.]+)_r(?P<r>-?[\d.]+)\.png$"
)
TRACKER_RE = re.compile(r"^(?P<prefix>[a-z]+\d*)_lv(?P<level>\d+)$")


@dataclass(frozen=True)
class Annotation:
    """文件名标注：地图名 + 该地图坐标系下的 (x, y)。"""

    zone: str
    x: float
    y: float


@dataclass(frozen=True)
class Decision:
    """keep=False 时 reason 是可直接进 skipped 统计的短标记。"""

    keep: bool
    reason: str
    delta: float | None


@dataclass(frozen=True)
class FilterData:
    """换算所需的上游数据：level 矩形（canvas 单位）与 canvas→Base.png 比例。"""

    levels: Mapping[str, Mapping[str, tuple[float, float]]]
    scales: Mapping[str, float]

    def base_position(self, level_name: str, x: float, y: float) -> tuple[float, float] | None:
        """MapTracker level 图像素坐标 -> 同区域 Base.png 像素坐标；不支持则 None。"""
        match = TRACKER_RE.match(level_name)
        if match is None or match.group("prefix") not in self.levels:
            return None
        prefix = match.group("prefix")
        rect = self.levels[prefix].get(level_name)
        if rect is None:
            return None
        scale = self.scales[prefix]
        return (
            rect[0] * scale + x * scale / SCALE_MAP_FACTOR,
            rect[1] * scale + y * scale / SCALE_MAP_FACTOR,
        )


def needs_conversion(zone: str) -> bool:
    """zone 是否需要在两个坐标系间换算（MapTracker 地图名）。"""
    return TRACKER_RE.match(zone) is not None


def parse_annotation(name: str) -> Annotation:
    """文件名 -> 标注；不带 `_x…_y…_r…` 的名字报错（管线里不该出现）。"""
    match = ANNOTATION_RE.match(name)
    if match is None:
        raise ValueError(f"cannot parse annotation from filename: {name}")
    return Annotation(match.group("zone"), float(match.group("x")), float(match.group("y")))


def load_filter_data(zmdmap_root: Path, assets_root: Path) -> FilterData:
    """读取各区域的 layout 与 Base.png 尺寸；缺任一文件即报错（数据未镜像/资产缺失）。"""
    levels: dict[str, dict[str, tuple[float, float]]] = {}
    scales: dict[str, float] = {}
    for prefix, region in PREFIX_REGION.items():
        layout_path = zmdmap_root / f"{prefix}_layout.json"
        if not layout_path.is_file():
            raise FileNotFoundError(
                f"missing ZmdMap layout for {prefix}: {layout_path} "
                "(mirror MaaEnd assets/data/ZmdMap/<prefix>_layout.json)"
            )
        base_path = assets_root / region / "Base.png"
        if not base_path.is_file():
            raise FileNotFoundError(f"missing MapLocator base image: {base_path}")
        layout = json.loads(layout_path.read_text(encoding="utf-8"))
        canvas_width = float(layout["canvas_width"])
        levels[prefix] = {
            name: (float(rect["x"]), float(rect["y"])) for name, rect in layout["levels"].items()
        }
        scales[prefix] = imread_png(base_path).shape[1] / canvas_width
    return FilterData(levels=levels, scales=scales)


def _compare(x: float, y: float, record: Mapping[str, Any]) -> Decision:
    delta = max(abs(x - float(record["x"])), abs(y - float(record["y"])))
    return Decision(delta <= MAX_DELTA, "" if delta <= MAX_DELTA else REASON_DELTA, delta)


def evaluate(
    annotation: Annotation, record: Mapping[str, Any], data: FilterData | None
) -> Decision:
    """标注与定位记录是否一致；record 是 locate.jsonl 的一条记录。

    同 zone 比较不需要 `data`；标注是 MapTracker 名时必须给出 `data`（换算依赖）。
    """
    locator_zone = str(record.get("zone", ""))
    if annotation.zone == locator_zone:
        # 同一 zone = 同一资产帧，直接比坐标
        return _compare(annotation.x, annotation.y, record)

    match = TRACKER_RE.match(annotation.zone)
    if match is None:
        # 标注本身也是 MapLocator zone 名，但与定位结果不同 zone：无法直接比
        return Decision(False, REASON_ZONE, None)
    if data is None:
        raise ValueError(f"tracker annotation requires FilterData: {annotation.zone}")
    prefix = match.group("prefix")
    region = PREFIX_REGION.get(prefix)
    if region is None:
        return Decision(False, REASON_UNSUPPORTED, None)
    if locator_zone != f"{region}_Base":
        return Decision(False, REASON_ZONE, None)
    base = data.base_position(annotation.zone, annotation.x, annotation.y)
    if base is None:
        return Decision(False, REASON_UNSUPPORTED, None)
    return _compare(base[0], base[1], record)
