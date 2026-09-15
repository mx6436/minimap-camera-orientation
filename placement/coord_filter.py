"""标注坐标一致性过滤：把文件名标注与定位记录放到同一资产帧上比对。

样本文件名带标注（`Wuling_L9_395_x578.0_y1666.9_r344.7.png`）：地图名 + 该地图坐标系
下的 (x, y)。标注与定位记录描述的应是同一张底图上的同一位置。

判据（单次、无图像匹配）：标注与定位记录的 zone 不同即判不一致（不是同一资产帧，
坐标不可比）；zone 相同时，`max(|Δx|, |Δy|)` 超过 `MAX_DELTA` 判不一致。两者都不
进入数据集。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from placement.placement import Placement

# 阈值单位 = 资产帧的像素（MaaEnd 上游判据为「5 个单位」）
MAX_DELTA = 5.0

REASON_DELTA = "coord_delta"
REASON_ZONE = "coord_zone"

ANNOTATION_RE = re.compile(
    r"^(?P<zone>.+?)_x(?P<x>-?[\d.]+)_y(?P<y>-?[\d.]+)_r(?P<r>-?[\d.]+)\.png$"
)


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


def parse_annotation(name: str) -> Annotation:
    """文件名 -> 标注；不带 `_x…_y…_r…` 的名字报错（管线里不该出现）。"""
    match = ANNOTATION_RE.match(name)
    if match is None:
        raise ValueError(f"cannot parse annotation from filename: {name}")
    return Annotation(match.group("zone"), float(match.group("x")), float(match.group("y")))


def evaluate(annotation: Annotation, placement: Placement) -> Decision:
    """标注与底图定位是否一致；placement 取自同一条定位记录。"""
    if annotation.zone != placement.zone:
        return Decision(False, REASON_ZONE, None)
    delta = max(abs(annotation.x - placement.x), abs(annotation.y - placement.y))
    return Decision(delta <= MAX_DELTA, "" if delta <= MAX_DELTA else REASON_DELTA, delta)
