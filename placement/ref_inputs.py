"""ref 输入侧：定位记录 + 工作台资产 -> 数据准备的输入侧值。

入选门、资产存在性与坐标一致性过滤都在这里判定，被剔除的样本连同原因随值返回；
渲染条带对的适配器也由这里给出（`RefInputs.renderer`）——它是「底图定位 -> 条带对」
在数据准备上的那一半，张量编码仍在 `endfield` 侧（ADR 0004、ADR 0006）。
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from endfield import prepare
from placement import coord_filter, workspace
from placement.placement import Placement, accept
from placement.records import load_records
from placement.sample import ReferenceSampler

ASSET_MISSING = "asset_missing"
NO_LOCATE_RECORD = "no_locate_record"


@dataclass(frozen=True)
class RefInputs:
    """ref 输入侧解析结果：输入侧值 + 渲染所需的定位映射与已校验的资产根。"""

    inputs: prepare.PrepareInputs
    placements: Mapping[str, Placement]
    accepted: Sequence[str]
    assets_root: Path

    def renderer(self, sampler: ReferenceSampler) -> prepare.RenderFn:
        """渲染适配器：观测 ROI + 该样本的底图定位 -> 条带对。"""

        def render(name: str, observed_roi: np.ndarray) -> prepare.StripPair:
            observed, reference = sampler.strips(observed_roi, self.placements[name])
            return prepare.StripPair(observed, reference)

        return render


def resolve(
    samples: Mapping[str, Path],
    *,
    locate_path: Path,
    assets_root: Path,
) -> RefInputs:
    """样本并集 + 定位产物 -> ref 输入侧值。

    `samples` 是原始目录并集；未入选（定位失败 / held / 低分 / 缺资产 / 坐标不一致）
    的样本进 `skipped`，其余按名单排序进产物。资产根经校验后随值返回。
    """
    if not samples:
        raise SystemExit("no raw png samples in the raw dirs")
    assets = workspace.require_assets(assets_root)
    accepted, skipped = _accepted_records(locate_path)
    accepted = {name: record for name, record in accepted.items() if name in samples}
    skipped = {name: reason for name, reason in skipped.items() if name in samples}
    for name in samples:
        if name not in accepted:
            skipped.setdefault(name, NO_LOCATE_RECORD)
    resolved = _resolve_placements(accepted, assets, skipped)
    resolved, rejected = _filter_coord_consistent(resolved)
    skipped.update(rejected)
    return RefInputs(
        inputs=prepare.PrepareInputs(
            sources=[(name, samples[name]) for name, _ in resolved],
            fingerprint=input_entries(resolved),
            skipped=skipped,
            provenance=workspace.provenance(assets),
        ),
        placements=dict(resolved),
        accepted=tuple(sorted(accepted)),
        assets_root=assets,
    )


def input_entries(resolved: Sequence[tuple[str, Placement]]) -> list[str]:
    """落戳指纹条目：样本名 + 它消费的底图定位（zone/x/y/scale）。

    其他定位字段（latencyMs 等）不进指纹：重跑 locate-dataset 只刷新时间戳时不该触发
    数据重算；资产路径由 zone 决定，不另记。
    """
    return [
        json.dumps(
            [name, placement.zone, placement.x, placement.y, placement.scale],
            ensure_ascii=True,
            separators=(",", ":"),
        )
        for name, placement in resolved
    ]


def _accepted_records(locate_path: Path) -> tuple[dict[str, dict], dict[str, str]]:
    """定位产物 ->（accepted 记录表, name -> 跳过原因）。"""
    records = load_records(locate_path)
    if not records:
        raise SystemExit(f"no MapLocator records: {locate_path} (run locate-dataset first)")
    accepted: dict[str, dict] = {}
    skipped: dict[str, str] = {}
    for name, record in records.items():
        ok, reason = accept(record)
        if ok:
            accepted[name] = record
        else:
            skipped[name] = reason
    if not accepted:
        raise SystemExit(f"no accepted MapLocator records: {locate_path}")
    return accepted, skipped


def _resolve_placements(
    accepted: Mapping[str, dict], assets_root: Path, skipped: dict[str, str]
) -> list[tuple[str, Placement]]:
    """accepted 记录 -> 可生成样本 (name, 底图定位)；缺资产的计入 skipped。"""
    resolved: list[tuple[str, Placement]] = []
    for name in sorted(accepted):
        placement = Placement.from_record(accepted[name])
        if placement.asset_path(assets_root) is None:
            skipped[name] = ASSET_MISSING
            continue
        resolved.append((name, placement))
    return resolved


def _filter_coord_consistent(
    resolved: list[tuple[str, Placement]],
) -> tuple[list[tuple[str, Placement]], dict[str, str]]:
    """标注坐标一致性过滤：返回（保留样本, name -> 拒绝原因）。"""
    kept: list[tuple[str, Placement]] = []
    rejected: dict[str, str] = {}
    for item in resolved:
        name, placement = item
        decision = coord_filter.evaluate(coord_filter.parse_annotation(name), placement)
        if decision.keep:
            kept.append(item)
        else:
            rejected[name] = decision.reason
    return kept, rejected
