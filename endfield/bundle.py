"""交付 bundle：交付角色词汇、manifest schema 与 manifest ↔ 工件一致性自检。

交付角色决定图文件名（跨仓契约，MaaEnd 交付布局）与分类器角色的输入模式；通道数仍由
`endfield/run_record.py` 单点定义，本模块不另立一份。manifest 字段 schema 属本仓；
验收剖面的取值（定义哈希、ORT 版本、容差剖面、fixture 清单）由调用方构造
`ManifestProfile` 注入——本模块不 import `conformance`，也不 import `preprocess`
（后者的 torch 依赖不该进入校验路径）。
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from endfield.findings import Finding
from endfield.run_record import InputMode, RunRecord, input_channels

MANIFEST_NAME = "manifest.json"
SCHEMA_VERSION = 1


class DeliveryRole(StrEnum):
    """交付 bundle 中一张图的角色名；决定文件名与（分类器角色的）输入模式。"""

    PREPROCESS = "preprocess"
    POLAR = "polar"
    POLAR_WITH_REF = "polar_with_ref"


@dataclass(frozen=True)
class GraphSpec:
    """一个交付角色的图契约：文件名、输入模式与输出角色映射。"""

    file: str
    input_mode: InputMode | None = None
    outputs: Mapping[str, str] | None = None


_SPECS: dict[DeliveryRole, GraphSpec] = {
    DeliveryRole.PREPROCESS: GraphSpec(
        file="preprocess.onnx",
        outputs={"observed": "observed", "reference": "reference"},
    ),
    DeliveryRole.POLAR: GraphSpec(file="polar.onnx", input_mode=InputMode.POLAR),
    DeliveryRole.POLAR_WITH_REF: GraphSpec(file="polar_with_ref.onnx", input_mode=InputMode.REF),
}


@dataclass(frozen=True)
class ManifestProfile:
    """manifest 记录的验收剖面：同源凭据与验证方复验所需的取值。"""

    definition_hash: str
    ort_version: str
    tolerances: Mapping[str, float]
    fixtures: Sequence[str]


def roles() -> tuple[DeliveryRole, ...]:
    """全部交付角色，按交付布局顺序。"""
    return tuple(DeliveryRole)


def classifier_roles() -> tuple[DeliveryRole, ...]:
    """带 run 的交付角色：分类器图由 run 导出，preprocess 由定义模块导出。"""
    return tuple(role for role in DeliveryRole if _SPECS[role].input_mode is not None)


def spec(role: DeliveryRole | str) -> GraphSpec:
    """交付角色 -> 图契约；未知角色给出含合法取值的 ValueError。"""
    try:
        return _SPECS[DeliveryRole(role)]
    except ValueError:
        expected = ", ".join(item.value for item in DeliveryRole)
        raise ValueError(f"unknown delivery role: {role!r}; expected one of {expected}") from None


def graph_file(role: DeliveryRole | str) -> str:
    """交付角色 -> 交付文件名（跨仓契约）。"""
    return spec(role).file


def input_mode(role: DeliveryRole | str) -> InputMode:
    """分类器交付角色的输入模式；preprocess 没有输入模式，调用即报错。"""
    mode = spec(role).input_mode
    if mode is None:
        raise ValueError(f"delivery role {DeliveryRole(role).value!r} has no input mode")
    return mode


def role_for_mode(mode: InputMode | str) -> DeliveryRole:
    """输入模式 -> 分类器交付角色（交付文件名的反向查询）。"""
    parsed = InputMode.parse(mode)
    for role in classifier_roles():
        if _SPECS[role].input_mode is parsed:
            return role
    raise ValueError(f"no delivery role for input mode {parsed.value!r}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(bundle_dir: Path) -> dict | None:
    """读 manifest.json；不存在返回 None；存在但不可解析即报错（产物损坏）。"""
    path = Path(bundle_dir) / MANIFEST_NAME
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path}: manifest 读取失败：{exc}") from exc


def _bundle_relative(path: Path, bundle_dir: Path) -> str:
    """bundle 目录相对路径（manifest 的 run_dir 口径；统一 posix 分隔符）。"""
    return Path(os.path.relpath(str(path), str(bundle_dir))).as_posix()


def build_manifest(
    out_dir: Path,
    runs: Mapping[DeliveryRole, Path],
    load_run: Callable[[Path], tuple[RunRecord, Mapping[str, Any]]],
    profile: ManifestProfile,
    *,
    git_commit: str,
) -> dict:
    """已导出的图 + 各分类器角色的 run -> manifest。

    run 产物的布局由调用方经 `load_run` 提供：本模块不持有 run 目录的形状。
    """
    out_dir = Path(out_dir)
    missing = sorted(role.value for role in classifier_roles() if role not in runs)
    if missing:
        raise ValueError(f"manifest needs a run for every classifier role; missing={missing}")

    preprocess_spec = _SPECS[DeliveryRole.PREPROCESS]
    graphs: dict[str, dict] = {
        DeliveryRole.PREPROCESS.value: {
            "file": preprocess_spec.file,
            "sha256": sha256_file(out_dir / preprocess_spec.file),
            "outputs": dict(preprocess_spec.outputs or {}),
        }
    }
    for role in classifier_roles():
        run_dir = Path(runs[role])
        record, summary = load_run(run_dir)
        mode = record.input_mode
        graphs[role.value] = {
            "file": _SPECS[role].file,
            "sha256": sha256_file(out_dir / _SPECS[role].file),
            "run_dir": _bundle_relative(run_dir, out_dir),
            "input_mode": mode,
            "input_channels": input_channels(mode),
            "metrics": {
                "best_epoch": int(summary["epoch"]),
                "val_count": int(summary["val_count"]),
                "val_rms_error_deg": round(float(summary["val_rms_error"]), 6),
                "val_expected_abs_error_deg": round(float(summary["val_expected_abs_error"]), 6),
            },
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "git_commit": git_commit,
        "definition_hash": profile.definition_hash,
        "ort_version": profile.ort_version,
        "graphs": graphs,
        "fixtures": list(profile.fixtures),
        "tolerances": dict(profile.tolerances),
    }


def check_structure(
    bundle_dir: Path,
    profile: ManifestProfile,
    *,
    require_manifest: bool = True,
) -> tuple[dict | None, list[Finding]]:
    """manifest ↔ 文件哈希 ↔ 图 metadata ↔ ORT 可加载 的一致性自检。

    `require_manifest=False` 允许无 manifest 的草稿 bundle：只报 warning 并跳过全部
    manifest 相关断言（`verify-artifact` 的草稿路径）；导出侧要求 manifest 必须存在。
    返回（读到的 manifest，结论）；manifest 缺失或不可解析时为 None。
    """
    bundle_dir = Path(bundle_dir)
    try:
        manifest = read_manifest(bundle_dir)
    except ValueError as exc:
        return None, [Finding("error", "manifest_invalid", str(exc))]
    if manifest is None:
        level = "error" if require_manifest else "warning"
        return None, [
            Finding(
                level,
                "manifest_missing",
                f"无 {MANIFEST_NAME}：{bundle_dir}（按草稿 bundle 校验）",
            )
        ]

    findings: list[Finding] = []

    def error(code: str, message: str) -> None:
        findings.append(Finding("error", code, message))

    if manifest.get("schema_version") != SCHEMA_VERSION:
        error(
            "schema_version",
            f"schema_version={manifest.get('schema_version')!r} != {SCHEMA_VERSION}",
        )
    if manifest.get("ort_version") != profile.ort_version:
        error(
            "manifest_ort_version",
            f"manifest ort_version={manifest.get('ort_version')!r} != {profile.ort_version}",
        )
    if not manifest.get("git_commit"):
        error("git_commit_missing", "git_commit 缺失")
    if manifest.get("definition_hash") != profile.definition_hash:
        error("definition_hash", "definition_hash 与当前定义模块不一致（需重导出）")

    graphs = manifest.get("graphs")
    if not isinstance(graphs, Mapping):
        error("graphs_missing", "graphs 缺失或不是对象")
    else:
        declared_roles = sorted(str(role) for role in graphs)
        expected_roles = sorted(role.value for role in DeliveryRole)
        if declared_roles != expected_roles:
            error("graphs_roles", f"graphs={declared_roles} != {expected_roles}")
        for role in DeliveryRole:
            graph_spec = graphs.get(role.value)
            if not isinstance(graph_spec, Mapping):
                error("graph_spec_missing", f"graphs.{role.value} 缺失")
                continue
            findings.extend(_check_graph(bundle_dir, manifest, profile, role, graph_spec))

    fixtures = manifest.get("fixtures")
    if not fixtures:
        error("fixtures_missing", "fixtures 缺失")
    else:
        known = set(profile.fixtures)
        unknown = sorted(str(name) for name in fixtures if name not in known)
        if unknown:
            error("fixture_unknown", f"fixtures 引用未知场景：{unknown}")

    tolerances = manifest.get("tolerances") or {}
    missing_profiles = sorted(set(profile.tolerances) - set(tolerances))
    if missing_profiles:
        error("tolerances_missing", f"tolerances 缺少剖面：{missing_profiles}")
    return manifest, findings


def _graph_metadata(path: Path) -> dict[str, str]:
    import onnx

    model = onnx.load(str(path))
    return {prop.key: prop.value for prop in model.metadata_props}


def _input_channels(model: Any) -> int:
    dims = model.graph.input[0].type.tensor_type.shape.dim
    return int(dims[3].dim_value)


def _check_graph(
    bundle_dir: Path,
    manifest: Mapping[str, Any],
    profile: ManifestProfile,
    role: DeliveryRole,
    declared: Mapping[str, Any],
) -> list[Finding]:
    """单张交付图：文件名、sha256、metadata 互证与 ORT 可加载。"""
    import onnx
    import onnxruntime as ort

    findings: list[Finding] = []

    def error(code: str, message: str) -> None:
        findings.append(Finding("error", code, message))

    graph_spec = _SPECS[role]
    file_name = graph_spec.file
    if declared.get("file") != file_name:
        error("graph_file", f"graphs.{role.value}.file={declared.get('file')!r} != {file_name!r}")
    path = bundle_dir / file_name
    if not path.is_file():
        error("graph_missing", f"缺少图文件：{file_name}")
        return findings
    if declared.get("sha256") != sha256_file(path):
        error("graph_sha256", f"{file_name}: sha256 与 manifest 不一致")
    try:
        model = onnx.load(str(path))
    except Exception as exc:  # noqa: BLE001 - 非法图给出结论而不是崩溃
        error("graph_invalid", f"{file_name}: onnx 加载失败：{exc}")
        return findings

    metadata = _graph_metadata(path)
    output_names = {value.name for value in model.graph.output}

    if role is DeliveryRole.PREPROCESS:
        if metadata.get("definition_hash") != manifest.get("definition_hash"):
            error(
                "graph_definition_hash",
                f"{file_name}: metadata definition_hash 与 manifest 不一致",
            )
        outputs = declared.get("outputs")
        if not isinstance(outputs, Mapping):
            error("outputs_missing", f"graphs.{role.value}.outputs 缺失")
        else:
            unknown = sorted(set(outputs.values()) - output_names)
            if unknown:
                error(
                    "outputs_unknown",
                    f"{file_name}: outputs 声明了图中不存在的输出 {unknown}",
                )
    else:
        mode = graph_spec.input_mode
        assert mode is not None  # 分类器角色恒有输入模式（表内不变量）
        if declared.get("input_mode") != mode.value:
            error(
                "graph_input_mode",
                f"graphs.{role.value}.input_mode={declared.get('input_mode')!r} != {mode.value!r}",
            )
        if metadata.get("input_mode") != mode.value:
            error(
                "graph_metadata_input_mode",
                f"{file_name}: 图 input_mode={metadata.get('input_mode')!r} != {mode.value!r}",
            )
        if metadata.get("git_commit") != manifest.get("git_commit"):
            error("graph_git_commit", f"{file_name}: 图 git_commit 与 manifest 不一致（旧图混入）")
        expected_channels = input_channels(mode)
        if declared.get("input_channels") != expected_channels:
            error("graph_channels", f"graphs.{role.value}.input_channels != {expected_channels}")
        if _input_channels(model) != expected_channels:
            error("graph_channels", f"{file_name}: 输入通道 != {expected_channels}")
        metrics = declared.get("metrics")
        if not isinstance(metrics, Mapping):
            error("graph_metrics_missing", f"graphs.{role.value}.metrics 缺失")
        else:
            findings.extend(_check_metrics(file_name, metadata, metrics))

    try:
        ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    except Exception as exc:  # noqa: BLE001 - 加载失败即不可交付
        error("graph_unloadable", f"{file_name}: ORT {profile.ort_version} 加载失败：{exc}")
    return findings


def _check_metrics(
    file_name: str, metadata: Mapping[str, str], metrics: Mapping[str, Any]
) -> list[Finding]:
    """manifest 指标与图 metadata 的格式化取值互证（错一位即判不一致）。"""
    findings: list[Finding] = []
    for key in ("val_rms_error_deg", "val_expected_abs_error_deg"):
        value = metrics.get(key)
        try:
            formatted = f"{float(value):.6f}"
        except (TypeError, ValueError):
            findings.append(
                Finding(
                    "error",
                    "graph_metric_invalid",
                    f"{file_name}: manifest 指标 {key}={value!r} 非法",
                )
            )
            continue
        if metadata.get(key) != formatted:
            findings.append(
                Finding(
                    "error",
                    "graph_metric_mismatch",
                    f"{file_name}: {key} 与 manifest 指标不一致",
                )
            )
    return findings
