"""交付 bundle 导出：`{preprocess,polar,polar_with_ref}.onnx` + `manifest.json`。

两个 run 的分类器图与定义模块导出的前处理图一次成型，对应 MaaEnd 交付布局
`assets/resource/model/map/cameraorientation/`（拷入步骤见 README「拷入 MaaEnd」）：

    uv run export-artifact --out runs/<name>/bundle \
        --polar-run runs/<polar_run> --ref-run runs/<ref_run>

- `preprocess.onnx` 由定义模块 `endfield/preprocess.py` 导出；
- `polar.onnx` / `polar_with_ref.onnx` 由各自 run 的 `best.pt` 导出（`export-onnx`）；
- `manifest.json` 记录 git commit、definition hash、模型指标、fixture 清单与容差剖面，
  并给每张图记 sha256，供 conformance 复验与「bundle 内版本一致」判定。

导出后跑结构自检（manifest ↔ 图 metadata ↔ 文件哈希互证、ORT 1.19.2 可加载），
失败退出码 1（图与 manifest 仍落盘，便于定位）。数值 conformance 证据用
`uv run verify-artifact --bundle <out>` 落报告（#30）。

重复导出确定性：同一 run + 同一定义 + 同一工具链 → 图与 manifest 逐字节一致。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path

from cli.export_onnx import export as export_classifier
from cli.export_onnx import git_commit
from endfield import conformance as cf
from endfield import preprocess, run_record

SCHEMA_VERSION = 1
GRAPH_FILES = {
    "preprocess": "preprocess.onnx",
    "polar": "polar.onnx",
    "polar_with_ref": "polar_with_ref.onnx",
}
# 交付角色 → run record 的 input_mode：polar 模式 3 通道、ref 模式 7 通道
ROLE_MODES = {"polar": "polar", "polar_with_ref": "ref"}

PREPROCESS_OUTPUTS = {"observed": "observed", "reference": "reference"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bundle_relative(path: Path, bundle_dir: Path) -> str:
    """bundle 目录相对路径（manifest 的 run_dir 口径，见 README；统一 posix 分隔符）。"""
    return Path(os.path.relpath(str(path), str(bundle_dir))).as_posix()


def graph_metadata(path: Path) -> dict[str, str]:
    import onnx

    model = onnx.load(str(path))
    return {prop.key: prop.value for prop in model.metadata_props}


def _input_channels(model: object) -> int:
    dims = model.graph.input[0].type.tensor_type.shape.dim  # type: ignore[attr-defined]
    return int(dims[3].dim_value)


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path}: 读取失败：{exc}") from exc


def build_manifest(out_dir: Path, polar_run: Path, ref_run: Path) -> dict:
    """按已导出的图与 run 产物组装 manifest（不含数值 conformance）。"""
    out_dir = Path(out_dir)
    graphs: dict[str, dict] = {
        "preprocess": {
            "file": GRAPH_FILES["preprocess"],
            "sha256": sha256_file(out_dir / GRAPH_FILES["preprocess"]),
            "outputs": dict(PREPROCESS_OUTPUTS),
        }
    }
    for role, run_dir in (("polar", Path(polar_run)), ("polar_with_ref", Path(ref_run))):
        record = run_record.read(run_dir)
        summary = _read_json(run_dir / "summary.json")
        mode = record.input_mode
        graphs[role] = {
            "file": GRAPH_FILES[role],
            "sha256": sha256_file(out_dir / GRAPH_FILES[role]),
            "run_dir": bundle_relative(run_dir, out_dir),
            "input_mode": mode,
            "input_channels": run_record.input_channels(mode),
            "metrics": {
                "best_epoch": int(summary["epoch"]),
                "val_count": int(summary["val_count"]),
                "val_rms_error_deg": round(float(summary["val_rms_error"]), 6),
                "val_expected_abs_error_deg": round(float(summary["val_expected_abs_error"]), 6),
            },
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "git_commit": git_commit(),
        "definition_hash": preprocess.definition_hash(),
        "ort_version": cf.ORT_VERSION,
        "graphs": graphs,
        "fixtures": [scenario.name for scenario in cf.builtin_scenarios()],
        "tolerances": dict(cf.DEFAULT_TOLERANCES),
    }


def export_bundle(out_dir: Path, polar_run: Path, ref_run: Path) -> Path:
    """导出三图并写 manifest.json；目录已存在时原地覆盖（重复导出确定性）。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    preprocess.export_onnx(out_dir / GRAPH_FILES["preprocess"])
    print(
        f"exported: {out_dir / GRAPH_FILES['preprocess']} "
        f"(definition_hash={preprocess.definition_hash()[:12]})"
    )
    for role, run_dir in (("polar", Path(polar_run)), ("polar_with_ref", Path(ref_run))):
        export_classifier(run_dir / "best.pt", out_dir / GRAPH_FILES[role])
    manifest = build_manifest(out_dir, Path(polar_run), Path(ref_run))
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return out_dir


def check_bundle(bundle_dir: Path) -> list[str]:
    """结构自检：manifest ↔ 文件哈希 ↔ 图 metadata ↔ 角色契约；返回错误清单。"""
    bundle_dir = Path(bundle_dir)
    manifest_path = bundle_dir / "manifest.json"
    if not manifest_path.is_file():
        return [f"缺少 manifest.json：{manifest_path}"]
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"manifest.json 读取失败：{exc}"]

    errors: list[str] = []
    if manifest.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version={manifest.get('schema_version')!r} != {SCHEMA_VERSION}")
    if manifest.get("ort_version") != cf.ORT_VERSION:
        errors.append(f"ort_version={manifest.get('ort_version')!r} != {cf.ORT_VERSION}")
    if not manifest.get("git_commit"):
        errors.append("git_commit 缺失")
    if manifest.get("definition_hash") != preprocess.definition_hash():
        errors.append("definition_hash 与当前定义模块不一致（需重导出）")

    graphs = manifest.get("graphs")
    if not isinstance(graphs, Mapping):
        errors.append("graphs 缺失或不是对象")
    else:
        if set(graphs) != set(GRAPH_FILES):
            errors.append(f"graphs={sorted(graphs)} != {sorted(GRAPH_FILES)}")
        for role, file_name in GRAPH_FILES.items():
            spec = graphs.get(role)
            if not isinstance(spec, Mapping):
                errors.append(f"graphs.{role} 缺失")
                continue
            errors.extend(_check_graph(bundle_dir, manifest, role, file_name, spec))

    fixtures = manifest.get("fixtures")
    known = cf.scenario_map()
    if not fixtures:
        errors.append("fixtures 缺失")
    else:
        unknown = sorted(name for name in fixtures if name not in known)
        if unknown:
            errors.append(f"fixtures 引用未知场景：{unknown}")
    tolerances = manifest.get("tolerances") or {}
    missing = sorted(set(cf.DEFAULT_TOLERANCES) - set(tolerances))
    if missing:
        errors.append(f"tolerances 缺少剖面：{missing}")
    return errors


def _check_graph(
    bundle_dir: Path, manifest: Mapping, role: str, file_name: str, spec: Mapping
) -> list[str]:
    import onnx
    import onnxruntime as ort

    errors: list[str] = []
    if spec.get("file") != file_name:
        errors.append(f"graphs.{role}.file={spec.get('file')!r} != {file_name!r}")
    path = bundle_dir / file_name
    if not path.is_file():
        errors.append(f"缺少图文件：{file_name}")
        return errors
    if spec.get("sha256") != sha256_file(path):
        errors.append(f"{file_name}: sha256 与 manifest 不一致")
    try:
        model = onnx.load(str(path))
    except Exception as exc:  # noqa: BLE001 - 非法图给出结论而不是崩溃
        errors.append(f"{file_name}: onnx 加载失败：{exc}")
        return errors
    metadata = {prop.key: prop.value for prop in model.metadata_props}
    output_names = {value.name for value in model.graph.output}

    if role == "preprocess":
        if metadata.get("definition_hash") != manifest.get("definition_hash"):
            errors.append("preprocess.onnx: metadata definition_hash 与 manifest 不一致")
        declared = spec.get("outputs")
        if not isinstance(declared, Mapping):
            errors.append("graphs.preprocess.outputs 缺失")
        else:
            unknown = sorted(set(declared.values()) - output_names)
            if unknown:
                errors.append(f"preprocess.onnx: outputs 声明了图中不存在的输出 {unknown}")
    else:
        mode = ROLE_MODES[role]
        if spec.get("input_mode") != mode:
            errors.append(f"graphs.{role}.input_mode={spec.get('input_mode')!r} != {mode!r}")
        if metadata.get("input_mode") != mode:
            errors.append(f"{file_name}: 图 input_mode={metadata.get('input_mode')!r} != {mode!r}")
        if metadata.get("git_commit") != manifest.get("git_commit"):
            errors.append(f"{file_name}: 图 git_commit 与 manifest 不一致（旧图混入）")
        expected_channels = run_record.input_channels(mode)
        if spec.get("input_channels") != expected_channels:
            errors.append(f"graphs.{role}.input_channels != {expected_channels}")
        if _input_channels(model) != expected_channels:
            errors.append(f"{file_name}: 输入通道 != {expected_channels}")
        metrics = spec.get("metrics")
        if not isinstance(metrics, Mapping):
            errors.append(f"graphs.{role}.metrics 缺失")
        else:
            errors.extend(_check_metrics(file_name, metadata, metrics))

    try:
        ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    except Exception as exc:  # noqa: BLE001 - 加载失败即不可交付
        errors.append(f"{file_name}: ORT {cf.ORT_VERSION} 加载失败：{exc}")
    return errors


def _check_metrics(file_name: str, metadata: Mapping, metrics: Mapping) -> list[str]:
    errors: list[str] = []
    for key in ("val_rms_error_deg", "val_expected_abs_error_deg"):
        value = metrics.get(key)
        try:
            formatted = f"{float(value):.6f}"
        except (TypeError, ValueError):
            errors.append(f"{file_name}: manifest 指标 {key}={value!r} 非法")
            continue
        if metadata.get(key) != formatted:
            errors.append(f"{file_name}: {key} 与 manifest 指标不一致")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True, help="bundle 输出目录")
    parser.add_argument(
        "--polar-run",
        type=Path,
        required=True,
        help="polar 分类器 run 目录（含 best.pt / record.json / summary.json）",
    )
    parser.add_argument(
        "--ref-run",
        type=Path,
        required=True,
        help="polar_with_ref 分类器 run 目录（含 best.pt / record.json / summary.json）",
    )
    args = parser.parse_args(argv)

    bundle = export_bundle(args.out, args.polar_run, args.ref_run)
    errors = check_bundle(bundle)
    for error in errors:
        print(f"ERROR: {error}")
    if errors:
        print(f"result: FAIL（{len(errors)} 项结构自检错误；图与 manifest 保留在 {bundle}）")
        return 1
    manifest = _read_json(bundle / "manifest.json")
    print(f"bundle: {bundle}")
    print(f"result: PASS（三图 + manifest；definition_hash={manifest['definition_hash'][:12]}）")
    print(f"conformance 证据：uv run verify-artifact --bundle {bundle}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
