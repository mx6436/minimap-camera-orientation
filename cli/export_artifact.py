"""交付 bundle 导出：`{preprocess,polar,polar_with_ref}.onnx` + `manifest.json`。

两个 run 的分类器图与定义模块导出的前处理图一次成型，对应 MaaEnd 交付布局
`assets/resource/model/map/cameraorientation/`（拷入步骤见 README「拷入 MaaEnd」）：

    uv run export-artifact --out runs/<name>/bundle \
        --polar-run runs/<polar_run> --ref-run runs/<ref_run>

- `preprocess.onnx` 由定义模块 `endfield/preprocess.py` 导出；
- `polar.onnx` / `polar_with_ref.onnx` 由各自 run 的 `best.pt` 导出（`export-onnx`）；
- `manifest.json` 由 `endfield/bundle.py` 按已导出的图与 run 产物组装：交付角色词汇、
  字段 schema 与结构自检都在那里，本模块只提供 run 产物的布局（`load_run`）。

导出后跑结构自检（manifest ↔ 图 metadata ↔ 文件哈希互证、ORT 1.19.2 可加载），
失败退出码 1（图与 manifest 仍落盘，便于定位）。数值 conformance 证据用
`uv run verify-artifact --bundle <out>` 落报告（#30）。

重复导出确定性：同一 run + 同一定义 + 同一工具链 → 图与 manifest 逐字节一致。
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from cli.export_onnx import export as export_classifier
from cli.export_onnx import git_commit
from endfield import bundle, preprocess, run_record
from endfield import conformance as cf

SUMMARY_NAME = "summary.json"


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path}: 读取失败：{exc}") from exc


def load_run(run_dir: Path) -> tuple[run_record.RunRecord, Mapping[str, Any]]:
    """run 目录布局（`record.json` / `summary.json`）-> 运行档案与训练汇总。

    run 产物的形状留在这里：bundle 侧经回调取用，不持有 run 目录布局（ADR 0004）。
    """
    return run_record.read(run_dir), _read_json(run_dir / SUMMARY_NAME)


def export_bundle(out_dir: Path, polar_run: Path, ref_run: Path) -> Path:
    """导出三图并写 manifest.json；目录已存在时原地覆盖（重复导出确定性）。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    preprocess_graph = bundle.graph_file(bundle.DeliveryRole.PREPROCESS)
    preprocess.export_onnx(out_dir / preprocess_graph)
    print(
        f"exported: {out_dir / preprocess_graph} "
        f"(definition_hash={preprocess.definition_hash()[:12]})"
    )
    runs: dict[bundle.DeliveryRole, Path] = {
        bundle.DeliveryRole.POLAR: Path(polar_run),
        bundle.DeliveryRole.POLAR_WITH_REF: Path(ref_run),
    }
    for role, run_dir in runs.items():
        export_classifier(run_dir / "best.pt", out_dir / bundle.graph_file(role))
    manifest = bundle.build_manifest(
        out_dir, runs, load_run, cf.profile(), git_commit=git_commit()
    )
    (out_dir / bundle.MANIFEST_NAME).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return out_dir


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

    bundle_dir = export_bundle(args.out, args.polar_run, args.ref_run)
    _, findings = bundle.check_structure(bundle_dir, cf.profile())
    errors = [finding for finding in findings if finding.level == "error"]
    for finding in errors:
        print(f"ERROR: {finding.message}")
    if errors:
        print(f"result: FAIL（{len(errors)} 项结构自检错误；图与 manifest 保留在 {bundle_dir}）")
        return 1
    manifest = _read_json(bundle_dir / bundle.MANIFEST_NAME)
    print(f"bundle: {bundle_dir}")
    print(f"result: PASS（三图 + manifest；definition_hash={manifest['definition_hash'][:12]}）")
    print(f"conformance 证据：uv run verify-artifact --bundle {bundle_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
