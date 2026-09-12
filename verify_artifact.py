"""工件校验 CLI：对 bundle 跑结构与数值 conformance，给出通过/失败与差异明细。

用法见 README「工件校验（conformance）」。示例：

    uv run verify_artifact.py --bundle runs/<name>/bundle
    uv run verify_artifact.py --bundle <dir> --require polar,polar_with_ref --report r.json
    uv run verify_artifact.py --dump-fixtures conformance/fixtures

退出码 0 = 通过（允许 warning），1 = 存在 error 或超容差比对。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from endfield.conformance import (
    BundleReport,
    dump_builtin_fixtures,
    verify_bundle,
)

ROLE_CHOICES = ("preprocess", "polar", "polar_with_ref")


def format_report(report: BundleReport) -> str:
    lines = [f"bundle: {report.bundle}"]
    if report.fixtures:
        lines.append(f"fixtures: {', '.join(report.fixtures)}")
    for finding in report.findings:
        lines.append(f"{finding.level.upper():7s} [{finding.code}] {finding.message}")
    for comparison in report.comparisons:
        status = "PASS" if comparison.passed else "FAIL"
        if comparison.max_abs is None:
            metrics = comparison.note
        else:
            metrics = (
                f"max_abs={comparison.max_abs:g} mean_abs={comparison.mean_abs:.5g} "
                f"p99={comparison.p99_abs:g} diff={comparison.diff_fraction:.4%} "
                f"limit={comparison.limit:g}"
            )
        lines.append(f"{status:4s} {comparison.label}: {metrics}")
    for name, value in sorted(report.metrics.items()):
        lines.append(f"METRIC {name}={value:.6g}")
    errors = sum(1 for finding in report.findings if finding.level == "error")
    warnings = sum(1 for finding in report.findings if finding.level == "warning")
    lines.append(
        f"result: {'PASS' if report.passed else 'FAIL'} "
        f"({errors} errors, {warnings} warnings, {len(report.comparisons)} comparisons)"
    )
    if not report.passed:
        for failure in report.failures():
            lines.append(f"  - {failure}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle", type=Path, default=Path("."), help="bundle 目录（含图与 manifest.json）"
    )
    parser.add_argument(
        "--fixture-dir", type=Path, default=None, help="fixture 目录（*.npz）；缺省用内置场景"
    )
    parser.add_argument(
        "--run-dir", type=Path, default=None, help="训练 run 目录（分类器数值比对参考）"
    )
    parser.add_argument(
        "--require",
        type=str,
        default=None,
        help=f"逗号分隔的图角色 {ROLE_CHOICES}；缺省由 manifest/草稿规则决定",
    )
    parser.add_argument("--report", type=Path, default=None, help="JSON 报告输出路径")
    parser.add_argument(
        "--dump-fixtures", type=Path, default=None, help="写出内置 fixtures（*.npz）后退出"
    )
    args = parser.parse_args(argv)

    if args.dump_fixtures is not None:
        paths = dump_builtin_fixtures(args.dump_fixtures)
        for path in paths:
            print(f"wrote: {path}")
        return 0

    require = tuple(args.require.split(",")) if args.require else None
    report = verify_bundle(
        args.bundle,
        fixture_dir=args.fixture_dir,
        run_dir=args.run_dir,
        require=require,
    )
    print(format_report(report))
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"report: {args.report}")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
