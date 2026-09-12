"""verify_artifact：bundle 编排与 CLI 端到端（草稿图 = GridSample 展开）。"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import onnx
import pytest
import torch

from endfield import conformance as cf
from endfield.model import ARCH_VERSION, AzimuthNet
from tests._onnx_builders import build_classifier, build_draft_preprocess
from verify_artifact import main

REPO_ROOT = Path(__file__).resolve().parents[1]


def write_bundle(
    tmp_path: Path, *, emit_reference: bool = False, manifest: dict | None = None
) -> Path:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    onnx.save(build_draft_preprocess(emit_reference=emit_reference), bundle / "preprocess.onnx")
    if manifest is not None:
        (bundle / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return bundle


def write_fixture_dir(tmp_path: Path, names: list[str]) -> Path:
    fixture_dir = tmp_path / "fixtures"
    fixture_dir.mkdir()
    scenarios = cf.scenario_map()
    for name in names:
        scenarios[name].to_npz(fixture_dir / f"{name}.npz")
    return fixture_dir


def write_run(tmp_path: Path, input_mode: str, channels: int) -> Path:
    run_dir = tmp_path / f"run_{input_mode}"
    run_dir.mkdir()
    torch.save(
        {"model": AzimuthNet(in_channels=channels).state_dict(), "arch": ARCH_VERSION},
        run_dir / "best.pt",
    )
    (run_dir / "record.json").write_text(
        json.dumps(
            {
                "version": 31,
                "input_mode": input_mode,
                "target_sigma": 3.0,
                "trainable_parameters": 0,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "summary.json").write_text(
        json.dumps({"val_expected_abs_error": 1.0, "val_rms_error": 2.0}), encoding="utf-8"
    )
    return run_dir


def test_run_model_adapts_batch_and_scalars(tmp_path: Path) -> None:
    path = tmp_path / "preprocess.onnx"
    onnx.save(build_draft_preprocess(), path)
    scenario = cf.scenario_map()["polar_basic"]
    outputs = cf.run_model(
        path,
        {
            "minimap": scenario.minimap,
            "asset": cf.normalize_asset(scenario.asset),
            "x": scenario.x,
            "y": scenario.y,
            "scale": scenario.scale,
        },
    )
    assert outputs["obs"].shape == (1, cf.STRIP_H, cf.STRIP_W, 3)
    with pytest.raises(ValueError, match="missing feeds"):
        cf.run_model(path, {"minimap": scenario.minimap})


def test_verify_bundle_passes_on_draft_graph(tmp_path: Path) -> None:
    bundle = write_bundle(tmp_path, emit_reference=True)
    fixture_dir = write_fixture_dir(tmp_path, ["ref_missing_alpha0"])
    report = cf.verify_bundle(bundle, fixture_dir=fixture_dir)
    assert report.passed, report.failures()
    labels = {comparison.label for comparison in report.comparisons}
    assert labels == {"ref_missing_alpha0.observed", "ref_missing_alpha0.reference"}
    assert report.metrics["ref_missing_alpha0.gap_fraction_error"] == 0.0
    assert any(finding.code == "manifest_missing" for finding in report.findings)


def test_verify_bundle_reports_diff_details_on_unsupported_fixtures(tmp_path: Path) -> None:
    bundle = write_bundle(tmp_path, emit_reference=True)
    report = cf.verify_bundle(bundle)
    assert not report.passed
    by_label = {comparison.label: comparison for comparison in report.comparisons}
    assert by_label["polar_basic.observed"].passed
    assert not by_label["polar_basic.reference"].passed
    assert by_label["polar_basic.reference"].max_abs > 1
    json.dumps(report.to_dict(), ensure_ascii=False)


def test_verify_bundle_honours_manifest_fixtures_and_tolerances(tmp_path: Path) -> None:
    manifest = {
        "ort_version": cf.ORT_VERSION,
        "definition_hash": cf.definition_hash(),
        "graphs": {
            "preprocess": {
                "file": "preprocess.onnx",
                "outputs": {"observed": "obs", "reference": "ref"},
            }
        },
        "fixtures": ["ref_pair_basic"],
        "tolerances": {"strips_uint8": 0.0},
    }
    bundle = write_bundle(tmp_path, emit_reference=True, manifest=manifest)
    report = cf.verify_bundle(bundle)
    assert report.fixtures == ["ref_pair_basic"]
    assert not report.passed
    failures = report.failures()
    assert any("ref_pair_basic.reference" in failure for failure in failures)
    assert not any(finding.code == "definition_hash" for finding in report.findings)


def test_verify_bundle_flags_stale_definition_hash(tmp_path: Path) -> None:
    manifest = {
        "ort_version": cf.ORT_VERSION,
        "definition_hash": "0" * 64,
        "graphs": {"preprocess": {"file": "preprocess.onnx"}},
    }
    bundle = write_bundle(tmp_path, emit_reference=True, manifest=manifest)
    report = cf.verify_bundle(bundle)
    assert any(finding.code == "definition_hash" for finding in report.findings)
    assert not report.passed


def test_verify_bundle_flags_missing_declared_graph(tmp_path: Path) -> None:
    manifest = {
        "ort_version": cf.ORT_VERSION,
        "definition_hash": cf.definition_hash(),
        "graphs": {
            "preprocess": {"file": "preprocess.onnx"},
            "polar_with_ref": {"file": "polar_with_ref.onnx"},
        },
    }
    bundle = write_bundle(tmp_path, emit_reference=True, manifest=manifest)
    report = cf.verify_bundle(bundle)
    assert any(finding.code == "graph_missing" for finding in report.findings)


def test_verify_bundle_reports_invalid_graph(tmp_path: Path) -> None:
    bundle = write_bundle(tmp_path)
    (bundle / "preprocess.onnx").write_bytes(b"not an onnx model")
    report = cf.verify_bundle(bundle)
    assert any(finding.code == "graph_invalid" for finding in report.findings)
    assert not report.passed


def test_verify_bundle_reports_invalid_manifest_inputs(tmp_path: Path) -> None:
    manifest = {
        "graphs": {"preprocess": {"file": "preprocess.onnx"}},
        "tolerances": {"strips_uint8": None},
    }
    bundle = write_bundle(tmp_path, emit_reference=True, manifest=manifest)
    report = cf.verify_bundle(bundle)
    assert any(finding.code == "tolerances_invalid" for finding in report.findings)

    empty = tmp_path / "empty_fixtures"
    empty.mkdir()
    report = cf.verify_bundle(bundle, fixture_dir=empty)
    assert any(finding.code == "fixtures_invalid" for finding in report.findings)
    assert not report.passed


def test_cli_exit_codes_and_report_file(tmp_path: Path) -> None:
    bundle = write_bundle(tmp_path, emit_reference=True)
    fixture_dir = write_fixture_dir(tmp_path, ["ref_missing_alpha0"])
    report_path = tmp_path / "report.json"
    passed = subprocess.run(
        [
            sys.executable,
            "verify_artifact.py",
            "--bundle",
            str(bundle),
            "--fixture-dir",
            str(fixture_dir),
            "--report",
            str(report_path),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert passed.returncode == 0, passed.stdout + passed.stderr
    assert "result: PASS" in passed.stdout
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["passed"] is True

    failing = subprocess.run(
        [sys.executable, "verify_artifact.py", "--bundle", str(bundle)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert failing.returncode == 1
    assert "result: FAIL" in failing.stdout


def test_cli_dump_fixtures(tmp_path: Path) -> None:
    out_dir = tmp_path / "fixtures"
    assert main(["--dump-fixtures", str(out_dir)]) == 0
    assert sorted(path.name for path in out_dir.glob("*.npz")) == sorted(
        f"{name}.npz" for name in cf.scenario_map()
    )


@pytest.mark.parametrize("role", ["polar", "polar_with_ref"])
def test_verify_bundle_checks_classifier_structure(tmp_path: Path, role: str) -> None:
    bundle = write_bundle(tmp_path)
    onnx.save(build_classifier(7 if role == "polar_with_ref" else 3), bundle / f"{role}.onnx")
    report = cf.verify_bundle(bundle, require=[role])
    assert not any(finding.level == "error" for finding in report.findings), report.failures()
    assert any(finding.code == "classifier_reference_missing" for finding in report.findings)


def test_verify_bundle_compares_classifier_with_checkpoint(tmp_path: Path) -> None:
    from export_onnx import export

    run_dir = write_run(tmp_path, "polar", 3)
    bundle = write_bundle(tmp_path)
    export(run_dir / "best.pt", bundle / "polar.onnx")
    report = cf.verify_bundle(bundle, require=["polar"], run_dir=run_dir)
    labels = [comparison.label for comparison in report.comparisons]
    assert any(label.endswith(".polar.pmf") for label in labels)
    assert all(comparison.passed for comparison in report.comparisons), report.failures()
    assert any(key.endswith("angle_error_deg") for key in report.metrics)
    assert not any(finding.level == "error" for finding in report.findings)


def test_verify_bundle_skips_mode_mismatched_checkpoint(tmp_path: Path) -> None:
    from export_onnx import export

    run_dir = write_run(tmp_path, "polar", 3)
    bundle = write_bundle(tmp_path)
    export(run_dir / "best.pt", bundle / "polar.onnx")
    onnx.save(build_classifier(7), bundle / "polar_with_ref.onnx")
    report = cf.verify_bundle(bundle, require=["polar_with_ref"], run_dir=run_dir)
    assert any(finding.code == "classifier_mode_mismatch" for finding in report.findings)
    assert report.comparisons == []
