"""export_artifact：三图 bundle 与 manifest 契约、conformance 可消费性与确定性。"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import onnx
import pytest
import torch

from cli.export_artifact import GRAPH_FILES, SCHEMA_VERSION, check_bundle, export_bundle, main
from endfield import conformance as cf
from endfield import preprocess
from endfield.model import ARCH_VERSION, AzimuthNet


def write_run(root: Path, name: str, input_mode: str, channels: int) -> Path:
    run_dir = root / name
    run_dir.mkdir(parents=True)
    torch.save(
        {"model": AzimuthNet(in_channels=channels).state_dict(), "arch": ARCH_VERSION},
        run_dir / "best.pt",
    )
    record = {
        "version": 31,
        "input_mode": input_mode,
        "target_sigma": 3.0,
        "trainable_parameters": 0,
    }
    if input_mode == "ref":
        record["ref_reference_assets_root"] = "local/maplocator/resource/image/MapLocator"
    (run_dir / "record.json").write_text(json.dumps(record), encoding="utf-8")
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "epoch": 7,
                "val_count": 11,
                "val_expected_abs_error": 1.0,
                "val_rms_error": 2.0,
            }
        ),
        encoding="utf-8",
    )
    return run_dir


def graph_metadata(path: Path) -> dict[str, str]:
    model = onnx.load(str(path))
    return {prop.key: prop.value for prop in model.metadata_props}


def read_manifest(bundle: Path) -> dict:
    return json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))


def write_manifest(bundle: Path, manifest: dict) -> None:
    (bundle / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


@pytest.fixture(scope="module")
def runs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("artifact-runs")
    return {
        "polar": write_run(root, "polar", "polar", 3),
        "ref": write_run(root, "ref", "ref", 7),
    }


@pytest.fixture(scope="module")
def bundle(tmp_path_factory: pytest.TempPathFactory, runs: dict[str, Path]) -> Path:
    out = tmp_path_factory.mktemp("artifact") / "bundle"
    export_bundle(out, runs["polar"], runs["ref"])
    return out


def test_bundle_has_three_graphs_and_complete_manifest(bundle: Path, runs: dict[str, Path]) -> None:
    manifest = read_manifest(bundle)

    assert manifest["schema_version"] == SCHEMA_VERSION
    assert manifest["definition_hash"] == preprocess.definition_hash()
    assert manifest["ort_version"] == cf.ORT_VERSION
    assert manifest["git_commit"]
    assert set(manifest["graphs"]) == set(GRAPH_FILES)
    assert manifest["fixtures"] == [scenario.name for scenario in cf.builtin_scenarios()]
    assert manifest["tolerances"] == cf.DEFAULT_TOLERANCES

    for role, file_name in GRAPH_FILES.items():
        spec = manifest["graphs"][role]
        assert spec["file"] == file_name
        path = bundle / file_name
        assert path.is_file()
        assert spec["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()

    assert manifest["graphs"]["preprocess"]["outputs"] == {
        "observed": "observed",
        "reference": "reference",
    }
    assert graph_metadata(bundle / "preprocess.onnx")["definition_hash"] == (
        preprocess.definition_hash()
    )

    expected_modes = {"polar": ("polar", 3), "polar_with_ref": ("ref", 7)}
    for role, (mode, channels) in expected_modes.items():
        spec = manifest["graphs"][role]
        assert spec["input_mode"] == mode
        assert spec["input_channels"] == channels
        assert spec["metrics"] == {
            "best_epoch": 7,
            "val_count": 11,
            "val_rms_error_deg": 2.0,
            "val_expected_abs_error_deg": 1.0,
        }
        run_dir = (bundle / spec["run_dir"]).resolve()
        assert run_dir == runs["polar" if role == "polar" else "ref"].resolve()
        assert (run_dir / "best.pt").is_file()
        metadata = graph_metadata(bundle / f"{role}.onnx")
        assert metadata["input_mode"] == mode
        assert metadata["val_rms_error_deg"] == "2.000000"

    assert check_bundle(bundle) == []


def test_manifest_is_consumable_by_conformance(bundle: Path) -> None:
    report = cf.verify_bundle(bundle)

    assert report.passed, report.failures()
    assert report.fixtures == [scenario.name for scenario in cf.builtin_scenarios()]
    assert not any(finding.code == "classifier_reference_missing" for finding in report.findings)
    labels = {comparison.label for comparison in report.comparisons}
    assert {
        "polar_basic.observed",
        "polar_basic.reference",
        "ref_pair_basic.polar.pmf",
        "ref_pair_basic.polar_with_ref.pmf",
    } <= labels


def test_check_bundle_detects_tampered_manifest_and_missing_graph(
    tmp_path: Path, bundle: Path
) -> None:
    copy = tmp_path / "bundle"
    shutil.copytree(bundle, copy)

    manifest = read_manifest(copy)
    manifest["graphs"]["polar"]["sha256"] = "0" * 64
    write_manifest(copy, manifest)
    errors = check_bundle(copy)
    assert any("polar.onnx" in error and "sha256" in error for error in errors)

    (copy / "preprocess.onnx").unlink()
    errors = check_bundle(copy)
    assert any("preprocess.onnx" in error for error in errors)


def test_check_bundle_detects_swapped_classifier_run(tmp_path: Path, runs: dict[str, Path]) -> None:
    bundle = tmp_path / "bundle"
    # polar 角色塞 ref run：manifest 声明应与图 metadata 对不上
    export_bundle(bundle, runs["ref"], runs["ref"])

    errors = check_bundle(bundle)
    assert any("polar.onnx" in error for error in errors)


def test_cli_requires_both_runs(tmp_path: Path) -> None:
    out = os.fspath(tmp_path / "bundle")
    with pytest.raises(SystemExit) as excinfo:
        main(["--out", out, "--polar-run", os.fspath(tmp_path / "polar")])
    assert excinfo.value.code == 2
    with pytest.raises(SystemExit):
        main(["--out", out])


def test_repeated_export_is_deterministic(
    tmp_path: Path, bundle: Path, runs: dict[str, Path]
) -> None:
    copy = tmp_path / "bundle"
    shutil.copytree(bundle, copy)
    before = {path.name: path.read_bytes() for path in copy.iterdir()}

    rc = main(
        [
            "--out",
            os.fspath(copy),
            "--polar-run",
            os.fspath(runs["polar"]),
            "--ref-run",
            os.fspath(runs["ref"]),
        ]
    )

    assert rc == 0
    after = {path.name: path.read_bytes() for path in copy.iterdir()}
    assert before == after
