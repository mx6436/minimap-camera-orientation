"""bundle 模块：交付角色词汇、manifest 组装与结构自检。

用最小手工 bundle 直接驱动接口（交付角色表、`build_manifest`、`check_structure`），
因此不导出 onnx、不训练；run 目录用最小真实产物构造（`endfield/run_dir.py`）。
"""

from __future__ import annotations

import json
from pathlib import Path

import onnx
import pytest

from endfield import bundle, run_dir, run_record
from endfield.run_record import InputMode, RunRecord
from endfield.train.metrics import Metrics
from tests._onnx_builders import build_classifier, build_draft_preprocess

PROFILE = bundle.ManifestProfile(
    definition_hash="a" * 64,
    ort_version="1.19.2",
    tolerances={"strips_uint8": 1.0, "gap_fraction": 0.01},
    fixtures=("polar_basic",),
)
GIT_COMMIT = "b" * 40
METRICS = {"val_rms_error_deg": "2.000000", "val_expected_abs_error_deg": "1.000000"}


def write_run_dir(path: Path, mode: InputMode) -> None:
    """最小 run 目录：运行档案 + 训练汇总（真实契约，不再注入假读取）。"""
    path.mkdir(parents=True, exist_ok=True)
    assets_root = (
        Path("local/maplocator/resource/image/MapLocator") if mode is InputMode.REF else None
    )
    run_record.write(path, RunRecord(mode, assets_root, 3.0, 0))
    run_dir.write_summary(
        path,
        run_dir.TrainingSummary(
            epoch=3,
            val_count=5,
            metrics=Metrics(expected_abs_error=1.0, rms_error=2.0),
        ),
    )


def assert_codes(findings: list, *codes: str) -> None:
    got = {finding.code for finding in findings}
    assert set(codes) <= got, f"missing {set(codes) - got} in {sorted(got)}"


def write_graph(path: Path, model: onnx.ModelProto, **metadata: str) -> None:
    for key, value in metadata.items():
        entry = model.metadata_props.add()
        entry.key = key
        entry.value = value
    onnx.save(model, path)


def _rename(model: onnx.ModelProto, names: dict[str, str]) -> None:
    for value in model.graph.output:
        value.name = names.get(value.name, value.name)
    for node in model.graph.node:
        for index, output in enumerate(node.output):
            node.output[index] = names.get(output, output)


def write_graphs(
    bundle_dir: Path, *, mode: InputMode = InputMode.REF
) -> dict[bundle.DeliveryRole, Path]:
    """交付集合内的图与 run 目录：preprocess + ref 分类器（`mode` 可控以构造模式错配）。"""
    bundle_dir.mkdir(parents=True, exist_ok=True)
    preprocess_graph = build_draft_preprocess(emit_reference=True)
    _rename(preprocess_graph, {"obs": "observed", "ref": "reference"})
    write_graph(
        bundle_dir / bundle.graph_file(bundle.DeliveryRole.PREPROCESS),
        preprocess_graph,
        definition_hash=PROFILE.definition_hash,
    )
    role = bundle.DeliveryRole.POLAR_WITH_REF
    write_graph(
        bundle_dir / bundle.graph_file(role),
        build_classifier(7 if mode is InputMode.REF else 3),
        input_mode=mode.value,
        git_commit=GIT_COMMIT,
        **METRICS,
    )
    runs = {role: bundle_dir / mode.value}
    write_run_dir(runs[role], mode)
    return runs


def build_bundle(bundle_dir: Path) -> tuple[Path, dict]:
    """写一次合法 bundle：build_manifest 落盘 manifest，返回 (目录, manifest)。"""
    manifest = bundle.build_manifest(
        bundle_dir, write_graphs(bundle_dir), PROFILE, git_commit=GIT_COMMIT
    )
    (bundle_dir / bundle.MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
    return bundle_dir, manifest


@pytest.fixture
def valid_bundle(tmp_path: Path) -> Path:
    return build_bundle(tmp_path / "bundle")[0]


def check(bundle_dir: Path, **kwargs) -> list:
    return bundle.check_structure(bundle_dir, PROFILE, **kwargs)[1]


# --------------------------------------------------------------------------- #
# 交付角色词汇
# --------------------------------------------------------------------------- #


def test_roles_cover_the_role_vocabulary() -> None:
    assert bundle.roles() == (
        bundle.DeliveryRole.PREPROCESS,
        bundle.DeliveryRole.POLAR,
        bundle.DeliveryRole.POLAR_WITH_REF,
    )
    assert [bundle.graph_file(role) for role in bundle.roles()] == [
        "preprocess.onnx",
        "polar.onnx",
        "polar_with_ref.onnx",
    ]


def test_delivered_roles_are_preprocess_and_ref() -> None:
    assert bundle.delivered_roles() == (
        bundle.DeliveryRole.PREPROCESS,
        bundle.DeliveryRole.POLAR_WITH_REF,
    )
    assert bundle.delivered_classifier_roles() == (bundle.DeliveryRole.POLAR_WITH_REF,)


def test_classifier_roles_exclude_preprocess() -> None:
    assert bundle.classifier_roles() == (
        bundle.DeliveryRole.POLAR,
        bundle.DeliveryRole.POLAR_WITH_REF,
    )
    with pytest.raises(ValueError, match="no input mode"):
        bundle.input_mode(bundle.DeliveryRole.PREPROCESS)


def test_roles_and_modes_round_trip() -> None:
    assert bundle.input_mode(bundle.DeliveryRole.POLAR) is InputMode.POLAR
    assert bundle.input_mode("polar_with_ref") is InputMode.REF
    assert bundle.role_for_mode("ref") is bundle.DeliveryRole.POLAR_WITH_REF
    assert bundle.role_for_mode(InputMode.POLAR) is bundle.DeliveryRole.POLAR


def test_unknown_role_names_the_valid_roles() -> None:
    with pytest.raises(ValueError, match="unknown delivery role"):
        bundle.graph_file("polar_ref")


# --------------------------------------------------------------------------- #
# manifest 组装
# --------------------------------------------------------------------------- #


def test_build_manifest_records_roles_and_metrics(tmp_path: Path) -> None:
    _, manifest = build_bundle(tmp_path / "bundle")

    assert manifest["schema_version"] == bundle.SCHEMA_VERSION
    assert manifest["git_commit"] == GIT_COMMIT
    assert manifest["definition_hash"] == PROFILE.definition_hash
    assert manifest["ort_version"] == PROFILE.ort_version
    assert manifest["fixtures"] == list(PROFILE.fixtures)
    assert manifest["tolerances"] == dict(PROFILE.tolerances)
    assert sorted(manifest["graphs"]) == sorted(role.value for role in bundle.delivered_roles())
    assert manifest["graphs"]["preprocess"]["outputs"] == {
        "observed": "observed",
        "reference": "reference",
    }
    ref = manifest["graphs"]["polar_with_ref"]
    assert ref["input_mode"] == "ref"
    assert ref["input_channels"] == 7
    assert ref["metrics"] == {
        "best_epoch": 3,
        "val_count": 5,
        "val_rms_error_deg": 2.0,
        "val_expected_abs_error_deg": 1.0,
    }


def test_build_manifest_requires_every_delivered_classifier_role(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    runs = write_graphs(bundle_dir)
    runs.clear()

    with pytest.raises(ValueError, match="every delivered classifier role"):
        bundle.build_manifest(bundle_dir, runs, PROFILE, git_commit=GIT_COMMIT)


def test_build_manifest_rejects_out_of_scope_role(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    runs = write_graphs(bundle_dir)
    runs[bundle.DeliveryRole.POLAR] = bundle_dir / "polar"

    with pytest.raises(ValueError, match="delivery scope excludes"):
        bundle.build_manifest(bundle_dir, runs, PROFILE, git_commit=GIT_COMMIT)


def test_build_manifest_is_deterministic(tmp_path: Path) -> None:
    first = build_bundle(tmp_path / "one")[1]
    second = build_bundle(tmp_path / "two")[1]

    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


# --------------------------------------------------------------------------- #
# 结构自检
# --------------------------------------------------------------------------- #


def test_check_structure_passes_on_a_consistent_bundle(valid_bundle: Path) -> None:
    manifest, findings = bundle.check_structure(valid_bundle, PROFILE)

    assert findings == []
    assert manifest is not None and manifest["graphs"]["polar_with_ref"]["input_channels"] == 7


def test_check_structure_flags_tampered_manifest_hash(valid_bundle: Path) -> None:
    manifest = json.loads((valid_bundle / bundle.MANIFEST_NAME).read_text(encoding="utf-8"))
    manifest["graphs"]["polar_with_ref"]["sha256"] = "0" * 64
    (valid_bundle / bundle.MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")

    assert_codes(check(valid_bundle), "graph_sha256")


def test_check_structure_flags_swapped_classifier_run(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    manifest = bundle.build_manifest(
        bundle_dir,
        write_graphs(bundle_dir, mode=InputMode.POLAR),
        PROFILE,
        git_commit=GIT_COMMIT,
    )
    (bundle_dir / bundle.MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")

    assert_codes(
        check(bundle_dir), "graph_input_mode", "graph_metadata_input_mode", "graph_channels"
    )


def test_check_structure_flags_out_of_scope_graph(valid_bundle: Path) -> None:
    manifest = json.loads((valid_bundle / bundle.MANIFEST_NAME).read_text(encoding="utf-8"))
    manifest["graphs"]["polar"] = dict(manifest["graphs"]["polar_with_ref"])
    (valid_bundle / bundle.MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")

    assert_codes(check(valid_bundle), "graphs_roles")


def test_check_structure_flags_stale_graph_metadata(valid_bundle: Path) -> None:
    write_graph(
        valid_bundle / bundle.graph_file(bundle.DeliveryRole.POLAR_WITH_REF),
        build_classifier(7),
        input_mode="ref",
        git_commit="c" * 40,
        **METRICS,
    )

    assert_codes(check(valid_bundle), "graph_git_commit", "graph_sha256")


def test_check_structure_flags_missing_graph(valid_bundle: Path) -> None:
    (valid_bundle / bundle.graph_file(bundle.DeliveryRole.PREPROCESS)).unlink()

    assert_codes(check(valid_bundle), "graph_missing")


def test_check_structure_flags_invalid_graph(valid_bundle: Path) -> None:
    (valid_bundle / bundle.graph_file(bundle.DeliveryRole.POLAR_WITH_REF)).write_bytes(b"not onnx")

    assert_codes(check(valid_bundle), "graph_invalid")


def test_check_structure_flags_stale_manifest(valid_bundle: Path) -> None:
    manifest = json.loads((valid_bundle / bundle.MANIFEST_NAME).read_text(encoding="utf-8"))
    manifest["schema_version"] = 0
    manifest["ort_version"] = "1.30.0"
    manifest["git_commit"] = ""
    manifest["definition_hash"] = "0" * 64
    manifest["fixtures"] = ["polar_basic", "no_such_fixture"]
    manifest["tolerances"] = {"strips_uint8": 1.0}
    manifest["graphs"].pop("polar_with_ref")
    (valid_bundle / bundle.MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")

    assert_codes(
        check(valid_bundle),
        "schema_version",
        "manifest_ort_version",
        "git_commit_missing",
        "definition_hash",
        "fixture_unknown",
        "tolerances_missing",
        "graphs_roles",
        "graph_spec_missing",
    )


def test_check_structure_reports_missing_sections(valid_bundle: Path) -> None:
    (valid_bundle / bundle.MANIFEST_NAME).write_text(json.dumps({}), encoding="utf-8")

    assert_codes(
        check(valid_bundle), "graphs_missing", "fixtures_missing", "tolerances_missing"
    )


def test_check_structure_flags_unreadable_manifest(valid_bundle: Path) -> None:
    (valid_bundle / bundle.MANIFEST_NAME).write_text("{ not json", encoding="utf-8")

    manifest, findings = bundle.check_structure(valid_bundle, PROFILE)

    assert manifest is None
    assert_codes(findings, "manifest_invalid")


def test_check_structure_draft_path_warns_instead_of_erroring(valid_bundle: Path) -> None:
    (valid_bundle / bundle.MANIFEST_NAME).unlink()

    manifest, draft = bundle.check_structure(valid_bundle, PROFILE, require_manifest=False)
    assert manifest is None
    assert [(finding.level, finding.code) for finding in draft] == [
        ("warning", "manifest_missing")
    ]

    _, required = bundle.check_structure(valid_bundle, PROFILE)
    assert [(finding.level, finding.code) for finding in required] == [
        ("error", "manifest_missing")
    ]
