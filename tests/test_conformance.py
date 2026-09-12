"""conformance 框架：fixtures、参考实现适配、结构断言与容差比对。"""

from __future__ import annotations

import numpy as np
import pytest

from endfield import conformance as cf
from endfield.polar import IMG_H, IMG_W, INNER_R, OUTER_R, unwrap
from endfield.ref import ROI_POLE
from tests._onnx_builders import build_classifier, build_draft_preprocess, find_node, set_attr

REQUIRED_CASES = {
    "polar_basic": "polar",
    "ref_pair_basic": "ref",
    "crop_out_of_bounds": "oob",
    "zone_non_1to1": "scale",
    "ref_missing_alpha0": "gap",
    "asset_rgb_3ch": "3ch",
}


def test_builtin_scenarios_cover_required_cases() -> None:
    scenarios = cf.builtin_scenarios()
    assert {scenario.name for scenario in scenarios} == set(REQUIRED_CASES)
    for scenario in scenarios:
        assert REQUIRED_CASES[scenario.name] in scenario.tags
        assert scenario.minimap.shape == (cf.ROI_H, cf.ROI_W, 3)
        assert scenario.minimap.dtype == np.uint8
        assert scenario.asset.ndim == 3
        assert scenario.asset.dtype == np.uint8
        assert scenario.description


def test_builtin_scenarios_are_deterministic() -> None:
    first = {scenario.name: scenario for scenario in cf.builtin_scenarios()}
    second = {scenario.name: scenario for scenario in cf.builtin_scenarios()}
    for name, scenario in first.items():
        assert np.array_equal(scenario.minimap, second[name].minimap)
        assert np.array_equal(scenario.asset, second[name].asset)
        assert (scenario.x, scenario.y, scenario.scale) == (
            second[name].x,
            second[name].y,
            second[name].scale,
        )


def test_scenario_shapes_match_cases() -> None:
    scenarios = cf.scenario_map()
    assert scenarios["crop_out_of_bounds"].asset.shape[0] < cf.ROI_H
    assert scenarios["crop_out_of_bounds"].asset.shape[1] < cf.ROI_W
    assert np.all(scenarios["ref_missing_alpha0"].asset[..., 3] == 0)
    assert scenarios["asset_rgb_3ch"].asset.shape[2] == 3
    assert scenarios["zone_non_1to1"].scale == 15.0 / 16.0
    assert scenarios["ref_pair_basic"].asset.shape[2] == 4
    alpha = scenarios["ref_pair_basic"].asset[..., 3]
    assert alpha.min() < 255 and alpha.max() == 255


def test_dump_and_load_fixtures(tmp_path) -> None:
    written = cf.dump_builtin_fixtures(tmp_path)
    assert [path.name for path in written] == [f"{name}.npz" for name in REQUIRED_CASES]
    loaded = {scenario.name: scenario for scenario in cf.load_fixtures(tmp_path)}
    assert set(loaded) == set(REQUIRED_CASES)
    for scenario in cf.builtin_scenarios():
        assert np.array_equal(scenario.minimap, loaded[scenario.name].minimap)
        assert np.array_equal(scenario.asset, loaded[scenario.name].asset)
        assert scenario.x == loaded[scenario.name].x
        assert scenario.tags == loaded[scenario.name].tags


def test_load_fixtures_rejects_empty_dir(tmp_path) -> None:
    with pytest.raises(ValueError, match="no \\*.npz fixtures"):
        cf.load_fixtures(tmp_path)


def test_reference_observed_strip_matches_unwrap() -> None:
    scenario = cf.scenario_map()["polar_basic"]
    observed, reference = cf.reference_strips(scenario)
    expected = unwrap(scenario.minimap, ROI_POLE[0], ROI_POLE[1], INNER_R, OUTER_R)
    assert observed.shape == (IMG_H, IMG_W, 3)
    assert reference.shape == (IMG_H, IMG_W, 4)
    assert np.array_equal(observed, expected)
    assert observed.dtype == np.uint8 and reference.dtype == np.uint8


def test_reference_missing_alpha_copies_observed() -> None:
    scenario = cf.scenario_map()["ref_missing_alpha0"]
    observed, reference = cf.reference_strips(scenario)
    assert np.all(reference[..., 3] == 0)
    assert np.array_equal(reference[..., :3], observed)
    assert cf.gap_fraction(reference) == 1.0


def test_reference_three_channel_asset_is_opaque() -> None:
    scenario = cf.scenario_map()["asset_rgb_3ch"]
    _, reference = cf.reference_strips(scenario)
    assert np.all(reference[..., 3] == 255)
    assert cf.gap_fraction(reference) == 0.0


def test_normalize_asset_pads_alpha() -> None:
    rgb = np.full((4, 5, 3), 7, dtype=np.uint8)
    padded = cf.normalize_asset(rgb)
    assert padded.shape == (4, 5, 4)
    assert np.all(padded[..., 3] == 255)
    assert np.array_equal(padded[..., :3], rgb)
    assert cf.normalize_asset(padded) is padded


def test_definition_hash_is_stable_hex() -> None:
    digest = cf.definition_hash()
    assert len(digest) == 64
    assert all(char in "0123456789abcdef" for char in digest)
    assert digest == cf.definition_hash()


def test_environment_matches_pinned_ort() -> None:
    assert cf.check_environment(None) == []


def test_check_environment_rejects_manifest_ort_mismatch() -> None:
    findings = cf.check_environment({"ort_version": "1.30.0"})
    assert any(finding.code == "manifest_ort_version" for finding in findings)


def test_compare_identical_arrays_passes() -> None:
    array = np.arange(24, dtype=np.uint8).reshape(2, 3, 4)
    result = cf.compare_arrays("x", array, array.copy(), cf.Tolerance(1.0))
    assert result.passed
    assert result.max_abs == 0.0
    assert result.diff_fraction == 0.0


def test_compare_within_tolerance_passes_and_reports_metrics() -> None:
    expected = np.zeros((4, 4), dtype=np.uint8)
    actual = expected.copy()
    actual[1, 2] = 1
    result = cf.compare_arrays("x", actual, expected, cf.Tolerance(1.0))
    assert result.passed
    assert result.max_abs == 1.0
    assert result.diff_fraction == pytest.approx(1 / 16)


def test_compare_over_tolerance_fails() -> None:
    expected = np.zeros((4, 4), dtype=np.uint8)
    actual = expected.copy()
    actual[0, 0] = 2
    result = cf.compare_arrays("x", actual, expected, cf.Tolerance(1.0))
    assert not result.passed
    assert result.max_abs == 2.0


def test_compare_squeezes_batch_dimension() -> None:
    expected = np.zeros((42, 360, 3), dtype=np.uint8)
    actual = expected[None].copy()
    result = cf.compare_arrays("x", actual, expected, cf.Tolerance(0.0))
    assert result.passed
    assert result.shape_ok


def test_compare_shape_and_dtype_mismatch_fail() -> None:
    expected = np.zeros((2, 2), dtype=np.uint8)
    shape = cf.compare_arrays("x", np.zeros((2, 3), dtype=np.uint8), expected, cf.Tolerance(1.0))
    assert not shape.passed and not shape.shape_ok
    dtype = cf.compare_arrays("x", expected.astype(np.float32), expected, cf.Tolerance(1.0))
    assert not dtype.passed and dtype.shape_ok and not dtype.dtype_ok


def test_gap_fraction_counts_missing_alpha() -> None:
    strip = np.zeros((2, 2, 4), dtype=np.uint8)
    strip[..., 3] = 255
    strip[0, 0, 3] = 0
    assert cf.gap_fraction(strip) == pytest.approx(0.25)
    with pytest.raises(ValueError):
        cf.gap_fraction(strip[..., :3])


def test_resolve_tolerances_merges_manifest_overrides() -> None:
    tolerances = cf.resolve_tolerances(
        {"tolerances": {"strips_uint8": 2, "pmf_float32": {"max_abs": 1e-3}}}
    )
    assert tolerances["strips_uint8"] == 2.0
    assert tolerances["pmf_float32"] == 1e-3
    assert tolerances["gap_fraction"] == cf.DEFAULT_TOLERANCES["gap_fraction"]


def test_check_preprocess_accepts_valid_draft() -> None:
    model = build_draft_preprocess(emit_reference=True)
    findings = cf.check_preprocess_model(model, {"observed": "obs", "reference": "ref"})
    assert [finding for finding in findings if finding.level == "error"] == []
    assert not any(finding.code == "gridsample_dtype" for finding in findings)


def test_check_preprocess_rejects_wrong_grid_sample_attrs() -> None:
    model = build_draft_preprocess()
    set_attr(find_node(model, "GridSample"), "mode", "linear")
    assert any(
        finding.code == "gridsample_mode" for finding in cf.check_preprocess_model(model, {})
    )

    model = build_draft_preprocess()
    set_attr(find_node(model, "GridSample"), "padding_mode", "zeros")
    assert any(
        finding.code == "gridsample_padding" for finding in cf.check_preprocess_model(model, {})
    )

    model = build_draft_preprocess()
    set_attr(find_node(model, "GridSample"), "align_corners", 1)
    assert any(
        finding.code == "gridsample_align_corners"
        for finding in cf.check_preprocess_model(model, {})
    )


def test_check_preprocess_requires_grid_sample() -> None:
    model = build_draft_preprocess()
    graph = model.graph
    del graph.node[:]
    findings = cf.check_preprocess_model(model, {})
    assert any(finding.code == "gridsample_missing" for finding in findings)


def test_check_preprocess_rejects_uint8_grid_sample_input() -> None:
    model = build_draft_preprocess()
    cast = find_node(model, "Cast")
    graph = model.graph
    for node in graph.node:
        if node.op_type == "GridSample":
            node.input[0] = "nhwc"
    graph.node.remove(cast)
    findings = cf.check_preprocess_model(model, {})
    assert any(finding.code == "gridsample_dtype" for finding in findings)


def test_check_preprocess_rejects_static_asset_spatial() -> None:
    model = build_draft_preprocess()
    asset = next(value for value in model.graph.input if value.name == "asset")
    for index, size in enumerate((720, 1280)):
        asset.type.tensor_type.shape.dim[index].dim_value = size
    findings = cf.check_preprocess_model(model, {})
    assert any(finding.code == "asset_static_spatial" for finding in findings)


def test_check_preprocess_rejects_contrib_domain() -> None:
    import onnx
    from onnx import helper

    model = build_draft_preprocess()
    model.graph.node.append(
        helper.make_node(
            "FusedConv",
            ["f32", "grid"],
            ["sampled"],
            domain="com.microsoft",
        )
    )
    findings = cf.check_preprocess_model(model, {})
    assert any(finding.code == "contrib_domain" for finding in findings)
    assert isinstance(model, onnx.ModelProto)


def test_check_preprocess_rejects_wrong_output_channels() -> None:
    model = build_draft_preprocess(emit_reference=True)
    findings = cf.check_preprocess_model(model, {"observed": "ref", "reference": "ref"})
    assert any(finding.code == "output_channels" for finding in findings)


def test_check_classifier_accepts_valid_graph() -> None:
    findings = cf.check_classifier_model(build_classifier(7), 7)
    assert [finding for finding in findings if finding.level == "error"] == []


def test_check_classifier_rejects_wrong_channels_and_missing_softmax() -> None:
    findings = cf.check_classifier_model(build_classifier(3), 7)
    assert any(finding.code == "input_channels" for finding in findings)
    no_softmax = cf.check_classifier_model(build_classifier(7, softmax=False), 7)
    assert any(finding.code == "softmax_missing" for finding in no_softmax)
