"""preprocess.onnx 导出契约：结构、ORT ↔ 定义模块数值对拍、动态维与确定性。"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import onnx
import pytest

from endfield import conformance as cf
from endfield import preprocess


def _feeds(scenario: cf.Scenario) -> dict[str, np.ndarray]:
    return {
        "minimap": scenario.minimap,
        "asset": cf.normalize_asset(scenario.asset),
        "x": np.float32(scenario.x),
        "y": np.float32(scenario.y),
        "scale": np.float32(scenario.scale),
    }


def test_export_preprocess_matches_definition_module(tmp_path: Path) -> None:
    path = preprocess.export_onnx(tmp_path / "preprocess.onnx")
    assert path.is_file()

    model = onnx.load(str(path))
    findings = cf.check_preprocess_model(model, {"observed": "observed", "reference": "reference"})
    assert [finding for finding in findings if finding.level == "error"] == []

    for name in cf.scenario_map():
        scenario = cf.scenario_map()[name]
        expected_observed, expected_reference = cf.reference_strips(scenario)
        outputs = cf.run_model(path, _feeds(scenario))
        assert (
            np.abs(
                outputs["observed"][0].astype(np.int16) - expected_observed.astype(np.int16)
            ).max()
            <= 1
        )
        assert (
            np.abs(
                outputs["reference"][0].astype(np.int16) - expected_reference.astype(np.int16)
            ).max()
            <= 1
        )


def _tensor_producers(model: object) -> dict[str, object]:
    producers: dict[str, object] = {}
    for node in model.graph.node:  # type: ignore[attr-defined]
        for output in node.output:
            producers[output] = node
    return producers


def _tensor_origins(model: object) -> object:
    """张量来源的图输入名集合（沿生产者链回溯）；用于识别资产角色的 GridSample。"""
    graph_inputs = {value.name for value in model.graph.input}  # type: ignore[attr-defined]
    producers = _tensor_producers(model)
    cache: dict[str, set[str]] = {}

    def origins(name: str) -> set[str]:
        if name in cache:
            return cache[name]
        if name in graph_inputs:
            return {name}
        node = producers.get(name)
        found: set[str] = set()
        if node is not None:
            for source in node.input:  # type: ignore[attr-defined]
                if source:
                    found |= origins(source)
        cache[name] = found
        return found

    return origins


def test_export_preprocess_crops_asset_before_float_cast(tmp_path: Path) -> None:
    """asset 的直接消费者只有动态 Slice 与 Shape，才说明裁剪发生在 float 转换之前。"""
    path = preprocess.export_onnx(tmp_path / "preprocess.onnx")
    model = onnx.load(str(path))
    origins = _tensor_origins(model)
    producers = _tensor_producers(model)
    graph_inputs = {value.name for value in model.graph.input}

    asset_grid_samples = [
        node
        for node in model.graph.node
        if node.op_type == "GridSample"
        and node.input
        and "asset" in origins(node.input[0])
        and "minimap" not in origins(node.input[0])
    ]
    assert len(asset_grid_samples) == 1

    seen: set[str] = set()
    stack = [asset_grid_samples[0].input[0]]
    cropped = False
    while stack:
        name = stack.pop()
        if name in seen or name in graph_inputs:
            continue
        seen.add(name)
        node = producers.get(name)
        if node is None:
            continue
        if node.op_type == "Slice" and "asset" in node.input:
            cropped = True
            break
        stack.extend(source for source in node.input if source)
    assert cropped, "asset 必须先经数据相关 Slice 裁采样窗再 Cast/采样（窗口优先）"

    direct = {node.op_type for node in model.graph.node if "asset" in node.input}
    assert direct <= {"Slice", "Shape"}, f"asset 的直接消费者只能是 Slice/Shape，实际 {direct}"


def test_export_preprocess_declares_contract_and_metadata(tmp_path: Path) -> None:
    path = preprocess.export_onnx(tmp_path / "preprocess.onnx")
    model = onnx.load(str(path))

    inputs = {value.name: value for value in model.graph.input}
    assert set(inputs) == {"minimap", "asset", "x", "y", "scale"}
    asset_dims = inputs["asset"].type.tensor_type.shape.dim
    assert asset_dims[1].dim_param and asset_dims[2].dim_param  # H/W 动态
    assert [dim.dim_value for dim in inputs["minimap"].type.tensor_type.shape.dim] == [
        1,
        preprocess.ROI_H,
        preprocess.ROI_W,
        3,
    ]
    outputs = {value.name: value for value in model.graph.output}
    assert set(outputs) == {"observed", "reference"}

    metadata = {prop.key: prop.value for prop in model.metadata_props}
    assert metadata["definition_hash"] == cf.definition_hash()
    assert "BGRA" in metadata["input_spec"]
    assert "ref.BGR" in metadata["output_spec"]


def test_export_preprocess_handles_other_asset_sizes(tmp_path: Path) -> None:
    path = preprocess.export_onnx(tmp_path / "preprocess.onnx")
    rng = np.random.default_rng(11)
    observed = rng.integers(0, 256, (preprocess.ROI_H, preprocess.ROI_W, 3), dtype=np.uint8)
    asset = rng.integers(0, 256, (90, 110, 4), dtype=np.uint8)

    expected_observed, expected_reference = preprocess.strips(observed, asset, 40.0, 40.0, 1.0)
    outputs = cf.run_model(
        path,
        {
            "minimap": observed,
            "asset": asset,
            "x": np.float32(40.0),
            "y": np.float32(40.0),
            "scale": np.float32(1.0),
        },
    )

    assert (
        np.abs(outputs["observed"][0].astype(np.int16) - expected_observed.astype(np.int16)).max()
        <= 1
    )
    assert (
        np.abs(outputs["reference"][0].astype(np.int16) - expected_reference.astype(np.int16)).max()
        <= 1
    )


def test_export_preprocess_is_deterministic(tmp_path: Path) -> None:
    first = preprocess.export_onnx(tmp_path / "first.onnx")
    second = preprocess.export_onnx(tmp_path / "second.onnx")
    assert (
        hashlib.sha256(first.read_bytes()).hexdigest()
        == hashlib.sha256(second.read_bytes()).hexdigest()
    )


def test_export_preprocess_cli(tmp_path: Path) -> None:
    from export_preprocess import main

    output = tmp_path / "cli" / "preprocess.onnx"
    assert main(["--out", str(output)]) == 0
    assert output.is_file()
    assert cf.check_environment(None) == []


@pytest.mark.parametrize("bad", [(4, 4, 2), (4, 4, 5)])
def test_normalize_asset_rejects_non_bgr_shapes(bad: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="HxWx3 or HxWx4"):
        preprocess.normalize_asset(np.zeros(bad, dtype=np.uint8))
