"""export_onnx：polar / ref 两种输入模式的 ONNX 导出契约。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnx
import pytest
import torch
from onnx.reference import ReferenceEvaluator

from endfield.model import ARCH_VERSION, AzimuthNet, load_model
from export_onnx import ExportWrapper, export, fold_input_conventions


def write_run(tmp_path: Path, input_mode: str, channels: int) -> Path:
    run_dir = tmp_path / input_mode
    run_dir.mkdir()
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
        json.dumps({"val_expected_abs_error": 1.0, "val_rms_error": 2.0}), encoding="utf-8"
    )
    return run_dir


@pytest.mark.parametrize(("input_mode", "channels"), [("polar", 3), ("ref", 7)])
def test_export_matches_torch_and_declares_mode(
    tmp_path: Path, input_mode: str, channels: int
) -> None:
    run_dir = write_run(tmp_path, input_mode, channels)
    output = run_dir / "model.onnx"
    export(run_dir / "best.pt", output)

    graph = onnx.load(output)
    dims = [dim.dim_value for dim in graph.graph.input[0].type.tensor_type.shape.dim]
    assert dims == [1, 42, 360, channels]
    metadata = {prop.key: prop.value for prop in graph.metadata_props}
    assert metadata["input_mode"] == input_mode
    if input_mode == "ref":
        assert "ref.BGR" in metadata["input_spec"] and "ref.A" in metadata["input_spec"]
    else:
        assert "polar unwrap" in metadata["input_spec"]

    net = load_model(run_dir / "best.pt")
    fold_input_conventions(net)
    wrapper = ExportWrapper(net).eval()
    rng = np.random.default_rng(0)
    sample = rng.integers(0, 256, (1, 42, 360, channels), dtype=np.uint8)
    with torch.no_grad():
        expected = wrapper(torch.from_numpy(sample)).numpy()
    got = ReferenceEvaluator(graph).run(None, {"strip": sample})[0]
    assert got.shape == (1, 360)
    assert np.allclose(expected, got, atol=1e-5)
