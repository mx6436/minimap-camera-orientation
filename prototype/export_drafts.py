"""Export the #23 draft `preprocess.onnx` variants and report their graph structure.

Run:  uv run python -m prototype.export_drafts [--out prototype/drafts]

Exports, for each variant in `prototype.preprocess_variants.VARIANTS`:

    prototype/drafts/preprocess_<variant>.onnx

draft graph contract (mirrors the #29 skeleton): inputs `minimap` [1,118,120,3] uint8,
`asset` [1,H,W,4] uint8 with dynamic H/W, scalar `x`/`y`/`scale`; outputs `observed`
[1,42,360,3] uint8, `reference` [1,42,360,4] uint8. Structure (GridSample attrs, dynamic
asset dims, dtype) is asserted here so the comparison harness can trust the drafts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from endfield.ref import ROI_H, ROI_W
from prototype.preprocess_variants import VARIANTS

STRIP_H, STRIP_W = 42, 360
DYNAMIC_ASSET = (1, 140, 160, 4)


def export_variant(name: str, out_dir: Path) -> Path:
    model = VARIANTS[name]().eval()
    minimap = torch.zeros(1, ROI_H, ROI_W, 3, dtype=torch.uint8)
    asset = torch.zeros(1, *DYNAMIC_ASSET[1:], dtype=torch.uint8)
    args = (minimap, asset, torch.tensor(80.0), torch.tensor(70.0), torch.tensor(15.0 / 16.0))
    path = out_dir / f"preprocess_{name}.onnx"
    torch.onnx.export(
        model,
        args,
        path,
        opset_version=18,
        input_names=["minimap", "asset", "x", "y", "scale"],
        output_names=["observed", "reference"],
        dynamic_shapes=(None, {1: "H", 2: "W"}, None, None, None),
        external_data=False,
    )
    return path


def describe(path: Path) -> dict:
    import onnx

    model = onnx.load(str(path))
    onnx.checker.check_model(model)

    inputs = {
        value.name: [dim.dim_param or dim.dim_value for dim in value.type.tensor_type.shape.dim]
        for value in model.graph.input
    }
    outputs = {
        value.name: [dim.dim_param or dim.dim_value for dim in value.type.tensor_type.shape.dim]
        for value in model.graph.output
    }
    grid_samples = []
    for node in model.graph.node:
        if node.op_type != "GridSample":
            continue
        attrs = {}
        for attr in node.attribute:
            attrs[attr.name] = attr.s.decode() if attr.type == attr.STRING else attr.i
        grid_samples.append(attrs)
    opset = next(entry.version for entry in model.opset_import if entry.domain in ("", "ai.onnx"))
    assert opset == 18, f"{path.name}: opset {opset} != 18"
    assert inputs["asset"][1] == "H" and inputs["asset"][2] == "W", (
        f"{path.name}: asset H/W not dynamic"
    )
    assert tuple(inputs["asset"][:1] + inputs["asset"][3:]) == (1, 4), f"{path.name}: asset shape"
    for attrs in grid_samples:
        assert attrs["mode"] == "bilinear", f"{path.name}: {attrs}"
        assert attrs["align_corners"] == 0, f"{path.name}: {attrs}"
        assert attrs["padding_mode"] in ("border", "zeros"), f"{path.name}: {attrs}"
    assert all(
        value.type.tensor_type.elem_type == onnx.TensorProto.UINT8 for value in model.graph.output
    ), f"{path.name}: outputs not uint8"

    import onnxruntime as ort

    ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])  # load check
    return {
        "file": str(path),
        "opset": opset,
        "inputs": inputs,
        "outputs": outputs,
        "grid_sample": grid_samples,
        "nodes": len(model.graph.node),
        "ort_version": ort.__version__,
        "ort_loads": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("prototype/drafts"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    report = {}
    for name in VARIANTS:
        path = export_variant(name, args.out)
        report[name] = describe(path)
        print(
            f"{name}: {path} nodes={report[name]['nodes']} "
            f"grid_sample={report[name]['grid_sample']}"
        )
    (args.out / "structure.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"structure report: {args.out / 'structure.json'}")


if __name__ == "__main__":
    main()
