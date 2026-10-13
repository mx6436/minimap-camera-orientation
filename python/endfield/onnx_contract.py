"""交付分类器图的 ONNX IO 契约与 ORT 可加载性检查。

预处理语义的权威在 MaaEnd C++（`CameraOrientationPreprocess.cpp`）；本模块只约束
`cli/export_onnx.py` 落盘图的契约：输入 `strip` 为 uint8 NHWC 参考配对条带，输出
`pmf` 为 float32 [1,360] 概率质量函数（softmax 已在图内）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from endfield.strip import INPUT_CHANNELS, STRIP_H, STRIP_W

# 与 MaaEnd 运行时同版本；版本不符时导出校验的结论作废
ORT_VERSION = "1.19.2"
INPUT_NAME = "strip"
OUTPUT_NAME = "pmf"
OUTPUT_BINS = 360

_ONNX_DOMAINS = ("", "ai.onnx")


def check_classifier_model(model: Any) -> list[str]:
    """返回交付图违反 IO 契约的原因；空列表即通过。"""
    import onnx

    problems: list[str] = []
    for node in model.graph.node:
        if node.domain not in _ONNX_DOMAINS:
            problems.append(f"non-standard domain node: {node.domain!r}")
    if not model.graph.input:
        return [*problems, "graph has no input"]
    if len(model.graph.input) != 1:
        problems.append(f"{len(model.graph.input)} inputs, expected 1")
    value = model.graph.input[0]
    if value.name != INPUT_NAME:
        problems.append(f"input name {value.name!r}, expected {INPUT_NAME!r}")
    if value.type.tensor_type.elem_type != onnx.TensorProto.UINT8:
        problems.append(f"input {value.name!r} is not uint8")
    dims = value.type.tensor_type.shape.dim
    if len(dims) != 4:
        problems.append(f"input rank {len(dims)}, expected 4 (NHWC)")
    else:
        shape = [dim.dim_value for dim in dims]
        if shape[1:] != [STRIP_H, STRIP_W, INPUT_CHANNELS]:
            problems.append(
                f"input shape {shape}, expected [1,{STRIP_H},{STRIP_W},{INPUT_CHANNELS}]"
            )
    if not model.graph.output:
        return [*problems, "graph has no output"]
    if len(model.graph.output) != 1:
        problems.append(f"{len(model.graph.output)} outputs, expected 1")
    value = model.graph.output[0]
    if value.name != OUTPUT_NAME:
        problems.append(f"output name {value.name!r}, expected {OUTPUT_NAME!r}")
    if value.type.tensor_type.elem_type != onnx.TensorProto.FLOAT:
        problems.append(f"output {value.name!r} is not float32")
    dims = value.type.tensor_type.shape.dim
    if len(dims) != 2 or dims[-1].dim_value != OUTPUT_BINS:
        problems.append(f"output shape {[dim.dim_value for dim in dims]}, expected [1,360]")
    if not any(node.op_type == "Softmax" for node in model.graph.node):
        problems.append("graph has no Softmax node")
    return problems


def check_onnx_runtime(path: Path) -> None:
    """ORT 加载校验：版本与加载失败即抛，合格图不返回结论。"""
    import onnxruntime as ort

    if ort.__version__ != ORT_VERSION:
        raise RuntimeError(
            f"onnxruntime {ort.__version__} != delivery contract {ORT_VERSION}: "
            "load check is meaningless"
        )
    ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
