"""测试用 ONNX 图构建器：与原型票（#23）同形的草稿 preprocess（GridSample 展开）。"""

from __future__ import annotations

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from endfield.polar import IMG_H, IMG_W, INNER_R, OUTER_R
from endfield.ref import ROI_H, ROI_POLE, ROI_W


def _grid() -> np.ndarray:
    radii = INNER_R + (OUTER_R - INNER_R) / IMG_H * (np.arange(IMG_H, dtype=np.float32) + 0.5)
    theta = np.deg2rad(np.arange(IMG_W, dtype=np.float32))
    map_x = ROI_POLE[0] + radii[:, None] * np.sin(theta)[None, :]
    map_y = ROI_POLE[1] - radii[:, None] * np.cos(theta)[None, :]
    grid = np.empty((1, IMG_H, IMG_W, 2), dtype=np.float32)
    grid[0, ..., 0] = (2.0 * map_x + 1.0) / ROI_W - 1.0
    grid[0, ..., 1] = (2.0 * map_y + 1.0) / ROI_H - 1.0
    return grid


def build_draft_preprocess(*, emit_reference: bool = False) -> onnx.ModelProto:
    """草稿 preprocess：NHWC uint8 -> GridSample 展开 -> obs（+可选的缺失参考）。

    与真实原型一样按 #22 约束搭：opset18、`mode="bilinear"`、`padding_mode="border"`、
    `align_corners=0`、图内 Cast float32、围绕 GridSample 的 Round+Cast 回 uint8。
    """
    inputs = [
        helper.make_tensor_value_info("minimap", TensorProto.UINT8, [1, ROI_H, ROI_W, 3]),
        helper.make_tensor_value_info("asset", TensorProto.UINT8, [None, None, 4]),
        helper.make_tensor_value_info("x", TensorProto.FLOAT, []),
        helper.make_tensor_value_info("y", TensorProto.FLOAT, []),
        helper.make_tensor_value_info("scale", TensorProto.FLOAT, []),
    ]
    outputs = [helper.make_tensor_value_info("obs", TensorProto.UINT8, [1, IMG_H, IMG_W, 3])]
    nodes = [
        helper.make_node("Transpose", ["minimap"], ["nhwc"], perm=[0, 3, 1, 2]),
        helper.make_node("Cast", ["nhwc"], ["f32"], to=TensorProto.FLOAT),
        helper.make_node(
            "GridSample",
            ["f32", "grid"],
            ["sampled"],
            mode="bilinear",
            padding_mode="border",
            align_corners=0,
        ),
        helper.make_node("Round", ["sampled"], ["rounded"]),
        helper.make_node("Cast", ["rounded"], ["u8"], to=TensorProto.UINT8),
        helper.make_node("Transpose", ["u8"], ["obs"], perm=[0, 2, 3, 1]),
    ]
    initializers = [numpy_helper.from_array(_grid(), "grid")]
    if emit_reference:
        outputs.append(
            helper.make_tensor_value_info("ref", TensorProto.UINT8, [1, IMG_H, IMG_W, 4])
        )
        initializers.append(
            numpy_helper.from_array(np.zeros((1, IMG_H, IMG_W, 1), dtype=np.uint8), "zero_alpha")
        )
        nodes.append(helper.make_node("Concat", ["obs", "zero_alpha"], ["ref"], axis=3))
    graph = helper.make_graph(nodes, "draft_preprocess", inputs, outputs, initializer=initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    return model


def find_node(model: onnx.ModelProto, op_type: str) -> onnx.NodeProto:
    return next(node for node in model.graph.node if node.op_type == op_type)


def set_attr(node: onnx.NodeProto, name: str, value: object) -> None:
    for attr in node.attribute:
        if attr.name == name:
            if isinstance(value, str):
                attr.s = value.encode("utf-8")
            else:
                attr.i = int(value)  # type: ignore[arg-type]
            return
    raise KeyError(f"{node.op_type}: no attribute {name!r}")


def build_classifier(channels: int = 7, *, softmax: bool = True) -> onnx.ModelProto:
    """分类器同形图：uint8 NHWC 条带 -> 通道均值 -> pmf [1,360]（权重全 1，仅结构用）。"""
    inp = helper.make_tensor_value_info("strip", TensorProto.UINT8, [1, IMG_H, IMG_W, channels])
    out = helper.make_tensor_value_info("pmf", TensorProto.FLOAT, [1, 360])
    weight = np.full((channels, 360), 1.0 / 360.0, dtype=np.float32)
    reshape_shape = np.asarray([1, channels], dtype=np.int64)
    axes = np.asarray([2, 3], dtype=np.int64)
    nodes = [
        helper.make_node("Transpose", ["strip"], ["nchw"], perm=[0, 3, 1, 2]),
        helper.make_node("Cast", ["nchw"], ["f32"], to=TensorProto.FLOAT),
        helper.make_node("ReduceMean", ["f32", "axes"], ["pooled"], keepdims=1),
        helper.make_node("Reshape", ["pooled", "reshape_shape"], ["flat"]),
        helper.make_node("MatMul", ["flat", "weight"], ["logits"]),
    ]
    if softmax:
        nodes.append(helper.make_node("Softmax", ["logits"], ["pmf"], axis=1))
    else:
        nodes.append(helper.make_node("Identity", ["logits"], ["pmf"]))
    graph = helper.make_graph(
        nodes,
        "classifier",
        [inp],
        [out],
        initializer=[
            numpy_helper.from_array(weight, "weight"),
            numpy_helper.from_array(reshape_shape, "reshape_shape"),
            numpy_helper.from_array(axes, "axes"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    return model
