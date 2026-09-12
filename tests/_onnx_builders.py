"""测试用 ONNX 图构建器：与定义模块（#25）同形的草稿 preprocess（两个 GridSample）。"""

from __future__ import annotations

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from endfield import preprocess


def _grid() -> np.ndarray:
    radii = preprocess.INNER_R + (preprocess.OUTER_R - preprocess.INNER_R) / preprocess.IMG_H * (
        np.arange(preprocess.IMG_H, dtype=np.float32) + 0.5
    )
    theta = np.deg2rad(np.arange(preprocess.IMG_W, dtype=np.float32))
    map_x = preprocess.ROI_POLE[0] + radii[:, None] * np.sin(theta)[None, :]
    map_y = preprocess.ROI_POLE[1] - radii[:, None] * np.cos(theta)[None, :]
    grid = np.empty((1, preprocess.IMG_H, preprocess.IMG_W, 2), dtype=np.float32)
    grid[0, ..., 0] = (2.0 * map_x + 1.0) / preprocess.ROI_W - 1.0
    grid[0, ..., 1] = (2.0 * map_y + 1.0) / preprocess.ROI_H - 1.0
    return grid


def build_draft_preprocess(*, emit_reference: bool = False) -> onnx.ModelProto:
    """草稿 preprocess：NHWC uint8 -> GridSample 展开 -> obs（+可选的参考采样）。

    与定义模块同形按 #22 约束搭：opset18、`mode="bilinear"`、`align_corners=0`、
    图内 Cast float32、围绕 GridSample 的 Round+Cast 回 uint8；观测（minimap）
    `padding_mode="border"`，资产 `padding_mode="zeros"`（#23 契约修正）。
    """
    inputs = [
        helper.make_tensor_value_info(
            "minimap", TensorProto.UINT8, [1, preprocess.ROI_H, preprocess.ROI_W, 3]
        ),
        helper.make_tensor_value_info("asset", TensorProto.UINT8, [1, None, None, 4]),
        helper.make_tensor_value_info("x", TensorProto.FLOAT, []),
        helper.make_tensor_value_info("y", TensorProto.FLOAT, []),
        helper.make_tensor_value_info("scale", TensorProto.FLOAT, []),
    ]
    outputs = [
        helper.make_tensor_value_info(
            "obs", TensorProto.UINT8, [1, preprocess.IMG_H, preprocess.IMG_W, 3]
        )
    ]
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
            helper.make_tensor_value_info(
                "ref", TensorProto.UINT8, [1, preprocess.IMG_H, preprocess.IMG_W, 4]
            )
        )
        nodes.extend(
            [
                helper.make_node("Transpose", ["asset"], ["asset_nhwc"], perm=[0, 3, 1, 2]),
                helper.make_node("Cast", ["asset_nhwc"], ["asset_f32"], to=TensorProto.FLOAT),
                helper.make_node(
                    "GridSample",
                    ["asset_f32", "grid"],
                    ["asset_sampled"],
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=0,
                ),
                helper.make_node("Round", ["asset_sampled"], ["asset_rounded"]),
                helper.make_node("Cast", ["asset_rounded"], ["asset_u8"], to=TensorProto.UINT8),
                helper.make_node("Transpose", ["asset_u8"], ["ref"], perm=[0, 2, 3, 1]),
            ]
        )
    graph = helper.make_graph(nodes, "draft_preprocess", inputs, outputs, initializer=initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    return model


def find_nodes(model: onnx.ModelProto, op_type: str) -> list[onnx.NodeProto]:
    return [node for node in model.graph.node if node.op_type == op_type]


def find_node(model: onnx.ModelProto, op_type: str) -> onnx.NodeProto:
    return find_nodes(model, op_type)[0]


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
    inp = helper.make_tensor_value_info(
        "strip", TensorProto.UINT8, [1, preprocess.IMG_H, preprocess.IMG_W, channels]
    )
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
