"""工件一致性（conformance）：fixtures、参考实现适配、图结构断言与容差比对。

本模块是训练侧交付物与 MaaEnd 运行时的验收口径实现，见 README「工件校验
（conformance）」一节。三件事：

- **fixtures**：6 个确定性合成场景，覆盖 polar / ref 配对 / 裁剪越界 /
  非 1:1 zone / 参考缺失 / 资产 3 通道。场景只提供输入（minimap、asset、
  x、y、scale），期望输出在比对时由参考实现实时计算。
- **参考实现**：即定义模块唯一实现。定义模块落地（#25）之前，这里适配到
  现行 cv2 路径（`endfield.polar` + `endfield.ref`），作为图比对的期望侧；
  定义模块落地后应把 `reference_strips()`/`definition_hash()` 指向它。
- **比对**：ORT 1.19.2 跑图，逐输出比对参考结果并按容差阈值判定，输出
  通过/失败与差异明细（max/mean/p99/差异像素占比、缺口占比误差）。

数值口径（#22）：图输出与参考期望不承诺逐位一致，uint8 条带按 ±1 LSB 预期；
差分语义与阈值见 README。
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from endfield import polar, ref

# 与 MaaEnd 运行时一致的 ORT 版本（pyproject dev 依赖固定）；版本不同证据作废。
ORT_VERSION = "1.19.2"

STRIP_H, STRIP_W = polar.IMG_H, polar.IMG_W
ROI_H, ROI_W = ref.ROI_H, ref.ROI_W

# 图输入契约（#25）：名称固定，运行侧按名喂 fixture。
INPUT_NAMES = ("minimap", "asset", "x", "y", "scale")
# 图输出角色：polar 模式只消费 observed；ref 模式另有 reference。
OUTPUT_ROLES = ("observed", "reference")

# 默认容差剖面：uint8 条带按 ±1 LSB；pmf 按 float32 导出等价；缺口占比按 1 个百分点。
DEFAULT_TOLERANCES: dict[str, float] = {
    "strips_uint8": 1.0,
    "pmf_float32": 1e-4,
    "gap_fraction": 0.01,
}


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Scenario:
    """一个确定性 fixture：输入侧场景，期望输出由参考实现实时计算。"""

    name: str
    description: str
    tags: tuple[str, ...]
    minimap: np.ndarray  # ROI_H x ROI_W x 3 uint8（观测 ROI，非整帧）
    asset: np.ndarray  # H x W x 3|4 uint8（原始底图，3 通道=不透明）
    x: float
    y: float
    scale: float

    def to_npz(self, path: Path) -> None:
        np.savez_compressed(
            path,
            name=np.asarray(self.name),
            description=np.asarray(self.description),
            tags=np.asarray(self.tags),
            minimap=self.minimap,
            asset=self.asset,
            x=np.float32(self.x),
            y=np.float32(self.y),
            scale=np.float32(self.scale),
        )

    @classmethod
    def from_npz(cls, path: Path) -> Scenario:
        with np.load(path, allow_pickle=False) as data:
            return cls(
                name=str(data["name"]),
                description=str(data["description"]),
                tags=tuple(str(t) for t in data["tags"]),
                minimap=data["minimap"].copy(),
                asset=data["asset"].copy(),
                x=float(data["x"]),
                y=float(data["y"]),
                scale=float(data["scale"]),
            )


def _texture(height: int, width: int, seed: int, channels: int) -> np.ndarray:
    """平滑梯度 + 噪声的确定性纹理：双线性采样的差异会显式暴露出来。"""
    yy = np.linspace(0.0, 255.0, height, dtype=np.float32)[:, None]
    xx = np.linspace(0.0, 255.0, width, dtype=np.float32)[None, :]
    base = (yy + xx) / 2.0
    planes = []
    for channel in range(channels):
        rng = np.random.default_rng(seed + channel)
        noise = rng.integers(-16, 17, (height, width)).astype(np.float32)
        planes.append(np.clip(base + noise, 0, 255).astype(np.uint8))
    return np.dstack(planes)


def _rgba(rgb: np.ndarray, alpha: np.ndarray | int) -> np.ndarray:
    if isinstance(alpha, int):
        alpha_plane = np.full(rgb.shape[:2], alpha, dtype=np.uint8)
    else:
        alpha_plane = alpha
    return np.dstack([rgb, alpha_plane])


def builtin_scenarios() -> list[Scenario]:
    """6 个内置合成场景，覆盖票面要求的全部 fixture 类别。"""
    minimap = _texture(ROI_H, ROI_W, seed=11, channels=3)

    pair_asset = _rgba(
        _texture(140, 160, seed=21, channels=3),
        _texture(140, 160, seed=24, channels=1)[..., 0],
    )
    opaque_asset = _rgba(_texture(140, 160, seed=31, channels=3), 255)
    small_asset = _rgba(_texture(48, 40, seed=41, channels=3), 255)
    hidden_asset = _rgba(_texture(140, 160, seed=51, channels=3), 0)
    rgb_asset = _texture(140, 160, seed=61, channels=3)

    return [
        Scenario(
            name="polar_basic",
            description="极坐标展开：观测 ROI 的 42x360x3 条带几何与采样",
            tags=("polar",),
            minimap=minimap,
            asset=opaque_asset,
            x=70.0,
            y=70.0,
            scale=1.0,
        ),
        Scenario(
            name="ref_pair_basic",
            description="参考配对：带连续 alpha 的资产在 (x,y) 处的裁剪、合成与展开",
            tags=("ref",),
            minimap=minimap,
            asset=pair_asset,
            x=80.0,
            y=70.0,
            scale=1.0,
        ),
        Scenario(
            name="crop_out_of_bounds",
            description="裁剪越界：小资产在靠近边角处裁剪，越界外侧按缺失处理",
            tags=("ref", "oob"),
            minimap=minimap,
            asset=small_asset,
            x=8.0,
            y=6.0,
            scale=1.0,
        ),
        Scenario(
            name="zone_non_1to1",
            description="非 1:1 zone：scale=15/16 的裁剪窗口缩放回 ROI 几何",
            tags=("ref", "scale"),
            minimap=minimap,
            asset=pair_asset,
            x=80.0,
            y=70.0,
            scale=15.0 / 16.0,
        ),
        Scenario(
            name="ref_missing_alpha0",
            description="参考缺失：资产全透明，ref.BGR 逐像素等于观测、ref.A 全 0",
            tags=("ref", "gap"),
            minimap=minimap,
            asset=hidden_asset,
            x=80.0,
            y=70.0,
            scale=1.0,
        ),
        Scenario(
            name="asset_rgb_3ch",
            description="资产 3 通道：无 alpha 的 BGR 资产视为完全不透明",
            tags=("ref", "3ch"),
            minimap=minimap,
            asset=rgb_asset,
            x=80.0,
            y=70.0,
            scale=1.0,
        ),
    ]


def scenario_map() -> dict[str, Scenario]:
    return {scenario.name: scenario for scenario in builtin_scenarios()}


def load_fixtures(fixture_dir: Path) -> list[Scenario]:
    """从目录加载 fixture（`*.npz`，`Scenario.to_npz` 的产物）。"""
    paths = sorted(fixture_dir.glob("*.npz"))
    if not paths:
        raise ValueError(f"{fixture_dir}: no *.npz fixtures")
    return [Scenario.from_npz(path) for path in paths]


def dump_builtin_fixtures(out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for scenario in builtin_scenarios():
        path = out_dir / f"{scenario.name}.npz"
        scenario.to_npz(path)
        written.append(path)
    return written


# --------------------------------------------------------------------------- #
# 参考实现（定义模块适配）
# --------------------------------------------------------------------------- #

# 定义模块落地前，参考实现 = 现行 cv2 路径的这两个模块；它们的字节内容即「定义」。
_DEFINITION_SOURCES = (polar.__file__, ref.__file__)


def normalize_asset(asset: np.ndarray) -> np.ndarray:
    """3 通道资产补 255 alpha 成全不透明 BGRA；4 通道原样。"""
    if asset.ndim != 3 or asset.shape[2] not in (3, 4):
        raise ValueError(f"asset must be HxWx3 or HxWx4 uint8, got {asset.shape}")
    if asset.dtype != np.uint8:
        raise ValueError(f"asset must be uint8, got {asset.dtype}")
    if asset.shape[2] == 4:
        return asset
    return _rgba(asset, 255)


def reference_strips(scenario: Scenario) -> tuple[np.ndarray, np.ndarray]:
    """参考期望：`(observed 42x360x3, reference 42x360x4)`，/255 语义与现行实现同源。"""
    seven = ref.ref_strip(scenario.minimap, scenario.asset, scenario.x, scenario.y, scenario.scale)
    return seven[..., :3].copy(), seven[..., 3:].copy()


def definition_hash() -> str:
    """定义模块内容哈希：当前为 cv2 参考实现源码；#25 落地后指向定义模块。"""
    digest = hashlib.sha256()
    for source in sorted(str(Path(path).resolve()) for path in _DEFINITION_SOURCES):
        digest.update(Path(source).read_bytes())
    return digest.hexdigest()


# --------------------------------------------------------------------------- #
# 容差与比对
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Tolerance:
    """单输出的容差：`max_abs` 为逐像素绝对差上限（通过条件），其余为报告指标。"""

    max_abs: float


def resolve_tolerances(manifest: Mapping[str, Any] | None) -> dict[str, float]:
    tolerances = dict(DEFAULT_TOLERANCES)
    if manifest:
        for name, value in (manifest.get("tolerances") or {}).items():
            if isinstance(value, Mapping):
                value = value.get("max_abs")
            if value is None:
                raise ValueError(f"manifest tolerance {name!r}: expected number or max_abs map")
            tolerances[name] = float(value)
    return tolerances


@dataclass
class CompareResult:
    label: str
    shape_ok: bool
    dtype_ok: bool
    max_abs: float | None
    mean_abs: float | None
    p99_abs: float | None
    diff_fraction: float | None
    limit: float
    note: str = ""

    @property
    def passed(self) -> bool:
        return (
            self.shape_ok
            and self.dtype_ok
            and self.max_abs is not None
            and self.max_abs <= self.limit
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "passed": self.passed,
            "shape_ok": self.shape_ok,
            "dtype_ok": self.dtype_ok,
            "max_abs": self.max_abs,
            "mean_abs": self.mean_abs,
            "p99_abs": self.p99_abs,
            "diff_fraction": self.diff_fraction,
            "limit": self.limit,
            "note": self.note,
        }


def compare_arrays(
    label: str, actual: np.ndarray, expected: np.ndarray, tolerance: Tolerance
) -> CompareResult:
    """比对单输出：批维 1 自动挤压；形状/类型不符判失败并记录原因。"""
    note = ""
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    if actual.ndim == expected.ndim + 1 and actual.shape[0] == 1:
        actual = actual[0]
    shape_ok = actual.shape == expected.shape
    dtype_ok = actual.dtype == expected.dtype
    if not shape_ok:
        note = f"shape mismatch: actual {actual.shape} != expected {expected.shape}"
        return CompareResult(
            label, False, dtype_ok, None, None, None, None, tolerance.max_abs, note
        )
    if not dtype_ok:
        note = f"dtype mismatch: actual {actual.dtype} != expected {expected.dtype}"
        return CompareResult(label, True, False, None, None, None, None, tolerance.max_abs, note)
    if actual.size == 0:
        return CompareResult(label, True, True, 0.0, 0.0, 0.0, 0.0, tolerance.max_abs)
    diff = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    return CompareResult(
        label=label,
        shape_ok=True,
        dtype_ok=True,
        max_abs=float(diff.max()),
        mean_abs=float(diff.mean()),
        p99_abs=float(np.percentile(diff, 99)),
        diff_fraction=float(np.mean(diff != 0)),
        limit=tolerance.max_abs,
    )


def gap_fraction(reference_strip: np.ndarray) -> float:
    """参考条带的缺失像素占比（`ref.A < 255`），与训练侧过滤口径一致。"""
    if reference_strip.ndim != 3 or reference_strip.shape[2] != 4:
        raise ValueError(f"reference strip must be HxWx4, got {reference_strip.shape}")
    return float(np.mean(reference_strip[..., 3] < 255))


# --------------------------------------------------------------------------- #
# 图结构断言
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Finding:
    level: str  # error / warning / info
    code: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"level": self.level, "code": self.code, "message": self.message}


def _is_dynamic(dim: Any) -> bool:
    """维度是否可变化：无 dim_value（含 dim_param 或未标注）或 0 视为动态。"""
    return not dim.HasField("dim_value") or dim.dim_value == 0


def _attr_value(node: Any, name: str, default: Any) -> Any:
    for attr in node.attribute:
        if attr.name == name:
            if attr.type == attr.STRING:
                return attr.s.decode("utf-8")
            return attr.i
    return default


def _opset_version(model: Any, domain: str = "") -> int | None:
    for entry in model.opset_import:
        if entry.domain == domain:
            return entry.version
    return None


def _node_domains(model: Any) -> list[str]:
    return sorted({node.domain for node in model.graph.node if node.domain not in ("", "ai.onnx")})


def _grid_sample_findings(model: Any) -> list[Finding]:
    findings: list[Finding] = []
    for domain in _node_domains(model):
        findings.append(
            Finding("error", "contrib_domain", f"非标准域节点：{domain}（不得依赖 contrib）")
        )
    opset = _opset_version(model)
    if opset != 18:
        findings.append(Finding("warning", "opset_mismatch", f"图 opset={opset}，契约固定为 18"))
    grid_samples = [node for node in model.graph.node if node.op_type == "GridSample"]
    if not grid_samples:
        findings.append(Finding("error", "gridsample_missing", "图内没有 GridSample 节点"))
        return findings
    # opset 16-19 的 mode 叫 bilinear；20+ 才叫 linear（#22）。
    expected_mode = "linear" if (opset or 0) >= 20 else "bilinear"
    for index, node in enumerate(grid_samples):
        mode = _attr_value(node, "mode", "bilinear")
        padding = _attr_value(node, "padding_mode", "zeros")
        align = _attr_value(node, "align_corners", 0)
        if mode != expected_mode:
            findings.append(
                Finding(
                    "error",
                    "gridsample_mode",
                    f"GridSample#{index} mode={mode!r}，opset {opset} 必须为 {expected_mode!r}",
                )
            )
        if padding != "border":
            findings.append(
                Finding(
                    "error",
                    "gridsample_padding",
                    f"GridSample#{index} padding_mode={padding!r}，必须为 'border'",
                )
            )
        if align != 0:
            findings.append(
                Finding(
                    "error",
                    "gridsample_align_corners",
                    f"GridSample#{index} align_corners={align}，必须为 0",
                )
            )
    return findings


def _grid_sample_dtype_findings(model: Any) -> list[Finding]:
    """GridSample 的 X/grid 必须是 float32（opset 16-19 只注册 float）。"""
    import onnx

    findings: list[Finding] = []
    known = {value.name: value.type.tensor_type.elem_type for value in model.graph.input}
    known.update({init.name: init.data_type for init in model.graph.initializer})
    grid_samples = [node for node in model.graph.node if node.op_type == "GridSample"]
    try:
        inferred = onnx.shape_inference.infer_shapes(model, strict_mode=False, data_prop=True)
        for value in (*inferred.graph.value_info, *inferred.graph.input, *inferred.graph.output):
            if value.name not in known:
                known[value.name] = value.type.tensor_type.elem_type
    except Exception as exc:  # noqa: BLE001 - 推理失败不阻断，降级为未核实
        return [Finding("warning", "gridsample_dtype", f"形状/类型推理失败：{exc}")]
    for index, node in enumerate(grid_samples):
        for position, name in enumerate(node.input):
            elem = known.get(name)
            if elem == onnx.TensorProto.UINT8:
                findings.append(
                    Finding(
                        "error",
                        "gridsample_dtype",
                        f"GridSample#{index} 输入{position} {name!r} 是 uint8，"
                        "必须图内 Cast 到 float32",
                    )
                )
            elif elem is None:
                findings.append(
                    Finding(
                        "warning",
                        "gridsample_dtype",
                        f"GridSample#{index} 输入{position} {name!r} 类型未能核实",
                    )
                )
            elif elem != onnx.TensorProto.FLOAT:
                findings.append(
                    Finding(
                        "error",
                        "gridsample_dtype",
                        f"GridSample#{index} 输入{position} {name!r} 类型不是 float32",
                    )
                )
    return findings


def check_preprocess_model(model: Any, outputs: Mapping[str, str]) -> list[Finding]:
    """preprocess.onnx 的结构契约：GridSample 属性/dtype + 动态 asset + uint8 条带输出。"""
    import onnx

    findings = _grid_sample_findings(model)
    findings.extend(_grid_sample_dtype_findings(model))

    inputs = {value.name: value for value in model.graph.input}
    asset = inputs.get("asset")
    if asset is None:
        findings.append(Finding("warning", "asset_input_missing", "缺少 asset 输入（草稿图容忍）"))
    else:
        dims = asset.type.tensor_type.shape.dim
        spatial = list(dims[1:3]) if len(dims) == 4 else list(dims[:2])
        if len(dims) not in (3, 4):
            findings.append(
                Finding(
                    "error", "asset_rank", f"asset 输入秩为 {len(dims)}，应为 3（HWC）或 4（NHWC）"
                )
            )
        elif all(not _is_dynamic(dim) for dim in spatial):
            findings.append(
                Finding(
                    "error",
                    "asset_static_spatial",
                    "asset 的 H/W 是静态维：不同 zone 底图尺寸将直接失败，必须导出动态维",
                )
            )
        if asset.type.tensor_type.elem_type != onnx.TensorProto.UINT8:
            findings.append(Finding("error", "asset_dtype", "asset 输入必须是 uint8"))

    if "minimap" not in inputs:
        findings.append(Finding("error", "minimap_input_missing", "缺少 minimap 输入"))

    graph_outputs = {value.name: value for value in model.graph.output}
    for role, name in outputs.items():
        value = graph_outputs.get(name)
        if value is None:
            findings.append(
                Finding("error", "output_missing", f"输出角色 {role} 指向不存在的 {name!r}")
            )
            continue
        if value.type.tensor_type.elem_type != onnx.TensorProto.UINT8:
            findings.append(Finding("error", "output_dtype", f"输出 {name!r}（{role}）不是 uint8"))
            continue
        dims = value.type.tensor_type.shape.dim
        channels = dims[-1].dim_value if dims else 0
        expected_channels = 3 if role == "observed" else 4
        if channels not in (0, expected_channels):
            findings.append(
                Finding(
                    "error",
                    "output_channels",
                    f"输出 {name!r}（{role}）通道数 {channels}，期望 {expected_channels}",
                )
            )
        spatial = dims[-3:-1] if len(dims) >= 3 else []
        if len(spatial) == 2:
            got = [dim.dim_value for dim in spatial]
            if got != [STRIP_H, STRIP_W]:
                findings.append(
                    Finding(
                        "error",
                        "output_shape",
                        f"输出 {name!r} 条带尺寸 {got}，期望 {[STRIP_H, STRIP_W]}",
                    )
                )
    return findings


def check_classifier_model(model: Any, channels: int) -> list[Finding]:
    """polar / polar_with_ref 的结构契约：uint8 NHWC 条带输入 + float32 [1,360] pmf 输出。"""
    import onnx

    findings: list[Finding] = []
    for domain in _node_domains(model):
        findings.append(
            Finding("error", "contrib_domain", f"非标准域节点：{domain}（不得依赖 contrib）")
        )
    if not model.graph.input:
        return findings + [Finding("error", "input_missing", "分类器图没有输入")]
    value = model.graph.input[0]
    if value.type.tensor_type.elem_type != onnx.TensorProto.UINT8:
        findings.append(Finding("error", "input_dtype", f"输入 {value.name!r} 不是 uint8"))
    dims = value.type.tensor_type.shape.dim
    if len(dims) not in (3, 4):
        findings.append(
            Finding("error", "input_rank", f"输入秩 {len(dims)}，应为 3（HWC）或 4（NHWC）")
        )
    else:
        channels_got = dims[-1].dim_value
        height = dims[-3].dim_value
        width = dims[-2].dim_value
        if channels_got not in (0, channels):
            findings.append(
                Finding("error", "input_channels", f"输入通道数 {channels_got}，期望 {channels}")
            )
        if [height, width] != [STRIP_H, STRIP_W]:
            findings.append(
                Finding(
                    "error",
                    "input_shape",
                    f"输入条带尺寸 {[height, width]}，期望 {[STRIP_H, STRIP_W]}",
                )
            )
    if not model.graph.output:
        findings.append(Finding("error", "output_missing", "分类器图没有输出"))
        return findings
    output = model.graph.output[0]
    if output.type.tensor_type.elem_type != onnx.TensorProto.FLOAT:
        findings.append(Finding("error", "output_dtype", f"输出 {output.name!r} 不是 float32"))
    dims = output.type.tensor_type.shape.dim
    if len(dims) not in (1, 2) or dims[-1].dim_value != 360:
        findings.append(Finding("error", "output_shape", f"输出 {output.name!r} 最后一维应为 360"))
    if not any(node.op_type == "Softmax" for node in model.graph.node):
        findings.append(Finding("warning", "softmax_missing", "图内未见 Softmax 节点"))
    return findings


# --------------------------------------------------------------------------- #
# 运行与编排
# --------------------------------------------------------------------------- #


def _adapt_feed(name: str, array: np.ndarray, declared: Any) -> np.ndarray:
    """把 fixture 输入适配到图声明的形状（HWC -> NHWC 加批维；标量按声明形状）。"""
    shape = list(declared.shape)
    array = np.asarray(array)
    if array.ndim <= 1 and len(shape) <= 1:
        return array.astype(np.float32).reshape(shape)
    if len(shape) == array.ndim + 1 and (shape[0] in (None, 1)):
        return array[None]
    if len(shape) == array.ndim:
        return array
    raise ValueError(f"input {name}: fixture shape {array.shape} 不适配图声明 {shape}")


def run_model(path: Path, feeds: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """在 ORT 1.19.2 上跑图；按声明名喂入并按名取回全部输出。"""
    import onnxruntime as ort

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    declared = {value.name: value for value in session.get_inputs()}
    missing = sorted(set(declared) - set(feeds))
    if missing:
        raise ValueError(f"{path}: missing feeds for inputs {missing}")
    unknown = sorted(set(feeds) - set(declared))
    if unknown:
        raise ValueError(f"{path}: feeds for undeclared inputs {unknown}")
    adapted = {name: _adapt_feed(name, feeds[name], declared[name]) for name in declared}
    names = [value.name for value in session.get_outputs()]
    outputs = session.run(names, adapted)
    return dict(zip(names, outputs, strict=True))


def _graph_spec(
    manifest: Mapping[str, Any] | None, role: str
) -> tuple[str, Mapping[str, str] | None]:
    default_files = {
        "preprocess": "preprocess.onnx",
        "polar": "polar.onnx",
        "polar_with_ref": "polar_with_ref.onnx",
    }
    if not manifest or "graphs" not in manifest:
        return default_files[role], None
    spec = (manifest.get("graphs") or {}).get(role)
    if spec is None:
        return default_files[role], None
    if isinstance(spec, str):
        return spec, None
    outputs = spec.get("outputs")
    return spec.get("file", default_files[role]), outputs


def _resolve_output_roles(model: Any, declared: Mapping[str, str] | None) -> dict[str, str]:
    names = [value.name for value in model.graph.output]
    if declared:
        return {role: name for role, name in declared.items() if name in names}
    roles = {}
    for role, name in zip(OUTPUT_ROLES, names, strict=False):
        roles[role] = name
    return roles


def check_environment(manifest: Mapping[str, Any] | None) -> list[Finding]:
    import onnxruntime as ort

    findings: list[Finding] = []
    if ort.__version__ != ORT_VERSION:
        findings.append(
            Finding(
                "error",
                "ort_version",
                f"运行时 onnxruntime {ort.__version__} != 契约 {ORT_VERSION}：证据作废",
            )
        )
    if manifest and manifest.get("ort_version") not in (None, ORT_VERSION):
        findings.append(
            Finding(
                "error",
                "manifest_ort_version",
                f"manifest ort_version={manifest['ort_version']} != {ORT_VERSION}",
            )
        )
    return findings


@dataclass
class BundleReport:
    bundle: str
    findings: list[Finding] = field(default_factory=list)
    comparisons: list[CompareResult] = field(default_factory=list)
    fixtures: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return not any(finding.level == "error" for finding in self.findings) and all(
            comparison.passed for comparison in self.comparisons
        )

    def failures(self) -> list[str]:
        out = [
            f"{finding.code}: {finding.message}"
            for finding in self.findings
            if finding.level == "error"
        ]
        out.extend(
            f"{comparison.label}: {comparison.to_dict()}"
            for comparison in self.comparisons
            if not comparison.passed
        )
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "bundle": self.bundle,
            "passed": self.passed,
            "fixtures": self.fixtures,
            "findings": [finding.to_dict() for finding in self.findings],
            "comparisons": [comparison.to_dict() for comparison in self.comparisons],
            "metrics": self.metrics,
        }


def _load_manifest(bundle_dir: Path, report: BundleReport) -> dict[str, Any] | None:
    path = bundle_dir / "manifest.json"
    if not path.exists():
        report.findings.append(
            Finding("warning", "manifest_missing", "无 manifest.json：按草稿 bundle 校验")
        )
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        report.findings.append(Finding("error", "manifest_invalid", f"manifest 读取失败：{exc}"))
        return None


def _verify_preprocess(
    bundle_dir: Path,
    manifest: Mapping[str, Any] | None,
    scenarios: Sequence[Scenario],
    tolerances: Mapping[str, float],
    report: BundleReport,
) -> None:
    import onnx

    file_name, declared_outputs = _graph_spec(manifest, "preprocess")
    path = bundle_dir / file_name
    if not path.exists():
        report.findings.append(Finding("error", "graph_missing", f"缺少图：{path.name}"))
        return
    try:
        model = onnx.load(str(path))
        onnx.checker.check_model(model)
    except Exception as exc:  # noqa: BLE001 - 图本身非法时给结论而不是崩溃
        report.findings.append(Finding("error", "graph_invalid", f"{path.name}: {exc}"))
        return
    roles = _resolve_output_roles(model, declared_outputs)
    report.findings.extend(check_preprocess_model(model, roles))
    if "observed" not in roles:
        report.findings.append(Finding("error", "output_missing", "preprocess 缺少 observed 输出"))
    strict = manifest is not None
    declared_inputs = {value.name for value in model.graph.input}
    unknown = sorted(declared_inputs - set(INPUT_NAMES))
    if unknown:
        report.findings.append(
            Finding("error", "unknown_input", f"preprocess 图输入名不在契约内：{unknown}")
        )
    required = set(INPUT_NAMES) if strict else {"minimap"}
    for name in sorted(required - declared_inputs):
        report.findings.append(Finding("error", "input_missing", f"preprocess 缺少输入 {name!r}"))

    strip_limit = tolerances["strips_uint8"]
    gap_limit = tolerances["gap_fraction"]
    for scenario in scenarios:
        report.fixtures.append(scenario.name)
        feeds: dict[str, np.ndarray] = {}
        for name in declared_inputs:
            if name in INPUT_NAMES:
                value = getattr(scenario, name)
                feeds[name] = value
        expected_observed, expected_reference = reference_strips(scenario)
        try:
            outputs = run_model(path, feeds)
        except Exception as exc:  # noqa: BLE001 - 运行失败记为比对失败
            report.comparisons.append(
                CompareResult(
                    f"{scenario.name}.run", False, False, None, None, None, None, 0.0, str(exc)
                )
            )
            continue
        if "observed" in roles and roles["observed"] in outputs:
            report.comparisons.append(
                compare_arrays(
                    f"{scenario.name}.observed",
                    outputs[roles["observed"]],
                    expected_observed,
                    Tolerance(strip_limit),
                )
            )
        if "reference" in roles and roles["reference"] in outputs:
            actual_reference = outputs[roles["reference"]]
            report.comparisons.append(
                compare_arrays(
                    f"{scenario.name}.reference",
                    actual_reference,
                    expected_reference,
                    Tolerance(strip_limit),
                )
            )
            squeezed = actual_reference[0] if actual_reference.ndim == 4 else actual_reference
            if squeezed.shape == expected_reference.shape:
                error = abs(gap_fraction(squeezed) - gap_fraction(expected_reference))
                report.metrics[f"{scenario.name}.gap_fraction_error"] = error
                if error > gap_limit:
                    report.findings.append(
                        Finding(
                            "error",
                            "gap_fraction",
                            f"{scenario.name}: 缺口占比误差 {error:.4f} > {gap_limit}",
                        )
                    )


def _resolve_run_dir(
    bundle_dir: Path, manifest: Mapping[str, Any] | None, role: str, run_dir: Path | None
) -> Path | None:
    """按 manifest 的 per-graph run_dir 优先解析；相对路径按 bundle 目录解析。"""
    spec = ((manifest or {}).get("graphs") or {}).get(role)
    candidate = spec.get("run_dir") if isinstance(spec, Mapping) else None
    if candidate:
        path = Path(candidate)
        return path if path.is_absolute() else (bundle_dir / path)
    return run_dir


def _circular_angle_error(bin_a: int, bin_b: int) -> float:
    difference = abs(int(bin_a) - int(bin_b)) % 360
    return float(min(difference, 360 - difference))


def _verify_classifier(
    bundle_dir: Path,
    manifest: Mapping[str, Any] | None,
    role: str,
    channels: int,
    scenarios: Sequence[Scenario],
    tolerances: Mapping[str, float],
    run_dir: Path | None,
    report: BundleReport,
) -> None:
    import onnx

    file_name, declared_outputs = _graph_spec(manifest, role)
    path = bundle_dir / file_name
    if not path.exists():
        report.findings.append(Finding("error", "graph_missing", f"缺少图：{path.name}"))
        return
    try:
        model = onnx.load(str(path))
        onnx.checker.check_model(model)
    except Exception as exc:  # noqa: BLE001 - 图本身非法时给结论而不是崩溃
        report.findings.append(Finding("error", "graph_invalid", f"{path.name}: {exc}"))
        return
    report.findings.extend(check_classifier_model(model, channels))
    if declared_outputs:
        report.findings.append(
            Finding(
                "info", "outputs_declared", f"{role} 的 outputs 声明暂不消费：{declared_outputs}"
            )
        )

    resolved_run = _resolve_run_dir(bundle_dir, manifest, role, run_dir)
    if resolved_run is None:
        report.findings.append(
            Finding(
                "warning",
                "classifier_reference_missing",
                f"{role}: 未提供 run_dir，仅做结构断言（数值比对需要 checkpoint）",
            )
        )
        return
    checkpoint = Path(resolved_run) / "best.pt"
    if not checkpoint.exists():
        report.findings.append(
            Finding("error", "checkpoint_missing", f"缺少 checkpoint：{checkpoint}")
        )
        return

    import torch

    from endfield.live import load_run_config
    from endfield.model import load_model
    from export_onnx import ExportWrapper, fold_input_conventions

    run_config = load_run_config(Path(resolved_run))
    expected_mode = "polar" if role == "polar" else "ref"
    if run_config.input_mode != expected_mode:
        report.findings.append(
            Finding(
                "warning",
                "classifier_mode_mismatch",
                f"{role}: run input_mode={run_config.input_mode!r}，跳过数值比对",
            )
        )
        return
    net = load_model(checkpoint, device="cpu")
    if net.in_channels != channels:
        report.findings.append(
            Finding(
                "error",
                "checkpoint_channels",
                f"{role}: checkpoint in_channels={net.in_channels} != {channels}",
            )
        )
        return
    fold_input_conventions(net)
    wrapper = ExportWrapper(net).eval()
    declared_input = model.graph.input[0].name
    for scenario in scenarios:
        observed, reference = reference_strips(scenario)
        strip = observed if channels == 3 else np.concatenate([observed, reference], axis=2)
        with torch.no_grad():
            expected_pmf = wrapper(torch.from_numpy(strip[None])).numpy()
        try:
            outputs = run_model(path, {declared_input: strip})
        except Exception as exc:  # noqa: BLE001 - 运行失败记为比对失败
            report.comparisons.append(
                CompareResult(f"{role}.pmf", False, False, None, None, None, None, 0.0, str(exc))
            )
            continue
        actual_pmf = next(iter(outputs.values()))
        comparison = compare_arrays(
            f"{scenario.name}.{role}.pmf",
            actual_pmf,
            expected_pmf,
            Tolerance(tolerances["pmf_float32"]),
        )
        report.comparisons.append(comparison)
        if comparison.max_abs is not None:
            bins = [int(array.reshape(-1).argmax()) for array in (actual_pmf, expected_pmf)]
            report.metrics[f"{scenario.name}.{role}.angle_error_deg"] = _circular_angle_error(*bins)


def verify_bundle(
    bundle_dir: Path,
    *,
    fixture_dir: Path | None = None,
    run_dir: Path | None = None,
    require: Iterable[str] | None = None,
) -> BundleReport:
    """校验一个 bundle：环境 → 结构 → 逐 fixture 数值比对。

    `require` 缺省时由 manifest 声明决定（无 manifest 仅要求 preprocess）。
    classifier 的数值比对需要 `run_dir`（checkpoint 参考）；缺失时只做结构断言。
    """
    bundle_dir = Path(bundle_dir)
    report = BundleReport(bundle=str(bundle_dir))
    manifest = _load_manifest(bundle_dir, report)
    report.findings.extend(check_environment(manifest))

    if manifest and manifest.get("definition_hash"):
        current = definition_hash()
        if manifest["definition_hash"] != current:
            report.findings.append(
                Finding(
                    "error",
                    "definition_hash",
                    "manifest 定义哈希与当前定义模块不一致：bundle 与源码不同源，需重导出",
                )
            )
    elif manifest:
        report.findings.append(Finding("warning", "definition_hash_missing", "manifest 无定义哈希"))

    if fixture_dir is not None:
        try:
            scenarios = load_fixtures(Path(fixture_dir))
        except (ValueError, KeyError) as exc:
            report.findings.append(Finding("error", "fixtures_invalid", f"{fixture_dir}: {exc}"))
            scenarios = []
    elif manifest and manifest.get("fixtures"):
        builtin = scenario_map()
        scenarios = []
        for entry in manifest["fixtures"]:
            name = entry.get("name") if isinstance(entry, Mapping) else entry
            if name not in builtin:
                report.findings.append(
                    Finding("error", "fixture_unknown", f"manifest 引用了未知 fixture {name!r}")
                )
                continue
            scenarios.append(builtin[name])
    else:
        scenarios = builtin_scenarios()

    if require is None:
        required = tuple((manifest.get("graphs") or {}).keys()) if manifest else ("preprocess",)
        if not required:
            required = ("preprocess",)
    else:
        required = tuple(require)
    known = {"preprocess", "polar", "polar_with_ref"}
    unknown_roles = sorted(set(required) - known)
    if unknown_roles:
        report.findings.append(Finding("error", "require_unknown", f"未知图角色：{unknown_roles}"))
        required = tuple(role for role in required if role in known)

    try:
        tolerances = resolve_tolerances(manifest)
    except ValueError as exc:
        report.findings.append(Finding("error", "tolerances_invalid", str(exc)))
        tolerances = dict(DEFAULT_TOLERANCES)
    if "preprocess" in required:
        _verify_preprocess(bundle_dir, manifest, scenarios, tolerances, report)

    channels = {"polar": 3, "polar_with_ref": 7}
    for role in required:
        if role in channels:
            _verify_classifier(
                bundle_dir, manifest, role, channels[role], scenarios, tolerances, run_dir, report
            )
    return report
