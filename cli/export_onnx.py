"""将训练好的 AzimuthNet checkpoint 导出为 ONNX 交付格式。

输入为 uint8 NHWC 条带，即 OpenCV remap 的原生输出布局，与训练契约同格式，
消费方无需任何格式转换即可零拷贝建张量：

- polar：`[1,42,360,3]`，观测极坐标展开（交付文件名 `polar.onnx`）；
- ref：`[1,42,360,7]`，`[obs.BGR, ref.BGR, ref.A]` 参考配对条带（交付文件名
  `polar_with_ref.onnx`，参考语义见 `endfield/preprocess.py`）。

/255 归一化对卷积是线性变换，折入首层卷积权重；图内仅保留 HWC→CHW 转置与
softmax。输出 Z/360Z 上的离散概率质量函数 [1,360]。循环 pad 原样导出为
Pad(wrap)，由 ONNX Runtime 支持。

导出后即跑结构断言（`endfield.conformance.check_classifier_model`）并在
ORT 1.19.2 加载校验，不合格的图不会当作交付物落盘。
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import torch

from endfield import bundle, run_dir, run_record
from endfield.model import ExportWrapper, fold_input_conventions, load_model
from endfield.preprocess import IMG_H as POLAR_H
from endfield.preprocess import IMG_W as POLAR_W
from endfield.run_record import InputMode, RunRecord

ROOT = Path(__file__).resolve().parents[1]

DESCRIPTIONS = {
    InputMode.POLAR: (
        "AzimuthNet: Endfield minimap camera angle classifier on polar-unwrapped ring strips"
    ),
    InputMode.REF: (
        "AzimuthNet: Endfield minimap camera angle classifier on reference-pair "
        "(observed + MapLocator reference + alpha) ring strips"
    ),
}

# 交付文件名由 `endfield/bundle.py` 按交付角色单点持有（MaaEnd 交付布局）

POLAR_GEOMETRY = (
    "polar unwrap of the world-anchored minimap ring: angle->x (1 deg/column, clockwise, "
    "north = column 0); radius->y, inner radius on top; r_in=12, r_out=54 at the 720p "
    "baseline (1280x720), ring center at (108,111)"
)

INPUT_SPECS = {
    InputMode.POLAR: (
        f"uint8 [1,42,360,3] NHWC BGR. {POLAR_GEOMETRY}. values in [0,255]; /255 is folded "
        "into the first convolution weights, HWC->CHW is a Transpose inside the graph"
    ),
    InputMode.REF: (
        "uint8 [1,42,360,7] NHWC. Reference pair: channels = [obs.BGR, ref.BGR, ref.A]. "
        f"obs = {POLAR_GEOMETRY}. ref = MapLocator zone asset sampled once on the strip grid "
        "at (x,y)+(q_roi-pole)*scale with the zone's ZoneTemplateScale (ValleyIV_Base 15/16, "
        "otherwise 1:1); out-of-bounds reads 0 (reference gap); ref.BGR = "
        "rgb*(a/255) + 255*(1-a/255) composited once in the strip domain over a white backdrop "
        "(transparent pixels, including out-of-bounds reads, are white), ref.A = raw asset "
        "alpha (0 = reference gap); both streams are defined by "
        "endfield/preprocess.py and delivered by preprocess.onnx. values in [0,255]; /255 is "
        "folded into the first convolution weights, HWC->CHW is a Transpose inside the graph"
    ),
}


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def attach_metadata(
    path: Path, record: RunRecord, summary: run_dir.TrainingSummary, checkpoint: Path
) -> None:
    import onnx

    model = onnx.load(path)
    metadata = {
        "description": DESCRIPTIONS[record.input_mode],
        "input_mode": record.input_mode.value,
        "input_spec": INPUT_SPECS[record.input_mode],
        "output_spec": (
            "float32 [1,360] discrete probability mass function over azimuth bins, softmax "
            "already applied (sums to 1). Bin j corresponds to azimuth j degrees clockwise "
            "from north; bin 359 and bin 0 are adjacent (circular topology)"
        ),
        "source_checkpoint": str(checkpoint),
        "git_commit": git_commit(),
        "val_expected_abs_error_deg": f"{summary.metrics.expected_abs_error:.6f}",
        "val_rms_error_deg": f"{summary.metrics.rms_error:.6f}",
        "target_sigma_deg": str(record.target_sigma),
        "trainable_parameters": str(record.trainable_parameters),
    }
    for key, value in metadata.items():
        entry = model.metadata_props.add()
        entry.key = key
        entry.value = value
    onnx.save(model, path)


def _validate_export(path: Path, channels: int) -> None:
    """导出即校验：结构断言 + ORT 1.19.2 可加载（与 MaaEnd 运行时同版本）。"""
    import onnx
    import onnxruntime as ort

    from endfield.conformance import check_classifier_model

    model = onnx.load(str(path))
    findings = check_classifier_model(model, channels)
    errors = [finding for finding in findings if finding.level == "error"]
    if errors:
        messages = "; ".join(f"{finding.code}: {finding.message}" for finding in errors)
        raise RuntimeError(f"{path}: classifier graph fails the delivery contract: {messages}")
    ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def export(checkpoint: Path, output: Path | None = None) -> Path:
    run_path = checkpoint.parent
    record, summary = run_dir.load_run(run_path)
    channels = run_record.input_channels(record.input_mode)
    net = load_model(checkpoint, device="cpu")
    try:
        run_record.validate_channels(record, net.in_channels)
    except ValueError as error:
        raise ValueError(f"{checkpoint}: {error}") from None
    fold_input_conventions(net)
    wrapper = ExportWrapper(net).eval()

    if output is None:
        output = run_path / bundle.graph_file(bundle.role_for_mode(record.input_mode))
    output = Path(output)
    dummy = torch.zeros(1, POLAR_H, POLAR_W, channels, dtype=torch.uint8)
    torch.onnx.export(
        wrapper,
        (dummy,),
        output,
        opset_version=18,
        input_names=["strip"],
        output_names=["pmf"],
        external_data=False,
    )
    attach_metadata(output, record, summary, checkpoint)
    _validate_export(output, channels)
    print(f"exported: {output} (input_mode={record.input_mode.value}, channels={channels})")
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="缺省 <run-dir>/polar.onnx 或 <run-dir>/polar_with_ref.onnx（按 input_mode）",
    )
    args = parser.parse_args(argv)
    export(run_dir.checkpoint_path(args.run_dir), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
