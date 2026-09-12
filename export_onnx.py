"""将训练好的 AzimuthNet checkpoint 导出为 ONNX 交付格式。

输入为 uint8 NHWC 条带，即 OpenCV remap 的原生输出布局，与训练契约同格式，
消费方无需任何格式转换即可零拷贝建张量：

- polar：`[1,42,360,3]`，观测极坐标展开；
- ref：`[1,42,360,7]`，`[obs.BGR, ref.BGR, ref.A]` 参考配对条带（参考合成见
  `endfield/ref.py`）。

/255 归一化对卷积是线性变换，折入首层卷积权重；图内仅保留 HWC→CHW 转置与
softmax。输出 Z/360Z 上的离散概率质量函数 [1,360]。循环 pad 原样导出为
Pad(wrap)，由 ONNX Runtime 支持。
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import torch
from torch import nn

from endfield.live import load_run_config
from endfield.model import load_model
from endfield.polar import IMG_H as POLAR_H
from endfield.polar import IMG_W as POLAR_W
from endfield.train.data import input_channels

ROOT = Path(__file__).resolve().parent

DESCRIPTIONS = {
    "polar": "AzimuthNet: Endfield minimap camera angle classifier on polar-unwrapped ring strips",
    "ref": (
        "AzimuthNet: Endfield minimap camera angle classifier on reference-pair "
        "(observed + MapLocator reference + alpha) ring strips"
    ),
}

POLAR_GEOMETRY = (
    "polar unwrap of the world-anchored minimap ring: angle->x (1 deg/column, clockwise, "
    "north = column 0); radius->y, inner radius on top; r_in=12, r_out=54 at the 720p "
    "baseline (1280x720), ring center at (108,111)"
)

INPUT_SPECS = {
    "polar": (
        f"uint8 [1,42,360,3] NHWC BGR. {POLAR_GEOMETRY}. values in [0,255]; /255 is folded "
        "into the first convolution weights, HWC->CHW is a Transpose inside the graph"
    ),
    "ref": (
        "uint8 [1,42,360,7] NHWC. Reference pair: channels = [obs.BGR, ref.BGR, ref.A]. "
        f"obs = {POLAR_GEOMETRY}. ref = MapLocator zone asset cropped at (x,y) with the "
        "zone's ZoneTemplateScale (ValleyIV_Base 15/16, otherwise 1:1), resized to 118x120, "
        "then unwrapped at the ROI center; ref.BGR = observed-backdrop composite "
        "black_ref + obs_roi*(1 - alpha/255) (alpha==0 -> observed pixels), ref.A = raw "
        "asset alpha (0 = reference gap). values in [0,255]; /255 is folded into the first "
        "convolution weights, HWC->CHW is a Transpose inside the graph"
    ),
}


def fold_input_conventions(net: nn.Module) -> None:
    """把 /255 归一化折入首层卷积权重。

    标量缩放对卷积是线性变换，模型首层卷积无偏置，折叠精确无损。
    模型内部训练契约是 BGR NCHW [0,1]，交付输入无需通道翻转。
    """
    conv0 = next(m for m in net.trunk if isinstance(m, nn.Conv2d))
    if conv0.bias is not None:
        raise ValueError("first conv is expected to be bias-free for exact folding")
    conv0.weight.data = conv0.weight.detach() / 255.0


class ExportWrapper(nn.Module):
    def __init__(self, net: nn.Module) -> None:
        super().__init__()
        self.net = net

    def forward(self, strip_bgr: torch.Tensor) -> torch.Tensor:
        features = strip_bgr.permute(0, 3, 1, 2).float()
        return torch.softmax(self.net(features), dim=1)


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def attach_metadata(
    path: Path, record: dict, summary: dict, checkpoint: Path, input_mode: str
) -> None:
    import onnx

    model = onnx.load(path)
    metadata = {
        "description": DESCRIPTIONS[input_mode],
        "input_mode": input_mode,
        "input_spec": INPUT_SPECS[input_mode],
        "output_spec": (
            "float32 [1,360] discrete probability mass function over azimuth bins, softmax "
            "already applied (sums to 1). Bin j corresponds to azimuth j degrees clockwise "
            "from north; bin 359 and bin 0 are adjacent (circular topology)"
        ),
        "source_checkpoint": str(checkpoint),
        "git_commit": git_commit(),
        "val_expected_abs_error_deg": f"{summary['val_expected_abs_error']:.6f}",
        "val_rms_error_deg": f"{summary['val_rms_error']:.6f}",
        "target_sigma_deg": str(record["target_sigma"]),
        "trainable_parameters": str(record["trainable_parameters"]),
    }
    for key, value in metadata.items():
        entry = model.metadata_props.add()
        entry.key = key
        entry.value = value
    onnx.save(model, path)


def export(checkpoint: Path, output: Path) -> None:
    run_dir = checkpoint.parent
    run_config = load_run_config(run_dir)
    channels = input_channels(run_config.input_mode)
    net = load_model(checkpoint, device="cpu")
    if net.in_channels != channels:
        raise ValueError(
            f"{checkpoint}: checkpoint has {net.in_channels} input channels but "
            f"input_mode {run_config.input_mode!r} expects {channels}"
        )
    fold_input_conventions(net)
    wrapper = ExportWrapper(net).eval()

    with open(run_dir / "record.json") as f:
        record = json.load(f)
    with open(run_dir / "summary.json") as f:
        summary = json.load(f)

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
    attach_metadata(output, record, summary, checkpoint, run_config.input_mode)
    print(f"exported: {output} (input_mode={run_config.input_mode}, channels={channels})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output = args.output or args.run_dir / "cameraorientation.onnx"
    export(args.run_dir / "best.pt", output)
