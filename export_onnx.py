"""将训练好的 AzimuthNet checkpoint 导出为 ONNX 交付格式。

输入为 uint8 BGR HWC [1,42,360,3] 极坐标条带，即 OpenCV remap 的原生输出
布局，与训练契约同格式，消费方无需任何格式转换即可零拷贝建张量。/255
归一化对卷积是线性变换，折入首层卷积权重；图内仅保留 HWC→CHW 转置与
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

from endfield.model import load_model
from endfield.polar import IMG_H as POLAR_H
from endfield.polar import IMG_W as POLAR_W

ROOT = Path(__file__).resolve().parent

DESCRIPTION = "AzimuthNet: Endfield minimap camera angle classifier on polar-unwrapped ring strips"


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


def attach_metadata(path: Path, record: dict, summary: dict, checkpoint: Path) -> None:
    import onnx

    model = onnx.load(path)
    metadata = {
        "description": DESCRIPTION,
        "input_spec": (
            "uint8 [1,42,360,3] NHWC BGR. Polar unwrap of the world-anchored minimap ring: "
            "angle->x (1 deg/column, clockwise, north = column 0); radius->y, inner radius on top. "
            "r_in=12, r_out=54 at the 720p baseline (1280x720), ring center at (108,111); "
            "values in [0,255]; /255 is folded into the first convolution "
            "weights, HWC->CHW is a Transpose inside the graph"
        ),
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
    net = load_model(checkpoint, device="cpu")
    fold_input_conventions(net)
    wrapper = ExportWrapper(net).eval()

    run_dir = checkpoint.parent
    with open(run_dir / "record.json") as f:
        record = json.load(f)
    with open(run_dir / "summary.json") as f:
        summary = json.load(f)

    dummy = torch.zeros(1, POLAR_H, POLAR_W, 3, dtype=torch.uint8)
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
    print(f"exported: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, default=ROOT / "runs" / "production_001" / "best.pt"
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output = args.output or args.checkpoint.parent / "cameraorientation.onnx"
    export(args.checkpoint, output)
