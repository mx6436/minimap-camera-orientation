"""将训练好的 AzimuthNet checkpoint 导出为 ONNX（MaaEnd cpp-algo 交付格式）。

图内 bake /255 归一化与 softmax：输入 uint8 极坐标条带 [1,3,42,360]，
输出 Z/360Z 上的离散概率质量函数 [1,360]。循环 pad 原样导出为 Pad(wrap)，
由消费侧运行时（MaaEnd ONNX Runtime）支持。
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


class ExportWrapper(nn.Module):
    def __init__(self, net: nn.Module) -> None:
        super().__init__()
        self.net = net

    def forward(self, strip_uint8: torch.Tensor) -> torch.Tensor:
        features = strip_uint8.float() / 255.0
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
            "uint8 [1,3,42,360] NCHW RGB. Polar unwrap of the world-anchored minimap ring: "
            "angle->x (1 deg/column, clockwise, north = column 0); radius->y, inner radius on top. "
            "r_in=12, r_out=54 at the 720p baseline (1280x720), ring center at (108,111); "
            "values in [0,255]; /255 is applied inside the graph"
        ),
        "output_spec": (
            "float32 [1,360] discrete probability mass function over azimuth bins, softmax "
            "already applied (sums to 1). Bin j corresponds to azimuth j degrees clockwise "
            "from north; bin 359 and bin 0 are adjacent (circular topology)"
        ),
        "source_checkpoint": str(checkpoint),
        "git_commit": git_commit(),
        "val_expected_mae_deg": f"{summary['val_expected_mae']:.6f}",
        "val_expected_rmse_deg": f"{summary['val_expected_rmse']:.6f}",
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
    wrapper = ExportWrapper(net).eval()

    run_dir = checkpoint.parent
    with open(run_dir / "record.json") as f:
        record = json.load(f)
    with open(run_dir / "summary.json") as f:
        summary = json.load(f)

    dummy = torch.zeros(1, 3, POLAR_H, POLAR_W, dtype=torch.uint8)
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
