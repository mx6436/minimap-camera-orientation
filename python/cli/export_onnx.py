"""将训练好的 AzimuthNet checkpoint 导出为 ONNX 交付格式。

输入为 uint8 NHWC 参考配对条带 `[1,42,360,7]`，即 `[obs.BGR, ref.BGR, ref.A]`：
观测与参考条带都由 MaaEnd C++ 的 `BuildOrientationStrips` 在条带域一次产出，消费方无需
格式转换即可零拷贝建张量；参考语义的权威在 MaaEnd C++（`CameraOrientationPreprocess`）。

/255 归一化对卷积是线性变换，折入首层卷积权重；图内仅保留 HWC→CHW 转置与 softmax。
输出 Z/360Z 上的离散概率质量函数 [1,360]。循环 pad 原样导出为 Pad(wrap)，由 ONNX
Runtime 支持。

交付文件名固定 `polar_with_ref.onnx`；导出后即跑 ONNX IO 契约与 ORT 1.19.2 加载校验
（`endfield/onnx_contract.py`），不合格的图不会当作交付物落盘。
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import torch

from endfield import run_dir, run_record
from endfield.input_encoding import REFERENCE_STRIP_SEMANTICS
from endfield.model import ExportWrapper, fold_input_conventions, load_model
from endfield.paths import REPO_ROOT, repo_path
from endfield.strip import INPUT_CHANNELS, STRIP_H, STRIP_W

DESCRIPTION = (
    "AzimuthNet: Endfield minimap camera angle classifier on reference-pair "
    "(observed + MapLocator reference + alpha) ring strips"
)

STRIP_GEOMETRY = (
    "polar unwrap of the world-anchored minimap ring: angle->x (1 deg/column, clockwise, "
    "north = column 0); radius->y, inner radius on top; r_in=12, r_out=54 at the 720p "
    "baseline (1280x720), ring center at (108,111)"
)

INPUT_SPEC = (
    f"uint8 [1,{STRIP_H},{STRIP_W},{INPUT_CHANNELS}] NHWC. Reference pair: "
    f"channels = [obs.BGR, ref.BGR, ref.A]. obs = {STRIP_GEOMETRY}. "
    f"{REFERENCE_STRIP_SEMANTICS}. values in [0,255]; /255 is "
    "folded into the first convolution weights, HWC->CHW is a Transpose inside the graph"
)


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def attach_metadata(
    path: Path, record: run_record.RunRecord, summary: run_dir.TrainingSummary, checkpoint: Path
) -> None:
    import onnx

    model = onnx.load(path)
    metadata = {
        "description": DESCRIPTION,
        "input_spec": INPUT_SPEC,
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


def _validate_export(path: Path) -> None:
    """导出即校验：ONNX IO 契约 + ORT 1.19.2 可加载（与 MaaEnd 运行时同版本）。"""
    import onnx

    from endfield.onnx_contract import check_classifier_model, check_onnx_runtime

    model = onnx.load(str(path))
    problems = check_classifier_model(model)
    if problems:
        raise RuntimeError(
            f"{path}: classifier graph violates the delivery contract: {'; '.join(problems)}"
        )
    check_onnx_runtime(path)


def export(checkpoint: Path, output: Path | None = None) -> Path:
    run_path = checkpoint.parent
    record, summary = run_dir.load_run(run_path)
    net = load_model(checkpoint, device="cpu")
    if net.in_channels != INPUT_CHANNELS:
        raise ValueError(
            f"{checkpoint}: expected a {INPUT_CHANNELS}-channel ref classifier, "
            f"got {net.in_channels}"
        )
    fold_input_conventions(net)
    wrapper = ExportWrapper(net).eval()

    if output is None:
        output = run_dir.graph_path(run_path)
    output = Path(output)
    dummy = torch.zeros(1, STRIP_H, STRIP_W, INPUT_CHANNELS, dtype=torch.uint8)
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
    _validate_export(output)
    print(f"exported: {output} (channels={INPUT_CHANNELS})")
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="run 产物目录（相对路径按 <repo>/ 解析）",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=f"缺省 <run-dir>/{run_dir.GRAPH_NAME}",
    )
    args = parser.parse_args(argv)
    export(run_dir.checkpoint_path(repo_path(args.run_dir)), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
