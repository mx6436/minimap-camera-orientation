#!/usr/bin/env python3
"""MaaFw 实时截图 → 环形小地图预处理 → AngleCNN 角度预测 → 单窗口实时绘制。

截图通道：MaaFramework Python 绑定（MaaFw）的 Linux 控制器，采用与
MaaEnd 的 Linux-Gamescope 控制器一致的方式：PipeWire 会话 daemon 节点
截图（screencap_method=PipeWire）+ Libei 输入（input_method=Libei）。

gamescope 实例通过 MaaToolkitGamescopeInstanceFindAll 自动发现：每个实例
以 $XDG_RUNTIME_DIR 下 gamescope-<n> 命名的 Wayland socket 为键，附带
PipeWire 节点 ID（gamescope_pipewire 协议）和同名 gamescope-<n>-ei EIS
socket 路径。默认自动选择唯一的实例，也可用 --display / --node-id 指定。

预处理与训练数据完全一致（见 crop_ring.py）：ROI（中心 (108,111)、内径 12、
外径 56，720p 基准）按实际截图尺寸等比缩放，裁 112x112 外接正方形，环形硬
掩膜，RGBA 输出，透明处 RGB 清零。缩放后的环先按原始分辨率裁出，再缩放至
112x112 输入模型；若实际分辨率恰为 1280x720，则与训练预处理逐像素一致。

窗口仅绘制一条角度直线（0°=正上方，顺时针增加）与角度文字。
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from crop_ring import ROI_CENTER, INNER_R, OUTER_R  # noqa: E402
from data_utils import decode_angle, round_angle  # noqa: E402
from model import AngleCNN, EXPECTED_PARAMETER_COUNT, count_trainable_parameters  # noqa: E402
from train import choose_device, load_checkpoint  # noqa: E402

BASE_W, BASE_H = 1280, 720  # 训练基准分辨率
DISPLAY_SCALE = 6  # 112x112 裁图放大倍数
ARROW_LENGTH = 42  # 箭头长度（112x112 裁图坐标系内）

# Linux 控制器 config_json 字段值，见 MaaFramework docs 2.4-控制方式说明：
# Screencap: Wlr=1, PipeWire=4；Input: Wlr=1, UInput=2, Libei=4
SCREENCAP_PIPEWIRE = 4
INPUT_LIBEI = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        default="production_001",
        help="runs 目录下的模型子目录名（读取 runs/<run>/best.pt，默认 production_001）",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="模型 checkpoint 完整路径；省略时使用 runs/<run>/best.pt",
    )
    parser.add_argument("--device", default=None, help="推理设备，默认自动选择")
    parser.add_argument(
        "--display",
        type=int,
        default=None,
        help="gamescope display 号（gamescope-<n> 的 n）；省略时自动选择唯一实例",
    )
    parser.add_argument(
        "--node-id",
        type=int,
        default=None,
        help="PipeWire 节点 ID；省略时从 gamescope 实例自动获取",
    )
    parser.add_argument(
        "--eis-socket",
        default=None,
        help="EIS socket 路径；省略时从 gamescope 实例自动获取",
    )
    return parser.parse_args()


def prepare_model(checkpoint: Path, device: str | None) -> torch.nn.Module:
    device = choose_device(device)
    model = AngleCNN().to(device)
    if count_trainable_parameters(model) != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError("unexpected model parameter count")
    load_checkpoint(checkpoint, model, device=device)
    model.eval()
    return model


def scaled_roi(frame_shape: tuple[int, int]) -> tuple[float, float, float, float]:
    """按实际截图尺寸等比缩放 ROI。

    返回 (cx, cy, r_in, r_out)，均为浮点像素坐标；半径按 x 方向缩放，
    若 y 方向缩放与 x 差异超过 1% 则打印警告（非等比缩放会破坏环形状）。
    """
    height, width = frame_shape[:2]
    sx, sy = width / BASE_W, height / BASE_H
    if abs(sx - sy) / max(sx, sy) > 0.01:
        print(f"WARNING: non-uniform scale sx={sx:.4f} sy={sy:.4f}; ring will be distorted")
    cx, cy = ROI_CENTER[0] * sx, ROI_CENTER[1] * sy
    return cx, cy, INNER_R * sx, OUTER_R * sx


def crop_ring(frame: np.ndarray, box: int) -> tuple[np.ndarray, float, float]:
    """按训练预处理从一帧截图裁出环形小地图。

    frame: MaaFw 截图（BGR）。返回 (112x112 RGBA 环, 实测环外径, 缩放比例)；
    缩放比例 = 外径像素 / 56，用于打印核对与训练分布是否吻合。
    """
    height, width = frame.shape[:2]
    cx, cy, r_in, r_out = scaled_roi((height, width))
    left = int(round(cx - r_out))
    top = int(round(cy - r_out))
    right = int(round(cx + r_out))
    bottom = int(round(cy + r_out))
    if not (0 <= left < right <= width and 0 <= top < bottom <= height):
        raise RuntimeError(
            f"ring crop box ({left},{top},{right},{bottom}) does not fit frame {width}x{height}"
        )

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    box_arr = rgb[top:bottom, left:right].copy()

    # 环形硬掩膜：按像素中心距离平方比较，与 crop_ring.ring_mask 一致。
    xs = np.arange(right - left) + left + 0.5 - cx
    ys = np.arange(bottom - top) + top + 0.5 - cy
    d2 = xs[None, :] ** 2 + ys[:, None] ** 2
    keep = (d2 >= r_in**2) & (d2 <= r_out**2)

    alpha = (keep * 255).astype(np.uint8)
    rgba = np.dstack([box_arr, alpha])
    rgba[alpha == 0, :3] = 0

    if rgba.shape[0] != box or rgba.shape[1] != box:
        interpolation = cv2.INTER_AREA if rgba.shape[0] > box else cv2.INTER_LINEAR
        rgba = cv2.resize(rgba, (box, box), interpolation=interpolation)
    return rgba, float(r_out), r_out / OUTER_R


def predict(model: torch.nn.Module, rgba: np.ndarray) -> tuple[float, int]:
    array = rgba.astype(np.float32) / 255.0
    features = torch.from_numpy(array.transpose(2, 0, 1)).unsqueeze(0)
    device = next(model.parameters()).device
    features = features.to(device)
    with torch.no_grad():
        output = model(features).cpu().numpy()
    continuous = float(decode_angle(output)[0])
    rounded = int(round_angle(continuous))
    return continuous, rounded


def draw_overlay(ring: np.ndarray, angle: float) -> np.ndarray:
    """绘制放大环 + 角度直线 + 角度文字。0°=正上，顺时针。"""
    size = ring.shape[0] * DISPLAY_SCALE
    display = cv2.resize(ring, (size, size), interpolation=cv2.INTER_NEAREST)
    display = cv2.cvtColor(display, cv2.COLOR_RGBA2BGR)
    center = (size // 2, size // 2)
    radians = math.radians(angle)
    length = ARROW_LENGTH * DISPLAY_SCALE
    tip = (
        int(round(center[0] + math.sin(radians) * length)),
        int(round(center[1] - math.cos(radians) * length)),
    )
    cv2.line(display, center, tip, (0, 0, 255), max(2, DISPLAY_SCALE // 3), cv2.LINE_AA)
    cv2.putText(
        display,
        f"angle={angle:.1f} (rounded {round_angle(angle)})",
        (12, size - 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    return display


def resolve_gamescope(toolkit: type, args: argparse.Namespace) -> tuple[int, str]:
    """确定 PipeWire 节点 ID 与 EIS socket 路径。

    命令行显式指定的值优先；否则枚举 gamescope 实例，--display 匹配或唯一
    实例自动选中。节点 ID 为 0（无截图节点）或 EIS socket 为空时报错。
    """
    if args.node_id is not None and args.eis_socket is not None:
        return args.node_id, args.eis_socket

    instances = toolkit.find_gamescope_instances()
    if not instances:
        raise SystemExit("未发现 gamescope 实例，请确认 gamescope 正在运行")

    instance = None
    if args.display is not None:
        for cand in instances:
            if cand.display_no == args.display:
                instance = cand
                break
        if instance is None:
            found = [c.display_no for c in instances]
            raise SystemExit(f"未找到 gamescope-{args.display}，当前实例: {found}")
    elif len(instances) == 1:
        instance = instances[0]
    else:
        found = [c.display_no for c in instances]
        raise SystemExit(f"发现多个 gamescope 实例 {found}，请用 --display 指定")

    node_id = args.node_id if args.node_id is not None else instance.pipewire_node_id
    eis_socket = args.eis_socket if args.eis_socket is not None else instance.eis_socket_path
    if not node_id:
        raise SystemExit(f"gamescope-{instance.display_no} 无可用 PipeWire 截图节点")
    if not eis_socket:
        raise SystemExit(f"gamescope-{instance.display_no} 无可用 EIS socket")
    return node_id, eis_socket


def main() -> None:
    args = parse_args()
    if args.checkpoint is None:
        args.checkpoint = ROOT / "runs" / args.run / "best.pt"
    model = prepare_model(args.checkpoint, args.device)

    try:
        from maa.controller import LinuxController
        from maa.toolkit import Toolkit
    except ImportError as exc:
        raise SystemExit(
            "MaaFw 缺少 Linux 控制器支持（需要 MaaFw>=5.13.0b5）："
            "请在项目根目录执行 `uv sync` 更新依赖"
        ) from exc

    node_id, eis_socket = resolve_gamescope(Toolkit, args)
    controller = LinuxController(
        {
            "screencap_method": SCREENCAP_PIPEWIRE,
            "input_method": INPUT_LIBEI,
            "pw_node_id": node_id,
            "eis_socket_path": eis_socket,
            "use_win32_vk_code": True,
        }
    )
    connection = controller.post_connection().wait()
    if not connection.status.succeeded:
        raise RuntimeError(f"无法连接 gamescope 节点（status: {connection.status}）")
    print(f"connected: pw_node_id={node_id}, eis_socket={eis_socket}")

    printed_info = False
    while True:
        try:
            frame = controller.post_screencap().get()
        except RuntimeError:
            continue  # 空帧/截图失败是正常情况，跳过
        if frame.size == 0:
            continue
        if not printed_info:
            _, r_out, scale = crop_ring(frame, 112)
            print(
                f"frame {frame.shape[1]}x{frame.shape[0]}, "
                f"ring outer radius = {r_out:.1f} px (scale x{scale:.3f})"
            )
            printed_info = True
        ring, _, _ = crop_ring(frame, 112)
        angle, _ = predict(model, ring)
        cv2.imshow("minimap angle", draw_overlay(ring, angle))
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
