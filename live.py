"""MaaFw 实时截图 → 极坐标展开 → AngleCNN 角度预测 → 单窗口实时绘制。

gamescope 实例通过 MaaToolkitGamescopeInstanceFindAll 自动发现：每个实例
以 $XDG_RUNTIME_DIR 下 gamescope-<n> 命名的 Wayland socket 为键，附带
PipeWire 节点 ID（gamescope_pipewire 协议）和同名 gamescope-<n>-ei EIS
socket 路径。

overlay 窗口另绘展示用圆形裁剪（完整圆盘，含中心圆与箭头）——仅给人看，
不进模型，模型永远看不到位于中心圆内的箭头（见 CONTEXT.md「采样一致性
假象」）。
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

import polar  # noqa: E402
from data_utils import decode_angle, round_angle  # noqa: E402
from model import choose_device, load_model  # noqa: E402

DISPLAY_BOX = 112  # 展示用圆盘边长（外径 56 的外接正方形，720p 基准）
DISPLAY_SCALE = 6
ARROW_LENGTH = 42

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
    return load_model(checkpoint, device=choose_device(device))


def prepare_input(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    height, width = frame.shape[:2]
    cx, cy, r_in, r_out = polar.scaled_roi((height, width))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    strip = polar.unwrap(rgb, cx, cy, r_in, r_out)

    left = int(round(cx - r_out))
    top = int(round(cy - r_out))
    right = int(round(cx + r_out))
    bottom = int(round(cy + r_out))
    if not (0 <= left < right <= width and 0 <= top < bottom <= height):
        raise RuntimeError(
            f"ring crop box ({left},{top},{right},{bottom}) does not fit frame {width}x{height}"
        )

    box = rgb[top:bottom, left:right].copy()
    xs = np.arange(right - left) + left + 0.5 - cx
    ys = np.arange(bottom - top) + top + 0.5 - cy
    d2 = xs[None, :] ** 2 + ys[:, None] ** 2
    alpha = (d2 <= r_out**2).astype(np.uint8) * 255
    disc = np.dstack([box, alpha])
    disc[alpha == 0, :3] = 0
    if disc.shape[0] != DISPLAY_BOX or disc.shape[1] != DISPLAY_BOX:
        interpolation = cv2.INTER_AREA if disc.shape[0] > DISPLAY_BOX else cv2.INTER_LINEAR
        disc = cv2.resize(disc, (DISPLAY_BOX, DISPLAY_BOX), interpolation=interpolation)
    return strip, disc, float(r_out), r_out / polar.OUTER_R


CONFIDENCE_THRESHOLD = 0.7


def predict(model: torch.nn.Module, strip: np.ndarray) -> tuple[float, int, float]:
    array = strip.astype(np.float32) / 255.0
    features = torch.from_numpy(array.transpose(2, 0, 1)).unsqueeze(0)
    device = next(model.parameters()).device
    features = features.to(device)
    with torch.no_grad():
        output = model(features).cpu().numpy()
    continuous = float(decode_angle(output)[0])
    rounded = int(round_angle(continuous))
    norm = float(np.linalg.norm(output[0]))
    return continuous, rounded, norm


def draw_overlay(disc: np.ndarray, angle: float, norm: float) -> np.ndarray:
    color = (0, 255, 255) if norm < CONFIDENCE_THRESHOLD else (0, 0, 255)
    size = disc.shape[0] * DISPLAY_SCALE
    display = cv2.resize(disc, (size, size), interpolation=cv2.INTER_NEAREST)
    display = cv2.cvtColor(display, cv2.COLOR_RGBA2BGR)
    center = (size // 2, size // 2)
    radians = math.radians(angle)
    length = ARROW_LENGTH * DISPLAY_SCALE
    tip = (
        int(round(center[0] + math.sin(radians) * length)),
        int(round(center[1] - math.cos(radians) * length)),
    )
    cv2.line(display, center, tip, color, max(2, DISPLAY_SCALE // 3), cv2.LINE_AA)
    cv2.putText(
        display,
        f"angle={angle:.1f} (rounded {round_angle(angle)}) norm={norm:.2f}",
        (12, size - 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
        cv2.LINE_AA,
    )
    return display


def resolve_gamescope(toolkit: type, args: argparse.Namespace) -> tuple[int, str]:
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
            _, _, r_out, scale = prepare_input(frame)
            print(
                f"frame {frame.shape[1]}x{frame.shape[0]}, "
                f"ring outer radius = {r_out:.1f} px (scale x{scale:.3f})"
            )
            printed_info = True
        strip, disc, _, _ = prepare_input(frame)
        angle, _, norm = predict(model, strip)
        cv2.imshow("minimap angle", draw_overlay(disc, angle, norm))
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
