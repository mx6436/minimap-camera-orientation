"""MaaFw 实时截图 → 极坐标展开 → 摄像机角度预测 → 单窗口实时绘制。

置信度为 AzimuthNet 输出的 360 概率方向向量的合成模长乘以解码方向与合成
方向夹角的余弦（见 predict_angle）。

gamescope 实例通过 MaaToolkitGamescopeInstanceFindAll 自动发现：每个实例
以 $XDG_RUNTIME_DIR 下 gamescope-<n> 命名的 Wayland socket 为键，附带
PipeWire 节点 ID（gamescope_pipewire 协议）和同名 gamescope-<n>-ei EIS
socket 路径。

overlay 窗口另绘展示用圆形裁剪（完整圆盘，含中心圆与箭头）——仅给人看，
不进模型，模型永远看不到位于中心圆内的箭头（见 CONTEXT.md「采样一致性
假象」）。圆盘下方以函数曲线绘制 AzimuthNet 输出的 360 bin 概率分布：
横轴为定义域 [0,360)°，纵轴为概率，竖线标记解码角。
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np

import endfield.polar as polar
from endfield.model import choose_device, load_model, predict_probs

ROOT = Path(__file__).resolve().parent

DISPLAY_BOX = 112  # 外径 56 的外接正方形，720p 基准
DISPLAY_SCALE = 6
ARROW_LENGTH = 42
CONFIDENCE_THRESHOLD = 0.7
# 概率分布函数曲线：定义域 [0, 360)°，x 轴 2 px/°，纵轴自适应峰值
PLOT_WIDTH = 720
PLOT_HEIGHT = 160
PLOT_MARGIN_L = 48
PLOT_MARGIN_R = 12
PLOT_MARGIN_T = 12
PLOT_MARGIN_B = 28
AXIS_COLOR = (96, 96, 96)
GRID_COLOR = (48, 48, 48)
CURVE_COLOR = (0, 255, 0)

# Linux 控制器 config_json 字段值，枚举定义见 MaaFramework docs「2.4 控制方式说明」
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


def render_disc(rgb: np.ndarray, cx: float, cy: float, r_out: float) -> np.ndarray:
    """polar.unwrap 已保证 r_out + 1 的边距落在图内，故此处裁剪无需越界检查。"""
    left = int(round(cx - r_out))
    top = int(round(cy - r_out))
    right = int(round(cx + r_out))
    bottom = int(round(cy + r_out))
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
    return disc


def _nice_step(rough: float) -> float:
    """1/2/5×10^k 中不小于 rough 的最小值，供坐标轴刻度取整。"""
    exponent = math.floor(math.log10(rough))
    for mantissa in (1, 2, 5):
        candidate = mantissa * 10.0**exponent
        if candidate >= rough:
            return candidate
    return 10.0 ** (exponent + 1)


def draw_distribution(probs: np.ndarray, angle: float, marker_color: tuple) -> np.ndarray:
    """概率分布函数曲线：横轴定义域 [0,360)°，纵轴上限取 1.2 倍峰值、
    刻度按 1/2/5 步长自适应；竖线标记解码角。"""
    width = PLOT_MARGIN_L + PLOT_WIDTH + PLOT_MARGIN_R
    height = PLOT_MARGIN_T + PLOT_HEIGHT + PLOT_MARGIN_B
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    x0, y0 = PLOT_MARGIN_L, PLOT_MARGIN_T
    x1, y1 = x0 + PLOT_WIDTH, y0 + PLOT_HEIGHT
    y_max = float(probs.max()) * 1.2
    step = _nice_step(y_max / 3)
    tick = step
    while tick <= y_max * (1 + 1e-6):
        y = round(y1 - tick / y_max * PLOT_HEIGHT)
        cv2.line(canvas, (x0, y), (x1, y), GRID_COLOR, 1, cv2.LINE_AA)
        cv2.line(canvas, (x0 - 4, y), (x0, y), AXIS_COLOR, 1, cv2.LINE_AA)
        label = f"{tick:g}"
        (_, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.putText(
            canvas,
            label,
            (x0 - 10, y + th // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            AXIS_COLOR,
            1,
            cv2.LINE_AA,
        )
        tick += step
    cv2.line(canvas, (x0, y1), (x1, y1), AXIS_COLOR, 1, cv2.LINE_AA)
    cv2.line(canvas, (x0, y0), (x0, y1), AXIS_COLOR, 1, cv2.LINE_AA)
    cv2.putText(
        canvas,
        "0",
        (x0 - 10, y1 + 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        AXIS_COLOR,
        1,
        cv2.LINE_AA,
    )
    for deg in range(90, 361, 90):
        x = round(x0 + deg * PLOT_WIDTH / 360)
        cv2.line(canvas, (x, y1), (x, y1 + 4), AXIS_COLOR, 1, cv2.LINE_AA)
        label = str(deg)
        (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.putText(
            canvas,
            label,
            (x - tw // 2, y1 + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            AXIS_COLOR,
            1,
            cv2.LINE_AA,
        )
    # 分布以 bin 0/359 为循环接缝，函数在 360° 处取 p(0) 补全定义域端点
    values = np.append(probs, probs[0])
    xs = x0 + np.arange(361) * (PLOT_WIDTH / 360.0)
    ys = y1 - values / y_max * PLOT_HEIGHT
    points = np.stack([xs, ys], axis=1).astype(np.int32)
    cv2.polylines(canvas, [points], False, CURVE_COLOR, 1, cv2.LINE_AA)
    x = round(x0 + angle / 360 * PLOT_WIDTH)
    cv2.line(canvas, (x, y0), (x, y1), marker_color, 1, cv2.LINE_AA)
    return canvas


def draw_overlay(
    disc: np.ndarray, angle: float, confidence: float, probs: np.ndarray
) -> np.ndarray:
    color = (0, 255, 255) if confidence < CONFIDENCE_THRESHOLD else (0, 0, 255)
    size = disc.shape[0] * DISPLAY_SCALE
    display = cv2.cvtColor(
        cv2.resize(disc, (size, size), interpolation=cv2.INTER_NEAREST), cv2.COLOR_RGBA2BGR
    )
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
        f"angle={angle:.1f} conf={confidence:.2f}",
        (12, size - 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
        cv2.LINE_AA,
    )
    plot = draw_distribution(probs, angle, color)
    width = max(display.shape[1], plot.shape[1])
    padded = np.zeros((display.shape[0] + plot.shape[0], width, 3), dtype=np.uint8)
    left = (width - display.shape[1]) // 2
    padded[: display.shape[0], left : left + display.shape[1]] = display
    padded[display.shape[0] :, : plot.shape[1]] = plot
    return padded


def resolve_gamescope(instances: list, args: argparse.Namespace) -> tuple[int, str]:
    """instances 为 Toolkit.find_gamescope_instances() 的返回值。

    优先级：显式 --node-id / --eis-socket > 按 --display 匹配 > 唯一实例。
    """
    if args.node_id is not None and args.eis_socket is not None:
        return args.node_id, args.eis_socket

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
    model = load_model(args.checkpoint, device=choose_device(args.device))

    try:
        from maa.controller import LinuxController
        from maa.toolkit import Toolkit
    except ImportError as exc:
        raise SystemExit(
            "MaaFw 缺少 Linux 控制器支持（需要 MaaFw>=5.13.0b5）："
            "请在项目根目录执行 `uv sync` 更新依赖"
        ) from exc

    node_id, eis_socket = resolve_gamescope(Toolkit.find_gamescope_instances(), args)
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
            continue  # 空帧/截图失败是正常情况
        if frame.size == 0:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cx, cy, r_in, r_out = polar.scaled_roi(frame.shape[:2])
        strip = polar.unwrap(rgb, cx, cy, r_in, r_out)
        angle, confidence, probs = predict_probs(model, strip)
        disc = render_disc(rgb, cx, cy, r_out)
        cv2.imshow("minimap angle", draw_overlay(disc, angle, confidence, probs))
        if not printed_info:
            print(
                f"frame {frame.shape[1]}x{frame.shape[0]}, "
                f"ring outer radius = {r_out:.1f} px (scale x{r_out / polar.OUTER_R:.3f})"
            )
            printed_info = True
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
