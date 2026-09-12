"""MaaFw 实时截图 → 模型输入 → 摄像机角度预测 → 单窗口实时绘制。

输入模式由 run 的 record.json 决定（见 endfield/live.py）：polar 每帧直接极坐标
展开；ref 依赖 MapLocator 流式定位——定位跑在独立线程里（map-locate --stream
常驻子进程，一帧路径进、一条 JSONL 出，跨帧不重置追踪状态），主循环只取最新结果，
用参考底图裁出同视野参考，按 `[obs.BGR, ref.BGR, ref.A]` 拼接 7 通道张量后喂
AzimuthNet；定位不可用（失败 / held / 低分 / 资产缺失）时显示等待态，不显示过期角度。

置信度为 AzimuthNet 输出的 360 概率方向向量的合成模长乘以解码方向与合成
方向夹角的余弦（见 predict_angle）。

gamescope 实例通过 MaaToolkitGamescopeInstanceFindAll 自动发现：每个实例
以 $XDG_RUNTIME_DIR 下 gamescope-<n> 命名的 Wayland socket 为键，附带
PipeWire 节点 ID（gamescope_pipewire 协议）和同名 gamescope-<n>-ei EIS
socket 路径。

overlay 窗口另绘展示用圆形裁剪（完整圆盘，含中心圆与箭头）——仅给人看，
不进模型，模型永远看不到位于中心圆内的箭头（见 CONTEXT.md「采样一致性
假象」）。圆盘下方依次是当前模型输入（极坐标展开 / ref 的观测与参考两路）与
AzimuthNet 输出的 360 bin 概率分布：横轴为定义域 [0,360)°，纵轴为概率，
竖线标记解码角。
"""

from __future__ import annotations

import argparse
import math
import queue
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

import endfield.polar as polar
from endfield.live import (
    MissingZoneAsset,
    load_run_config,
    ref_strip_at,
    to_base_frame,
)
from endfield.locate import LocalizerStream, accept
from endfield.model import choose_device, load_model, predict_probs

DISPLAY_BOX = 108  # 外径 54 的外接正方形，720p 基准
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

# 模型输入条带展示：42x360 -> 2x 最近邻，标题一行
STRIP_SCALE = 2
STRIP_LABEL_H = 24
# ref 输入一栏按通道分行展示的子标题
REF_ROW_LABELS = ("obs.BGR", "ref.BGR", "ref.A")

# MapLocator 流式定位 CLI 与资源（gitignored 本地工作台，重建见 local/maplocator/README.local.md）
LOCATOR_CLI = Path(__file__).resolve().parent / "local" / "maplocator" / "bin" / "map-locate"
LOCATOR_RESOURCE = Path(__file__).resolve().parent / "local" / "maplocator" / "resource"

# Linux 控制器 config_json 字段值，枚举定义见 MaaFramework docs「2.4 控制方式说明」
SCREENCAP_PIPEWIRE = 4
INPUT_LIBEI = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="run 产物目录，读取其中的 best.pt",
    )
    parser.add_argument("--device", default=None, help="推理设备，默认自动选择")
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="绘制帧率上限（fps），默认 30；设为 0 表示不限速",
    )
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
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=None,
        help="拿到首个有效定位后保存一张 overlay 到该路径并退出（实机 smoke 取证用）",
    )
    return parser.parse_args()


def render_disc(bgr: np.ndarray, cx: float, cy: float, r_out: float) -> np.ndarray:
    """polar.unwrap 已保证 r_out + 1 的边距落在图内，故此处裁剪无需越界检查。"""
    left = int(round(cx - r_out))
    top = int(round(cy - r_out))
    right = int(round(cx + r_out))
    bottom = int(round(cy + r_out))
    box = bgr[top:bottom, left:right].copy()
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


def _input_panel(strip: np.ndarray | None, label: str) -> np.ndarray:
    """模型输入一栏：标题 + 2x 最近邻放大的极坐标条带；无输入时留空。"""
    width = polar.IMG_W * STRIP_SCALE
    height = STRIP_LABEL_H + polar.IMG_H * STRIP_SCALE
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(panel, label, (4, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, AXIS_COLOR, 1, cv2.LINE_AA)
    if strip is None:
        cv2.putText(
            panel,
            "no model input yet",
            (16, STRIP_LABEL_H + polar.IMG_H),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            AXIS_COLOR,
            1,
            cv2.LINE_AA,
        )
        return panel
    big = cv2.resize(strip, (width, polar.IMG_H * STRIP_SCALE), interpolation=cv2.INTER_NEAREST)
    panel[STRIP_LABEL_H:, :] = big
    return panel


def _ref_input_panel(strip: np.ndarray | None, label: str) -> np.ndarray:
    """ref 模型输入一栏：obs.BGR / ref.BGR / ref.A 三个 2x 最近邻条带分行展示。"""
    width = polar.IMG_W * STRIP_SCALE
    row_h = polar.IMG_H * STRIP_SCALE
    height = STRIP_LABEL_H * (1 + len(REF_ROW_LABELS)) + row_h * len(REF_ROW_LABELS)
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(panel, label, (4, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, AXIS_COLOR, 1, cv2.LINE_AA)
    if strip is None:
        cv2.putText(
            panel,
            "no model input yet",
            (16, height // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            AXIS_COLOR,
            1,
            cv2.LINE_AA,
        )
        return panel
    planes = (strip[..., :3], strip[..., 3:6], cv2.cvtColor(strip[..., 6], cv2.COLOR_GRAY2BGR))
    y = STRIP_LABEL_H
    for row_label, plane in zip(REF_ROW_LABELS, planes, strict=True):
        cv2.putText(
            panel,
            row_label,
            (4, y + 17),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            AXIS_COLOR,
            1,
            cv2.LINE_AA,
        )
        y += STRIP_LABEL_H
        panel[y : y + row_h, :] = cv2.resize(
            plane, (width, row_h), interpolation=cv2.INTER_NEAREST
        )
        y += row_h
    return panel


def _model_input_panel(mode: str, strip: np.ndarray | None, label: str) -> np.ndarray:
    """按 run 的输入模式渲染「当前模型输入」一栏。"""
    if mode == "ref":
        return _ref_input_panel(strip, label)
    return _input_panel(strip, label)


def _compose(disc: np.ndarray, panel: np.ndarray, plot: np.ndarray) -> np.ndarray:
    """圆盘 / 模型输入 / 概率曲线三栏纵向居中堆叠。"""
    width = max(disc.shape[1], panel.shape[1], plot.shape[1])
    height = disc.shape[0] + panel.shape[0] + plot.shape[0]
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    y = 0
    for part in (disc, panel, plot):
        left = (width - part.shape[1]) // 2
        canvas[y : y + part.shape[0], left : left + part.shape[1]] = part
        y += part.shape[0]
    return canvas


def draw_overlay(
    disc: np.ndarray,
    angle: float,
    confidence: float,
    probs: np.ndarray,
    panel: np.ndarray,
) -> np.ndarray:
    color = (0, 255, 255) if confidence < CONFIDENCE_THRESHOLD else (0, 0, 255)
    size = disc.shape[0] * DISPLAY_SCALE
    display = cv2.cvtColor(
        cv2.resize(disc, (size, size), interpolation=cv2.INTER_NEAREST), cv2.COLOR_BGRA2BGR
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
    return _compose(display, panel, plot)


def draw_waiting(disc: np.ndarray, message: str, panel: np.ndarray) -> np.ndarray:
    """无可用定位：展示等待态（圆盘 + 原因），不显示过期角度与概率曲线。"""
    size = disc.shape[0] * DISPLAY_SCALE
    display = cv2.cvtColor(
        cv2.resize(disc, (size, size), interpolation=cv2.INTER_NEAREST), cv2.COLOR_BGRA2BGR
    )
    plot_width = PLOT_MARGIN_L + PLOT_WIDTH + PLOT_MARGIN_R
    plot_height = PLOT_MARGIN_T + PLOT_HEIGHT + PLOT_MARGIN_B
    plot = np.zeros((plot_height, plot_width, 3), dtype=np.uint8)
    (tw, th), _ = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.putText(
        plot,
        message,
        ((plot_width - tw) // 2, (plot_height + th) // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return _compose(display, panel, plot)


@dataclass(frozen=True)
class Localization:
    """一帧 720p 基准画面与其 MapLocator 定位记录的配对。"""

    frame: np.ndarray
    record: dict


class LocatorWorker(threading.Thread):
    """独立线程：把最新帧喂给 map-locate --stream，发布最新定位结果。

    只保留一帧待处理（新帧顶替旧帧），显示循环永不阻塞在定位上；latest() 返回
    最近一次结果，供主循环构造模型输入。
    """

    def __init__(self, stream: LocalizerStream) -> None:
        super().__init__(name="map-locate-stream", daemon=True)
        self._stream = stream
        self._frames: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest: Localization | None = None
        self._error: Exception | None = None

    def start(self) -> None:
        """先拉起 map-locate 进程，再起喂帧/读结果的线程。"""
        self._stream.start()
        super().start()

    def submit(self, frame: np.ndarray) -> None:
        try:
            self._frames.put_nowait(frame)
        except queue.Full:
            try:
                self._frames.get_nowait()
            except queue.Empty:
                pass
            self._frames.put_nowait(frame)

    def latest(self) -> Localization | None:
        with self._lock:
            return self._latest

    def error(self) -> Exception | None:
        with self._lock:
            return self._error

    def stop(self) -> None:
        self._stop.set()
        try:
            self._frames.put_nowait(None)
        except queue.Full:
            pass
        self.join(timeout=5)
        self._stream.close()

    def run(self) -> None:
        while not self._stop.is_set():
            try:
                frame = self._frames.get(timeout=0.1)
            except queue.Empty:
                continue
            if frame is None:
                break
            try:
                record = self._stream.locate(frame)
            except Exception as exc:  # 进程死亡等：留给主循环报错
                with self._lock:
                    self._error = exc
                return
            with self._lock:
                self._latest = Localization(frame, record)


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
    run_config = load_run_config(args.run_dir)
    mode = run_config.input_mode
    localized_mode = mode == "ref"
    device = choose_device(args.device)
    model = load_model(args.run_dir / "best.pt", device=device)
    input_label = f"model input: {mode}"

    try:
        from maa.controller import LinuxController
        from maa.toolkit import Toolkit
    except ImportError as exc:
        raise SystemExit(
            "MaaFw 缺少 Linux 控制器支持（需要 MaaFw>=5.13.0b5）："
            "请在项目根目录执行 `uv sync` 更新依赖"
        ) from exc

    if localized_mode and (not LOCATOR_CLI.exists() or not LOCATOR_RESOURCE.is_dir()):
        raise SystemExit(
            f"{mode} 实机推理需要 {LOCATOR_CLI} 与 {LOCATOR_RESOURCE}；"
            "见 local/maplocator/README.local.md"
        )

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
    if localized_mode:
        print(f"input_mode={mode}, assets_root={run_config.assets_root}")

    assets_cache: dict[Path, np.ndarray] = {}
    work_dir = tempfile.TemporaryDirectory(prefix="live-locator-")
    worker: LocatorWorker | None = None
    printed_info = False
    interval = 1.0 / args.fps if args.fps > 0 else 0.0
    ready_result: Localization | None = None
    ready_overlay: np.ndarray | None = None

    try:
        if localized_mode:
            worker = LocatorWorker(
                LocalizerStream([LOCATOR_CLI], LOCATOR_RESOURCE, Path(work_dir.name))
            )
            worker.start()

        while True:
            started = time.monotonic()
            try:
                frame = controller.post_screencap().get()
            except RuntimeError:
                continue  # 空帧/截图失败是正常情况
            if frame.size == 0:
                continue
            cx, cy, r_in, r_out = polar.scaled_roi(frame.shape[:2])
            if not printed_info:
                print(
                    f"frame {frame.shape[1]}x{frame.shape[0]}, "
                    f"ring outer radius = {r_out:.1f} px (scale x{r_out / polar.OUTER_R:.3f})"
                )
                printed_info = True

            if localized_mode:
                assert worker is not None
                assert run_config.assets_root is not None
                # 定位线程只管跑最新帧；主循环拿到什么画什么，不阻塞在定位上
                worker.submit(to_base_frame(frame))
                error = worker.error()
                if error is not None:
                    raise SystemExit(f"map-locate 流式进程失败: {error}")
                localization = worker.latest()
                status = None
                if localization is None:
                    status = "waiting for MapLocator..."
                else:
                    accepted, reason = accept(localization.record)
                    if not accepted:
                        status = f"localization unavailable: {reason}"
                        if reason == "below_loc_threshold":
                            value = float(localization.record.get("locConf", 0.0))
                            status += f" (locConf={value:.3f})"
                    elif localization is not ready_result:
                        try:
                            strip = ref_strip_at(
                                localization.frame,
                                localization.record,
                                run_config.assets_root,
                                assets_cache,
                            )
                            angle, confidence, probs = predict_probs(model, strip)
                        except MissingZoneAsset as exc:
                            status = f"localization unavailable: {exc.zone} asset missing"
                        else:
                            local_cx, local_cy, _, local_r_out = polar.scaled_roi(
                                localization.frame.shape[:2]
                            )
                            ready_overlay = draw_overlay(
                                render_disc(localization.frame, local_cx, local_cy, local_r_out),
                                angle,
                                confidence,
                                probs,
                                _model_input_panel(mode, strip, input_label),
                            )
                            ready_result = localization
                if status is None:
                    display = ready_overlay
                    ready = True
                else:
                    display = draw_waiting(
                        render_disc(frame, cx, cy, r_out),
                        status,
                        _model_input_panel(mode, None, input_label),
                    )
                    ready = False
            else:
                strip = polar.unwrap(frame, cx, cy, r_in, r_out)
                angle, confidence, probs = predict_probs(model, strip)
                disc = render_disc(frame, cx, cy, r_out)
                display = draw_overlay(disc, angle, confidence, probs, strip, input_label)
                ready = True

            cv2.imshow("minimap angle", display)
            if args.snapshot is not None and ready:
                cv2.imwrite(str(args.snapshot), display)
                print(f"snapshot saved: {args.snapshot}")
                break
            remaining_ms = round((interval - (time.monotonic() - started)) * 1000)
            key = cv2.waitKey(max(1, remaining_ms)) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        if worker is not None:
            worker.stop()
        work_dir.cleanup()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
