"""根目录 live.py 主循环的回归测试：用假 MaaFw 控制器跑通一帧 polar 主循环并落 snapshot。"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import cv2
import numpy as np
import pytest

import live
from endfield.live import RunConfig
from endfield.polar import IMG_H, IMG_W

FRAME = np.zeros((720, 1280, 3), dtype=np.uint8)


class _Future:
    def __init__(self, value: object) -> None:
        self._value = value

    def get(self) -> object:
        return self._value

    def wait(self) -> object:
        return self._value


class _FakeController:
    def post_connection(self) -> _Future:
        result = types.SimpleNamespace(status=types.SimpleNamespace(succeeded=True))
        return _Future(result)

    def post_screencap(self) -> _Future:
        return _Future(FRAME)


def _install_fake_maa(monkeypatch: pytest.MonkeyPatch) -> None:
    """注入最小 MaaFw 面：LinuxController + Toolkit.find_gamescope_instances。"""

    class _LinuxController(_FakeController):
        def __init__(self, config: dict) -> None:
            self.config = config

    class _Toolkit:
        @staticmethod
        def find_gamescope_instances() -> list:
            return [
                types.SimpleNamespace(
                    display_no=1, pipewire_node_id=7, eis_socket_path="/tmp/eis-fake"
                )
            ]

    maa = types.ModuleType("maa")
    controller = types.ModuleType("maa.controller")
    toolkit = types.ModuleType("maa.toolkit")
    controller.LinuxController = _LinuxController  # type: ignore[attr-defined]
    toolkit.Toolkit = _Toolkit  # type: ignore[attr-defined]
    maa.controller = controller  # type: ignore[attr-defined]
    maa.toolkit = toolkit  # type: ignore[attr-defined]
    for name, module in (
        ("maa", maa),
        ("maa.controller", controller),
        ("maa.toolkit", toolkit),
    ):
        monkeypatch.setitem(sys.modules, name, module)


def test_polar_main_composes_overlay(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_fake_maa(monkeypatch)
    monkeypatch.setattr(live, "load_run_config", lambda run_dir: RunConfig("polar", None))
    monkeypatch.setattr(live, "load_model", lambda path, device: object())
    monkeypatch.setattr(live, "choose_device", lambda device: "cpu")
    monkeypatch.setattr(
        live,
        "predict_probs",
        lambda model, strip: (123.0, 0.9, np.ones(360, dtype=np.float32) / 360.0),
    )
    monkeypatch.setattr(live.cv2, "imshow", lambda *args, **kwargs: None)
    monkeypatch.setattr(live.cv2, "destroyAllWindows", lambda *args, **kwargs: None)
    snapshot = tmp_path / "overlay.png"
    monkeypatch.setattr(
        sys, "argv", ["live.py", "--run-dir", "runs/fake", "--snapshot", str(snapshot)]
    )

    live.main()

    overlay = cv2.imread(str(snapshot))
    assert overlay is not None, "snapshot was not written"
    # 圆盘/输入栏/概率曲线三栏纵向堆叠，输入栏为 2x 条带宽
    assert overlay.shape[1] >= IMG_W * 2
    assert overlay.shape[0] > IMG_H * 2
