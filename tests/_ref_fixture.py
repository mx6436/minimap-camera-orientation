"""ref 测试夹具：两侧原始样本 + 最小底图资产 + locate.jsonl。

被 `test_prepare.py`（ref 数据准备）与 `test_ref_inputs.py`（ref 输入侧解析）共用。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from endfield import dataset
from placement import ref_inputs
from placement.placement import Placement
from placement.records import write_jsonl


def write_frame(path: Path, value: int = 200) -> None:
    frame = np.full((200, 200, 3), 7, dtype=np.uint8)
    frame[51:171, 49:167] = value
    assert cv2.imwrite(str(path), frame)


def placement(zone: str, x: float, y: float) -> Placement:
    return Placement(zone=zone, x=x, y=y, scale=1.0)


def locate_record(name: str, **overrides: object) -> dict:
    record = {
        "name": name,
        "status": 0,
        "message": "Global Search Success",
        "zone": "Test_Base",
        "x": 100.0,
        "y": 100.0,
        "rot": 0.0,
        "scale": 1.0,
        "locConf": 0.9,
        "isHeld": False,
        "attempts": 1,
    }
    record.update(overrides)
    return record


@dataclass
class RefFixture:
    train_raw: Path
    val_raw: Path
    assets_root: Path
    locate_path: Path
    names: dict[str, str]

    @property
    def samples(self) -> dict[str, Path]:
        return dataset.raw_samples(self.train_raw, self.val_raw)

    def resolve(self) -> ref_inputs.RefInputs:
        return ref_inputs.resolve(
            self.samples,
            locate_path=self.locate_path,
            assets_root=self.assets_root,
        )


def ref_fixture(tmp_path: Path) -> RefFixture:
    """两侧共五个样本：accepted 的 ok（val）/ ok2（train），与三类不可用记录。

    ok/held 在 val_raw，ok2/fail/noasset 在 train_raw；不可用样本用于验证从各自
    一侧剔除并计数。
    """
    train_raw, val_raw = tmp_path / "train_raw", tmp_path / "val_raw"
    assets_root = tmp_path / "assets"
    train_raw.mkdir()
    val_raw.mkdir()
    (assets_root / "Test").mkdir(parents=True)
    names = {
        "ok": "Test_Base_x100.0_y100.0_r0.0.png",
        "ok2": "Test_Base_x100.0_y100.0_r1.0.png",
        "held": "Test_Base_x100.0_y100.0_r2.0.png",
        "fail": "Test_Base_x100.0_y100.0_r3.0.png",
        "noasset": "Nowhere_Base_x100.0_y100.0_r4.0.png",
    }
    val_keys = ("ok", "held")
    for key in ("ok", "ok2", "held", "fail", "noasset"):
        write_frame((val_raw if key in val_keys else train_raw) / names[key])
    # 底图裁剪窗口（中心 (100,100) 的 118x120）内与观测一致，参考 BGR = 观测
    asset = np.full((200, 200, 4), 200, dtype=np.uint8)
    asset[..., 3] = 255
    assert cv2.imwrite(str(assets_root / "Test" / "Base.png"), asset)

    locate_path = tmp_path / "locate.jsonl"
    write_jsonl(
        locate_path,
        [
            locate_record(names["ok"]),
            locate_record(names["ok2"]),
            locate_record(names["held"], isHeld=True),
            locate_record(names["fail"], status=1, message="Global search failed."),
            locate_record(names["noasset"], zone="Nowhere_Base"),
        ],
    )
    return RefFixture(
        train_raw=train_raw,
        val_raw=val_raw,
        assets_root=assets_root,
        locate_path=locate_path,
        names=names,
    )
