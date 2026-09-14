"""定位记录的测试构造：全字段成功记录与失败记录（字段契约见 docs/maplocator-workspace.md）。"""

from __future__ import annotations

OK = {
    "name": "Wuling_Base_x1000.0_y1403.0_r346.1.png",
    "status": 0,
    "message": "Global Search Success",
    "zone": "Wuling_Base",
    "x": 1.0,
    "y": 2.0,
    "rot": 3.0,
    "scale": 1.0,
    "locConf": 0.9,
    "isHeld": False,
    "latencyMs": 10,
    "attempts": 1,
    "elapsedMs": 20,
}


def failed(name: str, status: int, message: str) -> dict:
    """失败记录：字段仍齐全，值取占位（zone 为空、坐标与分数归零）。"""
    return {
        **OK,
        "name": name,
        "status": status,
        "message": message,
        "zone": "",
        "locConf": 0.0,
        "attempts": 3,
    }
