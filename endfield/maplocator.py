"""本地 MapLocator 工作台：布局推导与按需探测。

布局、CLI 契约、重建步骤与资产来源见 docs/maplocator-workspace.md。
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Mapping
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT / "local" / "maplocator"
DOC_PATH = "docs/maplocator-workspace.md"
# processed 戳的 provenance 键：这批 ref 数据由哪个资产根裁出
PROVENANCE_KEY = "assets_root"

CLI_REL = Path("bin") / "map-locate"
RESOURCE_REL = Path("resource")
ASSETS_REL = RESOURCE_REL / "image" / "MapLocator"
ZMDMAP_REL = Path("data") / "ZmdMap"

_PROBE_TIMEOUT_S = 10


def cli_path(root: Path = WORKSPACE_ROOT) -> Path:
    return root / CLI_REL


def resource_dir(root: Path = WORKSPACE_ROOT) -> Path:
    return root / RESOURCE_REL


def assets_root(root: Path = WORKSPACE_ROOT) -> Path:
    return root / ASSETS_REL


def zmdmap_root(root: Path = WORKSPACE_ROOT) -> Path:
    return root / ZMDMAP_REL


def provenance(assets: Path) -> dict[str, str]:
    """processed 戳的 provenance：资产根是快照内容的一部分，换根即失效重生成。"""
    return {PROVENANCE_KEY: str(assets)}


def assets_root_from_provenance(stamp: Mapping[str, object] | None) -> str:
    """从 processed 戳取资产根；缺失即来源不可知，拒绝带病继续。"""
    recorded = stamp.get("provenance") if isinstance(stamp, Mapping) else None
    value = recorded.get(PROVENANCE_KEY) if isinstance(recorded, Mapping) else None
    if isinstance(value, str) and value:
        return value
    raise SystemExit(
        f"processed 数据缺少资产根溯源（{PROVENANCE_KEY}）："
        f"重跑 prepare_data.py --mode ref（见 {DOC_PATH}）"
    )


def require_locator(root: Path = WORKSPACE_ROOT, *, stream: bool = False) -> tuple[Path, Path]:
    """CLI 与资源目录；缺件、起不来或不支持 --stream 时硬报错。"""
    cli = _require_cli(cli_path(root))
    resource = _require_resource(resource_dir(root))
    usage = _usage(cli)
    if usage is None:
        raise SystemExit(f"{cli} 无法执行（缺共享库？见 {DOC_PATH}）")
    if stream and "--stream" not in usage:
        raise SystemExit(f"{cli} 不支持 --stream，需按 {DOC_PATH} 重建工作台")
    return cli, resource


def require_assets(assets: Path) -> Path:
    if not assets.is_dir():
        raise SystemExit(f"参考底图目录不存在: {assets}（见 {DOC_PATH}）")
    return assets


def require_zmdmap(zmdmap: Path) -> Path:
    if not zmdmap.is_dir():
        raise SystemExit(f"ZmdMap 数据目录不存在: {zmdmap}（见 {DOC_PATH}）")
    return zmdmap


def _require_cli(cli: Path) -> Path:
    if not cli.is_file() or not os.access(cli, os.X_OK):
        raise SystemExit(f"定位 CLI 不存在或不可执行: {cli}（见 {DOC_PATH}）")
    return cli


def _require_resource(resource: Path) -> Path:
    if not resource.is_dir():
        raise SystemExit(f"资源目录不存在: {resource}（见 {DOC_PATH}）")
    return resource


def _usage(cli: Path) -> str | None:
    """CLI 没有 --help：无参数运行会把 usage（含 --stream）打到 stderr 并以 2 退出。

    返回 stdout+stderr；起不来（缺共享库等）返回 None。
    """
    try:
        result = subprocess.run(
            [str(cli)], capture_output=True, text=True, timeout=_PROBE_TIMEOUT_S, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stderr + result.stdout
