"""本地 MapLocator 工作台：布局推导、按需探测与戳内资产根溯源。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from endfield import maplocator


def make_cli(path: Path, usage: str) -> Path:
    body = f"import sys\nprint({usage!r}, file=sys.stderr)\nraise SystemExit(2)\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"#!{sys.executable}\n" + body, encoding="utf-8")
    path.chmod(0o755)
    return path


def make_workspace(
    tmp_path: Path, usage: str = "usage: map-locate --resource-dir <dir> [--stream]"
) -> Path:
    root = tmp_path / "maplocator"
    make_cli(maplocator.cli_path(root), usage)
    maplocator.resource_dir(root).mkdir(parents=True)
    return root


def test_layout_derives_leaves_from_root(tmp_path: Path) -> None:
    root = tmp_path / "maplocator"

    assert maplocator.cli_path(root) == root / "bin" / "map-locate"
    assert maplocator.resource_dir(root) == root / "resource"
    assert maplocator.assets_root(root) == root / "resource" / "image" / "MapLocator"
    assert maplocator.zmdmap_root(root) == root / "data" / "ZmdMap"


def test_provenance_round_trips_assets_root(tmp_path: Path) -> None:
    assets = tmp_path / "assets"

    assert maplocator.provenance(assets) == {"assets_root": str(assets)}
    assert maplocator.assets_root_from_provenance({"provenance": {"assets_root": str(assets)}}) == (
        str(assets)
    )


@pytest.mark.parametrize(
    "stamp",
    [
        None,
        {},
        {"provenance": None},
        {"provenance": {}},
        {"provenance": {"assets_root": ""}},
        {"provenance": {"assets_root": 3}},
    ],
)
def test_assets_root_from_provenance_rejects_unknown_source(stamp: object) -> None:
    with pytest.raises(SystemExit, match="assets_root"):
        maplocator.assets_root_from_provenance(stamp)


def test_require_locator_reports_missing_cli(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="docs/maplocator-workspace.md"):
        maplocator.require_locator(tmp_path / "maplocator")


def test_require_locator_rejects_non_executable_cli(tmp_path: Path) -> None:
    root = tmp_path / "maplocator"
    maplocator.cli_path(root).parent.mkdir(parents=True)
    maplocator.cli_path(root).write_text("#!/bin/false\n", encoding="utf-8")
    maplocator.resource_dir(root).mkdir(parents=True)

    with pytest.raises(SystemExit, match="不可执行"):
        maplocator.require_locator(root)


def test_require_locator_reports_missing_resource(tmp_path: Path) -> None:
    root = make_workspace(tmp_path)
    maplocator.resource_dir(root).rmdir()

    with pytest.raises(SystemExit, match="资源目录"):
        maplocator.require_locator(root)


def test_require_locator_reports_cli_that_cannot_start(tmp_path: Path) -> None:
    root = make_workspace(tmp_path)
    maplocator.cli_path(root).write_text("not an executable\n", encoding="utf-8")
    maplocator.cli_path(root).chmod(0o755)

    with pytest.raises(SystemExit, match="无法执行"):
        maplocator.require_locator(root)


def test_require_locator_probes_stream_capability(tmp_path: Path) -> None:
    root = make_workspace(tmp_path)

    assert maplocator.require_locator(root, stream=True) == (
        maplocator.cli_path(root),
        maplocator.resource_dir(root),
    )


def test_require_locator_rejects_cli_without_stream(tmp_path: Path) -> None:
    root = make_workspace(tmp_path, usage="usage: map-locate --resource-dir <dir>")

    with pytest.raises(SystemExit, match="--stream"):
        maplocator.require_locator(root, stream=True)


def test_require_assets_and_zmdmap_report_missing_dirs(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="参考底图"):
        maplocator.require_assets(tmp_path / "assets")
    with pytest.raises(SystemExit, match="ZmdMap"):
        maplocator.require_zmdmap(tmp_path / "ZmdMap")

    assets, zmdmap = tmp_path / "assets", tmp_path / "ZmdMap"
    assets.mkdir()
    zmdmap.mkdir()
    assert maplocator.require_assets(assets) == assets
    assert maplocator.require_zmdmap(zmdmap) == zmdmap
