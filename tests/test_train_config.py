"""训练配置（input_mode / map_assets_root）与输入模式→数据目录、run 档案的映射。"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from endfield.ref import MAP_ASSETS_ROOT
from endfield.train.config import load_config
from endfield.train.data import (
    TRAIN_DIR,
    TRAIN_REF_DIR,
    VAL_DIR,
    VAL_REF_DIR,
    split_dirs,
)
from endfield.train.record import build_record


def write_config(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "train.toml"
    path.write_text(body, encoding="utf-8")
    return path


def test_config_defaults_are_polar_and_local_assets(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path, ""))
    assert config["input_mode"] == "polar"
    assert config["map_assets_root"] == str(MAP_ASSETS_ROOT)


def test_config_accepts_ref_input_mode(tmp_path: Path) -> None:
    config = load_config(
        write_config(tmp_path, 'input_mode = "ref"\nmap_assets_root = "/tmp/assets"\n')
    )
    assert config["input_mode"] == "ref"
    assert config["map_assets_root"] == "/tmp/assets"


def test_config_rejects_unknown_input_mode(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="input_mode"):
        load_config(write_config(tmp_path, 'input_mode = "bogus"\n'))


def test_config_rejects_removed_residual_mode(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="input_mode"):
        load_config(write_config(tmp_path, 'input_mode = "residual"\n'))


def test_config_rejects_non_string_assets_root(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="map_assets_root"):
        load_config(write_config(tmp_path, "map_assets_root = 3\n"))


def test_config_still_rejects_unknown_keys(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="unknown config keys"):
        load_config(write_config(tmp_path, "not_a_key = 1\n"))


def test_split_dirs_follows_input_mode() -> None:
    assert split_dirs("polar") == (TRAIN_DIR, VAL_DIR)
    assert split_dirs("ref") == (TRAIN_REF_DIR, VAL_REF_DIR)
    with pytest.raises(ValueError, match="unknown input_mode"):
        split_dirs("bogus")


def build(config: dict, tmp_path: Path) -> dict:
    return build_record(
        config,
        threads=8,
        device=torch.device("cpu"),
        train_count=1,
        val_count=1,
        train_sha256="a",
        val_sha256="b",
    )


def test_record_declares_polar_representation_by_default(tmp_path: Path) -> None:
    record = build(load_config(write_config(tmp_path, "")), tmp_path)
    assert record["input_mode"] == "polar"
    assert record["input_representation"].startswith("polar_unwrap")
    assert record["input_shape"] == [3, 42, 360]
    assert "ref_reference_assets_root" not in record


def test_record_declares_ref_representation_and_assets_root(tmp_path: Path) -> None:
    config = load_config(
        write_config(tmp_path, 'input_mode = "ref"\nmap_assets_root = "/tmp/assets"\n')
    )
    record = build(config, tmp_path)
    assert record["input_mode"] == "ref"
    assert record["input_representation"].startswith("ref_polar_unwrap")
    assert "obs_roi*(1 - alpha/255)" in record["input_representation"]
    assert record["input_shape"] == [7, 42, 360]
    assert record["ref_reference_assets_root"] == "/tmp/assets"
    # ref 是唯一编码，不记历史 pair 编码/缺口过滤字段
    assert "pair_encoding" not in record
    assert "pair_max_ref_missing" not in record
