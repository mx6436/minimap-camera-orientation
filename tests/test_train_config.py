"""训练配置（input_mode）与输入模式→数据目录、运行档案写侧 adapter。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from endfield import run_record
from endfield.train.config import load_config
from endfield.train.data import (
    TRAIN_DIR,
    TRAIN_REF_DIR,
    VAL_DIR,
    VAL_REF_DIR,
    names_fingerprint,
    split_dirs,
)
from endfield.train.record import build_record


def write_config(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "train.toml"
    path.write_text(body, encoding="utf-8")
    return path


def test_config_defaults_are_polar(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path, ""))
    assert config["input_mode"] == "polar"
    assert "map_assets_root" not in config


def test_config_accepts_ref_input_mode(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path, 'input_mode = "ref"\n'))
    assert config["input_mode"] == "ref"


def test_config_rejects_unknown_input_mode(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="input_mode"):
        load_config(write_config(tmp_path, 'input_mode = "bogus"\n'))


def test_config_rejects_removed_residual_mode(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="input_mode"):
        load_config(write_config(tmp_path, 'input_mode = "residual"\n'))


def test_config_defaults_max_ref_missing_to_none(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path, 'input_mode = "ref"\n'))
    assert config["max_ref_missing"] is None


def test_config_accepts_max_ref_missing(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path, 'input_mode = "ref"\nmax_ref_missing = 0.3\n'))
    assert config["max_ref_missing"] == 0.3


def test_config_rejects_max_ref_missing_without_ref_mode(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="max_ref_missing"):
        load_config(write_config(tmp_path, "max_ref_missing = 0.3\n"))


@pytest.mark.parametrize("value", ["0.0", "1.0", "-0.1", '"0.3"', "true"])
def test_config_rejects_invalid_max_ref_missing(tmp_path: Path, value: str) -> None:
    body = f'input_mode = "ref"\nmax_ref_missing = {value}\n'
    with pytest.raises(SystemExit, match="max_ref_missing"):
        load_config(write_config(tmp_path, body))


def test_config_defaults_hard_weight(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path, ""))
    assert config["hard_weight"] == 5.0


def test_config_accepts_hard_weight(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path, "hard_weight = 2.5\n"))
    assert config["hard_weight"] == 2.5


@pytest.mark.parametrize("value", ["0", "-1.5", "true", '"5"'])
def test_config_rejects_invalid_hard_weight(tmp_path: Path, value: str) -> None:
    with pytest.raises(SystemExit, match="hard_weight"):
        load_config(write_config(tmp_path, f"hard_weight = {value}\n"))


def test_config_rejects_removed_assets_root_key(tmp_path: Path) -> None:
    """资产根不再是训练配置：ref 数据的资产根由 processed 戳携带。"""
    with pytest.raises(SystemExit, match="unknown config keys"):
        load_config(write_config(tmp_path, 'map_assets_root = "/tmp/assets"\n'))


def test_config_still_rejects_unknown_keys(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="unknown config keys"):
        load_config(write_config(tmp_path, "not_a_key = 1\n"))


def test_split_dirs_follows_input_mode() -> None:
    assert split_dirs("polar") == (TRAIN_DIR, VAL_DIR)
    assert split_dirs("ref") == (TRAIN_REF_DIR, VAL_REF_DIR)
    with pytest.raises(ValueError, match="unknown input_mode"):
        split_dirs("bogus")


def build(
    config: dict,
    tmp_path: Path,
    assets_root: str | None = None,
    hard_names: tuple[str, ...] = (),
) -> dict:
    """跑通写侧 adapter 并落盘，返回 record.json 的 payload。"""
    record = build_record(
        config,
        threads=8,
        device=torch.device("cpu"),
        train_count=1,
        val_count=1,
        train_sha256="a",
        val_sha256="b",
        trainable_parameters=131169,
        assets_root=assets_root,
        hard_names=hard_names,
    )
    run_dir = tmp_path / "run"
    run_record.write(run_dir, record)
    return json.loads((run_dir / "record.json").read_text(encoding="utf-8"))


def test_record_declares_polar_representation_by_default(tmp_path: Path) -> None:
    record = build(load_config(write_config(tmp_path, "")), tmp_path)
    assert record["version"] == run_record.SCHEMA_VERSION
    assert record["trainable_parameters"] == 131169
    assert record["input_mode"] == "polar"
    assert record["input_representation"].startswith("polar_unwrap")
    assert record["input_shape"] == [3, 42, 360]
    assert "ref_reference_assets_root" not in record
    assert "max_ref_missing" not in record


def test_record_declares_ref_representation_and_assets_root(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path, 'input_mode = "ref"\n'))
    record = build(config, tmp_path, assets_root="/tmp/assets")
    assert record["input_mode"] == "ref"
    assert record["input_representation"].startswith("ref_polar_unwrap")
    assert "255*(1 - a/255)" in record["input_representation"]
    assert record["input_shape"] == [7, 42, 360]
    assert record["ref_reference_assets_root"] == "/tmp/assets"
    assert "pair_encoding" not in record
    assert "pair_max_ref_missing" not in record


def test_record_rejects_ref_without_assets_root(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path, 'input_mode = "ref"\n'))
    with pytest.raises(SystemExit, match="assets root"):
        build(config, tmp_path)


def test_record_declares_max_ref_missing_and_filter_text(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path, 'input_mode = "ref"\nmax_ref_missing = 0.3\n'))
    record = build(config, tmp_path, assets_root="/tmp/assets")
    assert record["max_ref_missing"] == 0.3
    assert "0.3" in record["input_representation"]


def test_record_max_ref_missing_defaults_to_null(tmp_path: Path) -> None:
    record = build(
        load_config(write_config(tmp_path, 'input_mode = "ref"\n')),
        tmp_path,
        assets_root="/tmp/assets",
    )
    assert record["max_ref_missing"] is None


def test_record_declares_hard_weight_and_names(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path, ""))
    record = build(config, tmp_path, hard_names=("b_r90.png", "a_r0.png"))
    assert record["hard_weight"] == 5.0
    assert record["hard_count"] == 2
    assert record["hard_names_sha256"] == names_fingerprint(["a_r0.png", "b_r90.png"])
    assert "sample-weighted" in record["loss"] and "unweighted" in record["loss"]


def test_record_hard_count_and_digest_follow_the_hard_set(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path, ""))
    baseline = build(config, tmp_path)
    one = build(config, tmp_path, hard_names=("a_r0.png",))
    other = build(config, tmp_path, hard_names=("b_r90.png",))

    assert baseline["hard_count"] == 0
    assert one["hard_count"] == 1
    digests = {
        baseline["hard_names_sha256"],
        one["hard_names_sha256"],
        other["hard_names_sha256"],
    }
    assert len(digests) == 3


def test_record_hard_weight_follows_config(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path, "hard_weight = 2.5\n"))
    record = build(config, tmp_path, hard_names=("a_r0.png",))
    assert record["hard_weight"] == 2.5
