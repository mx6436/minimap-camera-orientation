"""运行档案契约：输入模式词汇、record 读写与 checkpoint 通道核对。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from endfield.run_record import (
    ASSETS_ROOT_FIELD,
    RECORD_NAME,
    REPO_ROOT,
    SCHEMA_VERSION,
    InputMode,
    RunRecord,
    input_channels,
    read,
    validate_channels,
    write,
)


def write_raw(run_dir: Path, payload: dict) -> None:
    (run_dir / RECORD_NAME).write_text(json.dumps(payload), encoding="utf-8")


def polar_payload(**overrides: object) -> dict:
    payload: dict = {
        "version": 27,
        "target_sigma": 3.0,
        "trainable_parameters": 131169,
        "input_representation": "polar_unwrap_bgr_360x42",
    }
    payload.update(overrides)
    return payload


def test_input_channels_accepts_names_and_members() -> None:
    assert input_channels("polar") == 3
    assert input_channels(InputMode.REF) == 7
    with pytest.raises(ValueError, match="unknown input_mode"):
        input_channels("bogus")


def test_parse_rejects_non_string_values() -> None:
    with pytest.raises(ValueError, match="input_mode"):
        InputMode.parse(3)


def test_read_defaults_missing_input_mode_to_polar(tmp_path: Path) -> None:
    write_raw(tmp_path, polar_payload())
    record = read(tmp_path)
    assert record.input_mode is InputMode.POLAR
    assert record.assets_root is None
    assert record.target_sigma == 3.0
    assert record.trainable_parameters == 131169
    assert record.metadata["input_representation"].startswith("polar_unwrap")
    assert "version" not in record.metadata


def test_read_rejects_unknown_input_mode(tmp_path: Path) -> None:
    write_raw(tmp_path, polar_payload(input_mode="pair"))
    with pytest.raises(ValueError, match="unknown input_mode"):
        read(tmp_path)


def test_read_resolves_relative_ref_assets_root(tmp_path: Path) -> None:
    write_raw(tmp_path, polar_payload(input_mode="ref", **{ASSETS_ROOT_FIELD: "local/assets"}))
    assert read(tmp_path).assets_root == REPO_ROOT / "local" / "assets"


def test_read_keeps_absolute_ref_assets_root(tmp_path: Path) -> None:
    assets = tmp_path / "assets"
    write_raw(tmp_path, polar_payload(input_mode="ref", **{ASSETS_ROOT_FIELD: str(assets)}))
    assert read(tmp_path).assets_root == assets


@pytest.mark.parametrize("value", [None, "", 7])
def test_read_rejects_ref_without_usable_assets_root(tmp_path: Path, value: object) -> None:
    payload = polar_payload(input_mode="ref")
    if value is not None:
        payload[ASSETS_ROOT_FIELD] = value
    write_raw(tmp_path, payload)
    with pytest.raises(ValueError, match=ASSETS_ROOT_FIELD):
        read(tmp_path)


def test_read_rejects_missing_or_malformed_core_fields(tmp_path: Path) -> None:
    payload = polar_payload()
    del payload["target_sigma"]
    write_raw(tmp_path, payload)
    with pytest.raises(ValueError, match="target_sigma"):
        read(tmp_path)

    payload = polar_payload(trainable_parameters=True)
    write_raw(tmp_path, payload)
    with pytest.raises(ValueError, match="trainable_parameters"):
        read(tmp_path)


def test_read_rejects_missing_record_file(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="run record"):
        read(tmp_path)


def test_write_read_round_trip_preserves_metadata(tmp_path: Path) -> None:
    original = RunRecord(
        input_mode=InputMode.REF,
        assets_root=tmp_path / "assets",
        target_sigma=2.5,
        trainable_parameters=132321,
        metadata={"optimizer": "AdamW", "batch_size": 128},
    )
    path = write(tmp_path, original)
    assert path == tmp_path / RECORD_NAME
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["version"] == SCHEMA_VERSION
    assert payload["input_mode"] == "ref"
    assert payload[ASSETS_ROOT_FIELD] == str(tmp_path / "assets")
    assert payload["optimizer"] == "AdamW"
    assert read(tmp_path) == original


def test_write_rejects_inconsistent_assets_root(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="assets root"):
        write(tmp_path, RunRecord(InputMode.REF, None, 3.0, 1, {}))
    with pytest.raises(ValueError, match="assets root"):
        write(tmp_path, RunRecord(InputMode.POLAR, tmp_path / "assets", 3.0, 1, {}))


def test_validate_channels_matches_mode() -> None:
    polar = RunRecord(InputMode.POLAR, None, 3.0, 1, {})
    ref = RunRecord(InputMode.REF, Path("/tmp/assets"), 3.0, 1, {})
    validate_channels(polar, 3)
    validate_channels(ref, 7)
    with pytest.raises(ValueError, match="3 input channels"):
        validate_channels(ref, 3)
