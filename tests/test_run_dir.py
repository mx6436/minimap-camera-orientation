"""run 产物契约：路径常量、占用标记、训练汇总的读写与严格性。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from endfield import run_dir, run_record
from endfield.run_record import InputMode, RunRecord
from endfield.train.metrics import Metrics


def summary(**overrides: object) -> run_dir.TrainingSummary:
    values: dict[str, object] = {
        "epoch": 7,
        "val_count": 11,
        "metrics": Metrics(expected_abs_error=1.0, rms_error=2.0),
    }
    values.update(overrides)
    return run_dir.TrainingSummary(**values)  # type: ignore[arg-type]


def test_paths_are_relative_to_the_run_dir(tmp_path: Path) -> None:
    assert run_dir.checkpoint_path(tmp_path) == tmp_path / "best.pt"
    assert run_dir.record_path(tmp_path) == tmp_path / "record.json"
    assert run_dir.summary_path(tmp_path) == tmp_path / "summary.json"
    assert run_dir.history_path(tmp_path) == tmp_path / "history.json"


def test_occupied_lists_write_side_markers_only(tmp_path: Path) -> None:
    assert run_dir.occupied(tmp_path) == []
    (tmp_path / "best.pt").write_bytes(b"")
    (tmp_path / "history.json").write_text("{}", encoding="utf-8")
    # 诊断产物不进占用集合
    (tmp_path / "loss_curve.png").write_bytes(b"")
    assert run_dir.occupied(tmp_path) == ["best.pt", "history.json"]


def test_write_read_round_trip(tmp_path: Path) -> None:
    run_dir.write_summary(tmp_path, summary())
    assert run_dir.load_summary(tmp_path) == summary()


def test_write_summary_uses_val_prefixed_keys(tmp_path: Path) -> None:
    run_dir.write_summary(tmp_path, summary())
    payload = json.loads(run_dir.summary_path(tmp_path).read_text(encoding="utf-8"))

    assert payload["epoch"] == 7
    assert payload["val_count"] == 11
    assert payload["val_rms_error"] == 2.0
    assert payload["val_expected_abs_error"] == 1.0
    # 诊断指标由 Metrics 的口径补齐
    assert payload["val_entropy"] == 0.0


@pytest.mark.parametrize("key", ["epoch", "val_count", "val_rms_error", "val_expected_abs_error"])
def test_load_summary_rejects_missing_contract_fields(tmp_path: Path, key: str) -> None:
    run_dir.write_summary(tmp_path, summary())
    payload = json.loads(run_dir.summary_path(tmp_path).read_text(encoding="utf-8"))
    del payload[key]
    run_dir.summary_path(tmp_path).write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=key):
        run_dir.load_summary(tmp_path)


def test_load_summary_tolerates_missing_diagnostic_metrics(tmp_path: Path) -> None:
    run_dir.summary_path(tmp_path).write_text(
        json.dumps(
            {"epoch": 1, "val_count": 2, "val_rms_error": 2.0, "val_expected_abs_error": 1.0}
        ),
        encoding="utf-8",
    )
    loaded = run_dir.load_summary(tmp_path)

    assert loaded.metrics == Metrics(expected_abs_error=1.0, rms_error=2.0)
    assert loaded.extra == {}


def test_load_summary_keeps_unknown_keys(tmp_path: Path) -> None:
    run_dir.write_summary(tmp_path, summary(extra={"note": "keep me"}))

    assert run_dir.load_summary(tmp_path).extra == {"note": "keep me"}


def test_load_run_reads_record_and_summary(tmp_path: Path) -> None:
    run_record.write(tmp_path, RunRecord(InputMode.POLAR, None, 3.0, 0))
    run_dir.write_summary(tmp_path, summary())

    record, loaded = run_dir.load_run(tmp_path)

    assert record.input_mode is InputMode.POLAR
    assert loaded == summary()


def test_load_summary_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cannot read training summary"):
        run_dir.load_summary(tmp_path)
