from __future__ import annotations

import json
from pathlib import Path

import pytest

from splitproof import Assignment, Record, compare_manifests, create_manifest, save_manifest
from splitproof.cli import main


def manifest(assignments: tuple[Assignment, ...]):
    records = tuple(Record(item.record_id) for item in assignments)
    return create_manifest(
        records,
        assignments,
        algorithm="hash",
        algorithm_version="1",
        seed="s",
        ratios={"train": 0.5, "test": 0.5},
    )


def test_manifest_comparison_reports_moves_and_transitions() -> None:
    before = manifest((Assignment("a", "train"), Assignment("b", "test")))
    after = manifest((Assignment("a", "test"), Assignment("b", "test")))
    report = compare_manifests(before, after)
    assert report.same_data
    assert report.same_algorithm
    assert report.moved == (("a", "train", None, "test", None),)
    assert report.transitions == {"train->test": 1}
    assert report.changed
    assert report.to_dict()["moved"][0]["record_id"] == "a"


def test_manifest_comparison_reports_added_removed_and_type_errors() -> None:
    before = manifest((Assignment("a", "train"),))
    after = manifest((Assignment("a", "train"), Assignment("b", "test")))
    report = compare_manifests(before, after)
    assert report.added == ("b",) and report.removed == ()
    with pytest.raises(TypeError, match="SplitManifest"):
        compare_manifests(object(), after)  # type: ignore[arg-type]


def test_compare_cli_returns_drift_and_writes_json(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    before = tmp_path / "before.json"
    after = tmp_path / "after.json"
    save_manifest(manifest((Assignment("a", "train"), Assignment("b", "test"))), before)
    save_manifest(manifest((Assignment("a", "test"), Assignment("b", "test"))), after)
    output = tmp_path / "comparison.json"
    assert main(["compare", str(before), str(after), "--output", str(output)]) == 1
    assert json.loads(output.read_text(encoding="utf-8"))["changed"] is True
    same = tmp_path / "same.json"
    save_manifest(manifest((Assignment("a", "train"), Assignment("b", "test"))), same)
    assert main(["compare", str(before), str(same)]) == 0
    assert json.loads(capsys.readouterr().out)["changed"] is False
