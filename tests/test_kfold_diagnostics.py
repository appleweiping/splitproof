from __future__ import annotations

from collections import defaultdict

import pytest

from splitproof import Assignment, Record, assign_kfold, diagnose
from splitproof.constraints import ConstraintError


def test_kfold_is_complete_balanced_and_group_safe() -> None:
    rows = [
        Record(f"r{group}-{index}", group=f"g{group}", label=str(group % 2))
        for group in range(9)
        for index in range(2)
    ]
    result = assign_kfold(rows, 3, seed="cv", stratified=True)
    assert {item.fold for item in result} == {0, 1, 2}
    groups: dict[str, set[int | None]] = defaultdict(set)
    by_id = {item.id: item for item in rows}
    for item in result:
        groups[by_id[item.record_id].group or ""].add(item.fold)
    assert all(len(folds) == 1 for folds in groups.values())
    assert result == assign_kfold(reversed(rows), 3, seed="cv", stratified=True)


def test_kfold_rejects_impossible_fold_count() -> None:
    rows = [Record("a", group="one"), Record("b", group="one")]
    with pytest.raises(ConstraintError, match="indivisible groups"):
        assign_kfold(rows, 2)
    with pytest.raises(ConstraintError, match="at least 2"):
        assign_kfold(rows, 1)


def test_diagnostics_detects_leakage_and_coverage() -> None:
    rows = [Record("a", group="g"), Record("b", group="g"), Record("c")]
    assignments = [
        Assignment("a", "train"),
        Assignment("b", "test"),
        Assignment("unknown", "test"),
    ]
    report = diagnose(rows, assignments, {"train": 0.5, "test": 0.5})
    assert report.group_leakage == ("g",)
    assert report.missing_ids == ("c",)
    assert report.unexpected_ids == ("unknown",)
    assert not report.valid
