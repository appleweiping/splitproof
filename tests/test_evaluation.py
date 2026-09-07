from __future__ import annotations

import pytest

from splitproof import Assignment, Record, evaluate_repeated_kfold, repeated_kfold


def _records() -> tuple[Record, ...]:
    return tuple(
        Record(str(index), group=f"g-{index}", label="x" if index % 2 else "y")
        for index in range(6)
    )


def test_evaluate_repeated_kfold_reports_scores_and_failures() -> None:
    records = _records()
    assignments = repeated_kfold(records, folds=3, repeats=2, seed="eval")

    def scorer(train, test):
        return sum(item.label == "x" for item in test) / len(test)

    report = evaluate_repeated_kfold(records, assignments, scorer)
    assert len(report.scores) == 6
    assert report.complete
    assert report.mean_score is not None

    failed = evaluate_repeated_kfold(
        records, assignments, lambda _train, _test: 1 / 0, strict=False
    )
    assert not failed.complete
    assert len(failed.failed) == 6


def test_evaluation_rejects_leaking_groups() -> None:
    records = (Record("a", group="same"), Record("b", group="same"), Record("c", group="other"))
    assignments = (
        (
            # Deliberately place one group in two folds.
            Assignment("a", "fold-0", 0),
            Assignment("b", "fold-1", 1),
            Assignment("c", "fold-1", 1),
        ),
    )
    with pytest.raises(ValueError, match="group leakage"):
        evaluate_repeated_kfold(records, assignments, lambda _train, _test: 1.0)
