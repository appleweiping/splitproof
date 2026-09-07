import pytest

from splitproof import (
    Record,
    holdout_stability_report,
    repeated_group_holdout,
    repeated_kfold,
    stability_report,
)


def records() -> list[Record]:
    return [
        Record(str(index), group=f"g{index // 2}", label="a" if index % 2 else "b")
        for index in range(8)
    ]


def test_repeated_kfold_is_deterministic_and_group_safe() -> None:
    first = repeated_kfold(records(), 2, 3, seed="seed", stratified=True)
    second = repeated_kfold(records(), 2, 3, seed="seed", stratified=True)
    assert first == second
    for repetition in first:
        by_group: dict[str, set[str]] = {}
        for assignment in repetition:
            group = next(item.group for item in records() if item.id == assignment.record_id)
            by_group.setdefault(group or "", set()).add(assignment.split)
        assert all(len(splits) == 1 for splits in by_group.values())


def test_stability_report_entropy_and_validation() -> None:
    repetitions = repeated_kfold(records(), 2, 2, seed=1)
    report = stability_report(repetitions)
    assert report.repeats == 2 and report.folds == 2
    assert 0 <= report.pairwise_agreement <= 1
    assert set(report.fold_counts) == {str(index) for index in range(8)}
    with pytest.raises(ValueError, match="positive"):
        repeated_kfold(records(), 2, 0)
    with pytest.raises(ValueError, match="repetition"):
        stability_report([])


def test_repeated_group_holdout_reports_named_allocation_rates() -> None:
    repetitions = repeated_group_holdout(
        records(), {"train": 0.75, "test": 0.25}, 4, seed="holdout"
    )
    report = holdout_stability_report(repetitions)
    assert report.repeats == 4 and report.splits == ("test", "train")
    assert report.rate("0", "test") + report.rate("0", "train") == pytest.approx(1.0)
    assert all(len(repetition) == len(records()) for repetition in repetitions)
    with pytest.raises(ValueError, match="positive"):
        repeated_group_holdout(records(), {"train": 1, "test": 1}, 0)
