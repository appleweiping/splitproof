import pytest

from splitproof import (
    Record,
    evaluate_nested,
    evaluate_nested_candidates,
    nested_group_kfold,
    repeated_nested_group_kfold,
)
from splitproof.constraints import ConstraintError


def records() -> list[Record]:
    return [Record(f"r{i}", group=f"g{i // 2}", label="a" if i % 2 else "b") for i in range(24)]


def test_nested_folds_are_disjoint_and_reproducible() -> None:
    result = nested_group_kfold(records(), 3, 2, seed="seed", stratified=True)
    same = nested_group_kfold(records(), 3, 2, seed="seed", stratified=True)
    assert result == same
    all_ids = {record.id for record in records()}
    outer_validation = set()
    for fold in range(3):
        held = set(result.outer_validation(fold))
        outer_validation |= held
        assert set(result.outer_training(fold)) == all_ids - held
        inner_ids = {item.record_id for item in result.inner[fold]}
        assert inner_ids == all_ids - held
        assert not held & inner_ids
        assert (
            set().union(*(set(result.inner_validation(fold, inner)) for inner in range(2)))
            == inner_ids
        )
    assert outer_validation == all_ids


def test_groups_never_cross_one_assignment() -> None:
    result = nested_group_kfold(records(), 3, 2)
    for assignments in (result.outer, *(result.inner.values())):
        by_group: dict[str, set[str]] = {}
        for item in assignments:
            group = next(record.group for record in records() if record.id == item.record_id)
            by_group.setdefault(group or item.record_id, set()).add(item.split)
        assert all(len(splits) == 1 for splits in by_group.values())


@pytest.mark.parametrize("outer, inner", [(True, 2), (1, 2), (2, True), (2, 1)])
def test_invalid_fold_counts(outer: int, inner: int) -> None:
    with pytest.raises(ConstraintError):
        nested_group_kfold(records(), outer, inner)


def test_repeated_nested_splits_are_stable_and_disjoint() -> None:
    result = repeated_nested_group_kfold(records(), 3, 2, 3, seed="seed", stratified=True)
    same = repeated_nested_group_kfold(records(), 3, 2, 3, seed="seed", stratified=True)
    assert result == same and result.repeats == 3
    assert result.outer_stability().repeats == 3
    assert result.inner_stability(0).repeats == 3
    all_ids = {record.id for record in records()}
    for split in result.repetitions:
        for fold in range(3):
            assert set(split.outer_validation(fold)) | set(split.outer_training(fold)) == all_ids
            assert not set(split.outer_validation(fold)) & set(split.outer_training(fold))


def test_repeated_nested_requires_positive_repeats() -> None:
    with pytest.raises(ConstraintError, match="repeats"):
        repeated_nested_group_kfold(records(), 3, 2, 0)


def test_nested_evaluation_retains_inner_selection_and_outer_scores() -> None:
    split = nested_group_kfold(records(), 3, 2, seed="eval")

    def evaluator(train: tuple[Record, ...], test: tuple[Record, ...]) -> float:
        return len(train) / len(test)

    report = evaluate_nested(records(), split, evaluator)
    assert report.complete
    assert len(report.successful) == 3
    assert all(row.selected_inner_fold in {0, 1} for row in report.successful)
    assert report.mean_score is not None
    lower = evaluate_nested(records(), split, evaluator, direction="lower")
    assert all(row.selected_inner_fold in {0, 1} for row in lower.successful)


def test_nested_evaluation_can_retain_failures() -> None:
    split = nested_group_kfold(records(), 3, 2, seed="failure")
    report = evaluate_nested(records(), split, lambda _train, _test: 1 / 0, strict=False)
    assert not report.complete and len(report.failed) == 3
    assert "ZeroDivisionError" in (report.failed[0].error or "")
    with pytest.raises(ValueError, match="nested evaluation failed"):
        evaluate_nested(records(), split, lambda _train, _test: 1 / 0)


def test_nested_candidate_selection_stays_inside_outer_training_partition() -> None:
    split = nested_group_kfold(records(), 3, 2, seed="candidates")

    def evaluator(name: str, train: tuple[Record, ...], test: tuple[Record, ...]) -> float:
        assert train and test
        return 1.0 if name == "good" else 0.25

    report = evaluate_nested_candidates(records(), split, {"bad": evaluator, "good": evaluator})
    assert report.complete
    assert report.mean_score == 1.0
    assert report.selection_counts == {"good": 3}
    assert all(row.selected_candidate == "good" for row in report.successful)


def test_nested_candidate_selection_supports_lower_direction_and_retains_failures() -> None:
    split = nested_group_kfold(records(), 3, 2, seed="candidate-failure")

    def evaluator(name: str, _train: tuple[Record, ...], _test: tuple[Record, ...]) -> float:
        if name == "broken":
            raise RuntimeError("model unavailable")
        return 0.1 if name == "fast" else 0.2

    lower = evaluate_nested_candidates(
        records(), split, {"fast": evaluator, "slow": evaluator}, direction="lower"
    )
    assert lower.selection_counts == {"fast": 3}
    failed = evaluate_nested_candidates(
        records(), split, {"broken": evaluator, "fast": evaluator}, strict=False
    )
    assert not failed.complete and len(failed.failed) == 3
    with pytest.raises(ValueError, match="nested candidate evaluation failed"):
        evaluate_nested_candidates(records(), split, {"broken": evaluator})
