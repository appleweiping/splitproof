import pytest

from splitproof import Record, nested_group_kfold, repeated_nested_group_kfold
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
