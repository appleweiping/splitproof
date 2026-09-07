"""Nested group-aware cross-validation without train/validation contamination."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .constraints import ConstraintError, validate_records
from .kfold import assign_kfold
from .models import Assignment, Record


@dataclass(frozen=True, slots=True)
class NestedSplit:
    """Outer assignments and one inner assignment per outer training fold."""

    outer: tuple[Assignment, ...]
    inner: Mapping[int, tuple[Assignment, ...]]
    outer_folds: int
    inner_folds: int

    def outer_validation(self, fold: int) -> tuple[str, ...]:
        """IDs held out by one outer fold."""
        if fold < 0 or fold >= self.outer_folds:
            raise IndexError(fold)
        return tuple(item.record_id for item in self.outer if item.fold == fold)

    def outer_training(self, fold: int) -> tuple[str, ...]:
        """IDs available for model selection in one outer fold."""
        held = set(self.outer_validation(fold))
        return tuple(item.record_id for item in self.outer if item.record_id not in held)

    def inner_validation(self, outer_fold: int, inner_fold: int) -> tuple[str, ...]:
        """IDs held out by an inner fold within an outer training set."""
        if outer_fold not in self.inner:
            raise IndexError(outer_fold)
        if inner_fold < 0 or inner_fold >= self.inner_folds:
            raise IndexError(inner_fold)
        return tuple(item.record_id for item in self.inner[outer_fold] if item.fold == inner_fold)


def nested_group_kfold(
    records: Sequence[Record],
    outer_folds: int,
    inner_folds: int,
    *,
    seed: str | int = "0",
    stratified: bool = False,
) -> NestedSplit:
    """Create inner folds only from each outer training partition."""
    materialized = validate_records(records)
    if isinstance(outer_folds, bool) or not isinstance(outer_folds, int) or outer_folds < 2:
        raise ConstraintError("outer_folds must be an integer of at least two")
    if isinstance(inner_folds, bool) or not isinstance(inner_folds, int) or inner_folds < 2:
        raise ConstraintError("inner_folds must be an integer of at least two")
    outer = assign_kfold(materialized, outer_folds, seed=f"{seed}:outer", stratified=stratified)
    inner_by_outer: dict[int, tuple[Assignment, ...]] = {}
    for fold in range(outer_folds):
        held = {item.record_id for item in outer if item.fold == fold}
        training = tuple(record for record in materialized if record.id not in held)
        inner_by_outer[fold] = assign_kfold(
            training, inner_folds, seed=f"{seed}:inner:{fold}", stratified=stratified
        )
    return NestedSplit(tuple(outer), inner_by_outer, outer_folds, inner_folds)
