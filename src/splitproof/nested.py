"""Nested group-aware cross-validation without train/validation contamination."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .constraints import ConstraintError, validate_records
from .hashing import stable_digest
from .kfold import assign_kfold
from .models import Assignment, Record

if TYPE_CHECKING:
    from .repeat import StabilityReport


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


@dataclass(frozen=True, slots=True)
class RepeatedNestedSplit:
    """Independent nested splits with deterministic repeat-level seeds."""

    repetitions: tuple[NestedSplit, ...]

    @property
    def repeats(self) -> int:
        """Number of nested repetitions."""
        return len(self.repetitions)

    @property
    def outer_folds(self) -> int:
        """Number of outer folds shared by every repetition."""
        return self.repetitions[0].outer_folds if self.repetitions else 0

    @property
    def inner_folds(self) -> int:
        """Number of inner folds shared by every repetition."""
        return self.repetitions[0].inner_folds if self.repetitions else 0

    def outer_stability(self) -> StabilityReport:
        """Return fold-agreement diagnostics for outer assignments."""
        from .repeat import stability_report

        return stability_report((split.outer for split in self.repetitions), folds=self.outer_folds)

    def inner_stability(self, outer_fold: int) -> StabilityReport:
        """Return agreement diagnostics for one outer fold's inner assignments."""
        if not self.repetitions or outer_fold < 0 or outer_fold >= self.outer_folds:
            raise IndexError(outer_fold)
        from .repeat import stability_report

        assignments = tuple(split.inner[outer_fold] for split in self.repetitions)
        common_ids = set.intersection(
            *(set(item.record_id for item in items) for items in assignments)
        )
        if not common_ids:
            raise ValueError("outer fold has no records shared across repetitions")
        return stability_report(
            (
                tuple(item for item in items if item.record_id in common_ids)
                for items in assignments
            ),
            folds=self.inner_folds,
        )


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


def repeated_nested_group_kfold(
    records: Sequence[Record],
    outer_folds: int,
    inner_folds: int,
    repeats: int,
    *,
    seed: str | int = "0",
    stratified: bool = False,
) -> RepeatedNestedSplit:
    """Create independently seeded nested group-aware cross-validation repeats."""

    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ConstraintError("repeats must be a positive integer")
    materialized = tuple(records)
    if not materialized:
        raise ConstraintError("at least one record is required")
    seed_text = str(seed)
    splits = tuple(
        nested_group_kfold(
            materialized,
            outer_folds,
            inner_folds,
            seed=stable_digest(str(repeat), seed=seed_text, domain="nested-repeat"),
            stratified=stratified,
        )
        for repeat in range(repeats)
    )
    return RepeatedNestedSplit(splits)
