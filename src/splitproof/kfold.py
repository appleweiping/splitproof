"""Deterministic group-aware k-fold assignment."""

from __future__ import annotations

from collections.abc import Iterable

from .assigners import _groups, _optimize_groups
from .constraints import ConstraintError, require_labels, validate_records
from .models import Assignment, Record


def assign_kfold(
    records: Iterable[Record],
    folds: int,
    *,
    seed: str | int = "0",
    stratified: bool = False,
    max_local_iterations: int | None = None,
) -> tuple[Assignment, ...]:
    """Assign intact groups to folds, optionally balancing label counts."""
    materialized = validate_records(records)
    if isinstance(folds, bool) or not isinstance(folds, int):
        raise ConstraintError("folds must be an integer")
    if folds < 2:
        raise ConstraintError("folds must be at least 2")
    groups = _groups(materialized)
    if folds > len(groups):
        raise ConstraintError(
            f"cannot create {folds} non-empty folds from {len(groups)} indivisible groups"
        )
    if stratified:
        require_labels(materialized)
    seed_text = str(seed)
    ratios = {f"fold-{index}": 1 / folds for index in range(folds)}
    destinations = _optimize_groups(
        groups,
        ratios,
        seed=seed_text,
        stratified=stratified,
        max_local_iterations=max_local_iterations,
    )
    result = [
        Assignment(
            record.id,
            destinations[group.key],
            int(destinations[group.key].removeprefix("fold-")),
        )
        for group in groups
        for record in group.records
    ]
    return tuple(sorted(result, key=lambda item: item.record_id))
