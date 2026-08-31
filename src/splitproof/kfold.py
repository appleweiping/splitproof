"""Deterministic group-aware k-fold assignment."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

from .assigners import _Group, _group_order, _groups
from .constraints import ConstraintError, require_labels, validate_records
from .hashing import stable_digest
from .models import Assignment, Record


def assign_kfold(
    records: Iterable[Record],
    folds: int,
    *,
    seed: str | int = "0",
    stratified: bool = False,
) -> tuple[Assignment, ...]:
    """Assign intact groups to folds, optionally balancing label counts."""
    materialized = validate_records(records)
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
    ordered = _group_order(groups, seed_text, stratified=stratified)
    sizes = [0] * folds
    label_counts = [Counter[str]() for _ in range(folds)]
    totals = Counter(item.label for item in materialized if item.label is not None)
    result: list[Assignment] = []

    def score(fold: int, group: _Group) -> tuple[float, str]:
        size_cost = sizes[fold] + group.size
        label_cost = 0.0
        if stratified:
            for label, count in group.labels.items():
                target = totals[label] / folds
                label_cost += abs(label_counts[fold][label] + count - target) / max(1.0, target)
        tie = stable_digest(group.key, str(fold), seed=seed_text, domain="fold-choice")
        return label_cost * len(materialized) + size_cost, tie

    for group in ordered:
        fold = min(range(folds), key=lambda index: score(index, group))
        sizes[fold] += group.size
        label_counts[fold].update(group.labels)
        result.extend(Assignment(record.id, f"fold-{fold}", fold) for record in group.records)
    return tuple(sorted(result, key=lambda item: item.record_id))
