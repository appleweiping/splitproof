"""Input validation and constraint definitions."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Mapping

from .models import Assignment, Record


class ConstraintError(ValueError):
    """Raised when a requested split cannot satisfy declared constraints."""


def validate_records(records: Iterable[Record]) -> tuple[Record, ...]:
    """Materialize records and reject duplicate IDs or inconsistent group weights."""
    result = tuple(records)
    if not result:
        raise ConstraintError("at least one record is required")
    counts = Counter(item.id for item in result)
    duplicates = sorted(identifier for identifier, count in counts.items() if count > 1)
    if duplicates:
        preview = ", ".join(duplicates[:5])
        raise ConstraintError(f"duplicate record ids: {preview}")
    group_weights: dict[str, set[float]] = {}
    for record in result:
        if record.group is not None and record.group_weight is not None:
            group_weights.setdefault(record.group, set()).add(record.group_weight)
    inconsistent = sorted(group for group, weights in group_weights.items() if len(weights) > 1)
    if inconsistent:
        raise ConstraintError(
            "records in a group must use one group_weight; inconsistent groups: "
            + ", ".join(inconsistent[:5])
        )
    validate_aggregate_weights(result)
    return result


def validate_aggregate_weights(records: Iterable[Record]) -> None:
    """Reject totals that overflow even though each individual weight is finite."""
    result = tuple(records)
    grouped_records: dict[str, list[Record]] = {}
    for record in result:
        group_key = f"group:{record.group}" if record.group is not None else f"record:{record.id}"
        grouped_records.setdefault(group_key, []).append(record)
    total_record_weight = sum(record.weight for record in result)
    effective_group_weights = []
    for members in grouped_records.values():
        explicit = [record.group_weight for record in members if record.group_weight is not None]
        effective_group_weights.append(
            min(explicit) if explicit else sum(r.weight for r in members)
        )
    if not math.isfinite(total_record_weight) or not math.isfinite(sum(effective_group_weights)):
        raise ConstraintError("aggregate record and group weights must remain finite")


def validate_ratios(ratios: Mapping[str, float]) -> dict[str, float]:
    """Validate a non-empty ratio mapping whose finite values sum to one."""
    if not ratios:
        raise ConstraintError("at least one split ratio is required")
    result: dict[str, float] = {}
    for name, value in ratios.items():
        if not isinstance(name, str) or not name:
            raise ConstraintError("split names must be non-empty strings")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConstraintError("split ratios must be numbers")
        result[name] = float(value)
    if any(not math.isfinite(value) or value <= 0 for value in result.values()):
        raise ConstraintError("split ratios must be finite and greater than zero")
    total = sum(result.values())
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ConstraintError(f"split ratios must sum to 1.0, got {total:.12g}")
    return dict(sorted(result.items()))


def validate_minimum_counts(
    assignments: Iterable[Assignment], minimum_counts: Mapping[str, int] | None
) -> None:
    """Ensure each named split has at least its requested record count."""
    if not minimum_counts:
        return
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in minimum_counts.values()
    ):
        raise ConstraintError("minimum counts must be non-negative integers")
    counts = Counter(item.split for item in assignments)
    failures = {
        split: (counts[split], minimum)
        for split, minimum in minimum_counts.items()
        if counts[split] < minimum
    }
    if failures:
        details = ", ".join(
            f"{name}={actual}<{minimum}" for name, (actual, minimum) in sorted(failures.items())
        )
        raise ConstraintError(f"minimum split counts are unsatisfied: {details}")


def require_labels(records: Iterable[Record]) -> None:
    """Reject missing labels for a stratified operation."""
    missing = [item.id for item in records if not item.all_labels]
    if missing:
        raise ConstraintError(
            "stratified splitting requires every record to have a label; "
            f"missing for {', '.join(missing[:5])}"
        )
