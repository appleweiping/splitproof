"""Input validation and constraint definitions."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Mapping

from .models import Assignment, Record


class ConstraintError(ValueError):
    """Raised when a requested split cannot satisfy declared constraints."""


def validate_records(records: Iterable[Record]) -> tuple[Record, ...]:
    """Materialize records and reject duplicate identifiers."""
    result = tuple(records)
    counts = Counter(item.id for item in result)
    duplicates = sorted(identifier for identifier, count in counts.items() if count > 1)
    if duplicates:
        preview = ", ".join(duplicates[:5])
        raise ConstraintError(f"duplicate record ids: {preview}")
    return result


def validate_ratios(ratios: Mapping[str, float]) -> dict[str, float]:
    """Validate a non-empty ratio mapping whose finite values sum to one."""
    if not ratios:
        raise ConstraintError("at least one split ratio is required")
    result = {str(name): float(value) for name, value in ratios.items()}
    if any(not name for name in result):
        raise ConstraintError("split names must not be empty")
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
    if any(value < 0 for value in minimum_counts.values()):
        raise ConstraintError("minimum counts cannot be negative")
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
    missing = [item.id for item in records if item.label is None]
    if missing:
        raise ConstraintError(
            "stratified splitting requires every record to have a label; "
            f"missing for {', '.join(missing[:5])}"
        )
