"""Diagnostics for split coverage, leakage, and distribution drift."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping

from .models import Assignment, Record, SplitDiagnostics


def diagnose(
    records: Iterable[Record],
    assignments: Iterable[Assignment],
    expected_ratios: Mapping[str, float] | None = None,
) -> SplitDiagnostics:
    """Compare assignments with records and report invariant violations."""
    rows = tuple(records)
    assigned = tuple(assignments)
    record_ids = {item.id for item in rows}
    assigned_ids = {item.record_id for item in assigned}
    by_id = {item.id: item for item in rows}
    counts = Counter(item.split for item in assigned if item.record_id in record_ids)
    total = sum(counts.values())
    ratios = {name: count / total if total else 0.0 for name, count in sorted(counts.items())}
    labels: dict[str, Counter[str]] = defaultdict(Counter)
    group_splits: dict[str, set[str]] = defaultdict(set)
    for item in assigned:
        record = by_id.get(item.record_id)
        if record is None:
            continue
        if record.label is not None:
            labels[item.split][record.label] += 1
        if record.group is not None:
            group_splits[record.group].add(item.split)
    leakage = tuple(sorted(group for group, splits in group_splits.items() if len(splits) > 1))
    deviation = 0.0
    if expected_ratios:
        deviation = max(
            (abs(ratios.get(name, 0.0) - expected) for name, expected in expected_ratios.items()),
            default=0.0,
        )
    return SplitDiagnostics(
        counts=dict(sorted(counts.items())),
        ratios=ratios,
        label_counts={name: dict(sorted(value.items())) for name, value in sorted(labels.items())},
        max_ratio_deviation=deviation,
        group_leakage=leakage,
        missing_ids=tuple(sorted(record_ids - assigned_ids)),
        unexpected_ids=tuple(sorted(assigned_ids - record_ids)),
    )
