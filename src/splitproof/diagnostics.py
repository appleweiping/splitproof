"""Diagnostics for split coverage, leakage, and distribution drift."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping

from .constraints import validate_ratios, validate_records
from .models import Assignment, Record, SplitDiagnostics
from .scoring import balance_components


def diagnose(
    records: Iterable[Record],
    assignments: Iterable[Assignment],
    expected_ratios: Mapping[str, float] | None = None,
    *,
    include_label_balance: bool = True,
) -> SplitDiagnostics:
    """Compare assignments with records and report invariant violations."""
    checked_ratios = validate_ratios(expected_ratios) if expected_ratios is not None else None
    materialized = validate_records(records)
    rows = tuple(
        sorted(
            materialized,
            key=lambda record: (
                record.id,
                record.group is None,
                record.group or "",
                record.all_labels,
                record.weight,
                record.group_weight is None,
                record.group_weight or 0.0,
            ),
        )
    )
    assigned = tuple(
        sorted(
            assignments,
            key=lambda item: (
                item.record_id,
                item.split,
                -1 if item.fold is None else item.fold,
            ),
        )
    )
    record_ids = {item.id for item in rows}
    assigned_ids = {item.record_id for item in assigned}
    by_id = {item.id: item for item in rows}
    counts = Counter(item.split for item in assigned if item.record_id in record_ids)
    total = sum(counts.values())
    ratios = {name: count / total if total else 0.0 for name, count in sorted(counts.items())}
    labels: dict[str, Counter[str]] = defaultdict(Counter)
    label_weights: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    record_weights: dict[str, float] = defaultdict(float)
    group_splits: dict[str, set[str]] = defaultdict(set)
    group_records: dict[str, list[Record]] = defaultdict(list)
    for record in rows:
        group_key = f"group:{record.group}" if record.group is not None else f"record:{record.id}"
        group_records[group_key].append(record)
    for item in assigned:
        matched = by_id.get(item.record_id)
        if matched is None:
            continue
        record_weights[item.split] += matched.weight
        for label in matched.all_labels:
            labels[item.split][label] += 1
            label_weights[item.split][label] += matched.weight
        if matched.group is not None:
            group_splits[matched.group].add(item.split)
    assigned_splits_by_id: dict[str, set[str]] = defaultdict(set)
    for item in assigned:
        if item.record_id in record_ids:
            assigned_splits_by_id[item.record_id].add(item.split)
    group_weights: dict[str, float] = defaultdict(float)
    total_group_weight = 0.0
    for group_key in sorted(group_records):
        members = group_records[group_key]
        explicit = {item.group_weight for item in members if item.group_weight is not None}
        effective_weight = min(explicit) if explicit else sum(item.weight for item in members)
        assert effective_weight is not None
        total_group_weight += effective_weight
        observed_splits = {
            split for member in members for split in assigned_splits_by_id.get(member.id, set())
        }
        if len(observed_splits) == 1:
            group_weights[next(iter(observed_splits))] += effective_weight
    leakage = tuple(sorted(group for group, splits in group_splits.items() if len(splits) > 1))
    deviation = 0.0
    record_weight_deviation = 0.0
    group_weight_deviation = 0.0
    label_deviation = 0.0
    label_deviations: dict[str, float] = {}
    total_record_weight = sum(record_weights.values())
    total_observed_group_weight = sum(group_weights.values())
    record_weight_ratios = {
        name: value / total_record_weight if total_record_weight else 0.0
        for name, value in sorted(record_weights.items())
    }
    group_weight_ratios = {
        name: value / total_observed_group_weight if total_observed_group_weight else 0.0
        for name, value in sorted(group_weights.items())
    }
    if checked_ratios:
        deviation = max(
            (abs(ratios.get(name, 0.0) - expected) for name, expected in checked_ratios.items()),
            default=0.0,
        )
        record_weight_deviation = max(
            (
                abs(record_weight_ratios.get(name, 0.0) - expected)
                for name, expected in checked_ratios.items()
            ),
            default=0.0,
        )
        group_weight_deviation = max(
            (
                abs(group_weight_ratios.get(name, 0.0) - expected)
                for name, expected in checked_ratios.items()
            ),
            default=0.0,
        )
        all_labels = sorted({label for values in label_weights.values() for label in values})
        for label in all_labels:
            label_total = sum(values.get(label, 0.0) for values in label_weights.values())
            if label_total <= 0:
                continue
            observed_deviation = max(
                abs(label_weights.get(split, {}).get(label, 0.0) / label_total - expected)
                for split, expected in checked_ratios.items()
            )
            label_deviations[label] = observed_deviation
            label_deviation = max(label_deviation, observed_deviation)
    objective_ratios = checked_ratios or ratios
    scoring_label_weights: dict[str, dict[str, float]] = defaultdict(dict)
    label_totals: dict[str, float] = defaultdict(float)
    for split, values in label_weights.items():
        for label, weight in values.items():
            scoring_label_weights[label][split] = weight
            label_totals[label] += weight
    components = balance_components(
        ratios=objective_ratios,
        counts=counts,
        total_count=float(len(rows)),
        record_weights=record_weights,
        total_record_weight=sum(record.weight for record in rows),
        group_weights=group_weights,
        total_group_weight=total_group_weight,
        label_weights=scoring_label_weights,
        label_totals=label_totals,
        include_labels=include_label_balance,
    )
    return SplitDiagnostics(
        counts=dict(sorted(counts.items())),
        ratios=ratios,
        record_weights=dict(sorted(record_weights.items())),
        record_weight_ratios=record_weight_ratios,
        group_weights=dict(sorted(group_weights.items())),
        group_weight_ratios=group_weight_ratios,
        label_counts={name: dict(sorted(value.items())) for name, value in sorted(labels.items())},
        label_weights={
            name: dict(sorted(value.items())) for name, value in sorted(label_weights.items())
        },
        label_deviations=dict(sorted(label_deviations.items())),
        max_ratio_deviation=deviation,
        max_record_weight_deviation=record_weight_deviation,
        max_group_weight_deviation=group_weight_deviation,
        max_label_deviation=label_deviation,
        objective_score=components["total"],
        objective_components=components,
        group_leakage=leakage,
        missing_ids=tuple(sorted(record_ids - assigned_ids)),
        unexpected_ids=tuple(sorted(assigned_ids - record_ids)),
    )
