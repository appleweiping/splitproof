"""Deterministic record and group-aware split algorithms."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from .constraints import (
    require_labels,
    validate_minimum_counts,
    validate_ratios,
    validate_records,
)
from .hashing import stable_digest, stable_unit_interval
from .models import Assignment, Record

ALGORITHM_VERSION = "2"


@dataclass(frozen=True, slots=True)
class _Group:
    key: str
    records: tuple[Record, ...]
    labels: Counter[str]

    @property
    def size(self) -> int:
        return len(self.records)


def _choose_by_ratio(value: float, ratios: Mapping[str, float]) -> str:
    cumulative = 0.0
    items = list(ratios.items())
    for name, ratio in items:
        cumulative += ratio
        if value < cumulative:
            return name
    return items[-1][0]


def _groups(records: tuple[Record, ...]) -> list[_Group]:
    buckets: dict[str, list[Record]] = defaultdict(list)
    for record in records:
        key = f"group:{record.group}" if record.group is not None else f"record:{record.id}"
        buckets[key].append(record)
    return [
        _Group(
            key=key,
            records=tuple(sorted(items, key=lambda item: item.id)),
            labels=Counter(item.label for item in items if item.label is not None),
        )
        for key, items in buckets.items()
    ]


def hash_split(
    records: Iterable[Record],
    ratios: Mapping[str, float],
    *,
    seed: str | int = "0",
    minimum_counts: Mapping[str, int] | None = None,
) -> tuple[Assignment, ...]:
    """Assign records independently by a versioned stable hash.

    This algorithm is ideal when appending records must not move existing rows.
    It does not preserve groups; use :func:`balanced_group_split` when rows from
    the same source must remain together.
    """
    materialized = validate_records(records)
    checked = validate_ratios(ratios)
    assignments = tuple(
        Assignment(
            record_id=record.id,
            split=_choose_by_ratio(
                stable_unit_interval(record.id, seed=str(seed), domain="record-split"),
                checked,
            ),
        )
        for record in sorted(materialized, key=lambda item: item.id)
    )
    validate_minimum_counts(assignments, minimum_counts)
    return assignments


def _group_order(groups: list[_Group], seed: str, *, stratified: bool) -> list[_Group]:
    label_totals: Counter[str] = Counter()
    for group in groups:
        label_totals.update(group.labels)

    def rarity(group: _Group) -> float:
        if not stratified or not group.labels:
            return 0.0
        return max(count / label_totals[label] for label, count in group.labels.items())

    return sorted(
        groups,
        key=lambda group: (
            -rarity(group),
            -group.size,
            stable_digest(group.key, seed=seed, domain="group-order"),
        ),
    )


def _size_cost(
    split: str,
    group: _Group,
    counts: Mapping[str, int],
    targets: Mapping[str, float],
) -> float:
    before = abs(counts[split] - targets[split])
    after = abs(counts[split] + group.size - targets[split])
    overflow = max(0.0, counts[split] + group.size - targets[split])
    return after - before + overflow * 0.25


def balanced_group_split(
    records: Iterable[Record],
    ratios: Mapping[str, float],
    *,
    seed: str | int = "0",
    minimum_counts: Mapping[str, int] | None = None,
) -> tuple[Assignment, ...]:
    """Keep groups intact while greedily balancing record counts.

    Exact ratios are impossible when groups are indivisible. The deterministic
    greedy objective minimizes count deviation, with stable hashes breaking ties.
    """
    materialized = validate_records(records)
    checked = validate_ratios(ratios)
    seed_text = str(seed)
    groups = _group_order(_groups(materialized), seed_text, stratified=False)
    targets = {name: ratio * len(materialized) for name, ratio in checked.items()}
    counts = Counter({name: 0 for name in checked})
    result: list[Assignment] = []
    for group in groups:
        destination = min(
            checked,
            key=lambda split: (
                _size_cost(split, group, counts, targets),
                stable_digest(group.key, split, seed=seed_text, domain="group-choice"),
            ),
        )
        counts[destination] += group.size
        result.extend(Assignment(item.id, destination) for item in group.records)
    assignments = tuple(sorted(result, key=lambda item: item.record_id))
    validate_minimum_counts(assignments, minimum_counts)
    return assignments


def stratified_group_split(
    records: Iterable[Record],
    ratios: Mapping[str, float],
    *,
    seed: str | int = "0",
    minimum_counts: Mapping[str, int] | None = None,
) -> tuple[Assignment, ...]:
    """Keep groups intact while balancing both sizes and label distributions.

    The operation fails for missing labels. Because indivisible groups can make
    exact stratification impossible, output is the deterministic best greedy
    allocation rather than a promise of exact per-label counts.
    """
    materialized = validate_records(records)
    require_labels(materialized)
    checked = validate_ratios(ratios)
    seed_text = str(seed)
    groups = _group_order(_groups(materialized), seed_text, stratified=True)
    total_labels = Counter(item.label for item in materialized if item.label is not None)
    size_targets = {name: ratio * len(materialized) for name, ratio in checked.items()}
    label_targets = {
        name: {label: ratio * count for label, count in total_labels.items()}
        for name, ratio in checked.items()
    }
    sizes = Counter({name: 0 for name in checked})
    labels: dict[str, Counter[str]] = {name: Counter() for name in checked}
    result: list[Assignment] = []

    def cost(split: str, group: _Group) -> float:
        size_delta = _size_cost(split, group, sizes, size_targets)
        label_delta = 0.0
        for label, addition in group.labels.items():
            target = label_targets[split][label]
            scale = max(1.0, target)
            before = abs(labels[split][label] - target) / scale
            after = abs(labels[split][label] + addition - target) / scale
            label_delta += after - before
        return label_delta * 2.0 + size_delta / max(1.0, len(materialized))

    for group in groups:
        destination = min(
            checked,
            key=lambda split: (
                cost(split, group),
                stable_digest(group.key, split, seed=seed_text, domain="stratified-choice"),
            ),
        )
        sizes[destination] += group.size
        labels[destination].update(group.labels)
        result.extend(Assignment(item.id, destination) for item in group.records)
    assignments = tuple(sorted(result, key=lambda item: item.record_id))
    validate_minimum_counts(assignments, minimum_counts)
    return assignments
