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
from .scoring import balance_components

ALGORITHM_VERSION = "3"


@dataclass(frozen=True, slots=True)
class _Group:
    key: str
    records: tuple[Record, ...]
    labels: Counter[str]
    label_weights: dict[str, float]
    record_weight: float
    allocation_weight: float

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
    result: list[_Group] = []
    for key in sorted(buckets):
        items = buckets[key]
        ordered = tuple(sorted(items, key=lambda item: item.id))
        labels: Counter[str] = Counter()
        label_weights: dict[str, float] = defaultdict(float)
        for item in ordered:
            labels.update(item.all_labels)
            for label in item.all_labels:
                label_weights[label] += item.weight
        record_weight = sum(item.weight for item in ordered)
        explicit_group_weights = {
            item.group_weight for item in ordered if item.group_weight is not None
        }
        allocation_weight = min(explicit_group_weights) if explicit_group_weights else record_weight
        result.append(
            _Group(
                key=key,
                records=ordered,
                labels=labels,
                label_weights=dict(label_weights),
                record_weight=record_weight,
                allocation_weight=allocation_weight,
            )
        )
    return result


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
    label_totals: dict[str, float] = defaultdict(float)
    for group in groups:
        for label, weight in group.label_weights.items():
            label_totals[label] += weight

    def rarity(group: _Group) -> float:
        if not stratified or not group.labels:
            return 0.0
        return max(weight / label_totals[label] for label, weight in group.label_weights.items())

    return sorted(
        groups,
        key=lambda group: (
            -rarity(group),
            -group.allocation_weight,
            -group.record_weight,
            -group.size,
            stable_digest(group.key, seed=seed, domain="group-order"),
        ),
    )


def _assignment_score(
    groups: list[_Group],
    destinations: Mapping[str, str],
    ratios: Mapping[str, float],
    *,
    stratified: bool,
) -> float:
    counts: Counter[str] = Counter()
    record_weights: dict[str, float] = defaultdict(float)
    group_weights: dict[str, float] = defaultdict(float)
    label_weights: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    label_totals: dict[str, float] = defaultdict(float)
    for group in groups:
        for label, weight in group.label_weights.items():
            label_totals[label] += weight
        destination = destinations.get(group.key)
        if destination is None:
            continue
        counts[destination] += group.size
        record_weights[destination] += group.record_weight
        group_weights[destination] += group.allocation_weight
        for label, weight in group.label_weights.items():
            label_weights[label][destination] += weight
    components = balance_components(
        ratios=ratios,
        counts=counts,
        total_count=float(sum(group.size for group in groups)),
        record_weights=record_weights,
        total_record_weight=sum(group.record_weight for group in groups),
        group_weights=group_weights,
        total_group_weight=sum(group.allocation_weight for group in groups),
        label_weights=label_weights,
        label_totals=label_totals,
        include_labels=stratified,
    )
    return components["total"]


def _optimize_groups(
    groups: list[_Group],
    ratios: Mapping[str, float],
    *,
    seed: str,
    stratified: bool,
    max_local_iterations: int | None,
) -> dict[str, str]:
    """Greedily place groups, then apply deterministic improving moves.

    Every accepted move strictly reduces the shared balance objective. Search
    terminates at the first one-move local optimum or at the configured finite
    iteration limit. For at most 32 groups, deterministic pair swaps are also
    considered. Stable BLAKE2 digests resolve equal-score candidates.
    """
    if max_local_iterations is not None and (
        isinstance(max_local_iterations, bool)
        or not isinstance(max_local_iterations, int)
        or max_local_iterations < 0
    ):
        raise ValueError("max_local_iterations must be a non-negative integer or null")
    ordered = _group_order(groups, seed, stratified=stratified)
    splits = tuple(ratios)
    destinations: dict[str, str] = {}
    for index, group in enumerate(ordered):
        empty = tuple(split for split in splits if split not in destinations.values())
        remaining = len(ordered) - index
        candidates = empty if empty and remaining == len(empty) else splits
        choices: list[tuple[float, str, str]] = []
        for split in candidates:
            proposed = {**destinations, group.key: split}
            score = _assignment_score(groups, proposed, ratios, stratified=stratified)
            tie = stable_digest(group.key, split, seed=seed, domain="greedy-choice-v3")
            choices.append((score, tie, split))
        destinations[group.key] = min(choices)[2]

    if max_local_iterations is None:
        iteration_limit = min(200, len(groups) * 2)
    else:
        iteration_limit = max_local_iterations
    ordered_keys = tuple(sorted(destinations))
    preserve_nonempty = len(groups) >= len(splits)
    for _ in range(iteration_limit):
        current_score = _assignment_score(groups, destinations, ratios, stratified=stratified)
        source_counts = Counter(destinations.values())
        best_key: tuple[float, str] | None = None
        best_action: tuple[str, str, str] | None = None

        for group_key in ordered_keys:
            source = destinations[group_key]
            if preserve_nonempty and source_counts[source] <= 1:
                continue
            for destination in splits:
                if destination == source:
                    continue
                proposed = {**destinations, group_key: destination}
                score = _assignment_score(groups, proposed, ratios, stratified=stratified)
                if score >= current_score - 1e-15:
                    continue
                tie = stable_digest(
                    group_key, source, destination, seed=seed, domain="local-move-v3"
                )
                key = (score, tie)
                if best_key is None or key < best_key:
                    best_key = key
                    best_action = ("move", group_key, destination)

        if len(groups) <= 32:
            for left_index, left in enumerate(ordered_keys):
                for right in ordered_keys[left_index + 1 :]:
                    if destinations[left] == destinations[right]:
                        continue
                    proposed = dict(destinations)
                    proposed[left], proposed[right] = proposed[right], proposed[left]
                    score = _assignment_score(groups, proposed, ratios, stratified=stratified)
                    if score >= current_score - 1e-15:
                        continue
                    tie = stable_digest(left, right, seed=seed, domain="local-swap-v3")
                    key = (score, tie)
                    if best_key is None or key < best_key:
                        best_key = key
                        best_action = ("swap", left, right)

        if best_action is None:
            break
        action, left, right = best_action
        if action == "move":
            destinations[left] = right
        else:
            destinations[left], destinations[right] = (
                destinations[right],
                destinations[left],
            )
    return destinations


def balanced_group_split(
    records: Iterable[Record],
    ratios: Mapping[str, float],
    *,
    seed: str | int = "0",
    minimum_counts: Mapping[str, int] | None = None,
    max_local_iterations: int | None = None,
) -> tuple[Assignment, ...]:
    """Keep groups intact while balancing count and both weight types.

    Exact ratios are impossible when groups are indivisible. The deterministic
    greedy phase is followed by bounded strictly improving local search.
    """
    materialized = validate_records(records)
    checked = validate_ratios(ratios)
    seed_text = str(seed)
    groups = _groups(materialized)
    destinations = _optimize_groups(
        groups,
        checked,
        seed=seed_text,
        stratified=False,
        max_local_iterations=max_local_iterations,
    )
    result = [
        Assignment(item.id, destinations[group.key]) for group in groups for item in group.records
    ]
    assignments = tuple(sorted(result, key=lambda item: item.record_id))
    validate_minimum_counts(assignments, minimum_counts)
    return assignments


def stratified_group_split(
    records: Iterable[Record],
    ratios: Mapping[str, float],
    *,
    seed: str | int = "0",
    minimum_counts: Mapping[str, int] | None = None,
    max_local_iterations: int | None = None,
) -> tuple[Assignment, ...]:
    """Keep groups intact while balancing counts, weights, and label sets.

    The operation fails for missing labels. Because indivisible groups can make
    exact stratification impossible, output is a deterministic greedy-plus-local
    best effort rather than a promise of exact per-label proportions.
    """
    materialized = validate_records(records)
    require_labels(materialized)
    checked = validate_ratios(ratios)
    seed_text = str(seed)
    groups = _groups(materialized)
    destinations = _optimize_groups(
        groups,
        checked,
        seed=seed_text,
        stratified=True,
        max_local_iterations=max_local_iterations,
    )
    result = [
        Assignment(item.id, destinations[group.key]) for group in groups for item in group.records
    ]
    assignments = tuple(sorted(result, key=lambda item: item.record_id))
    validate_minimum_counts(assignments, minimum_counts)
    return assignments
