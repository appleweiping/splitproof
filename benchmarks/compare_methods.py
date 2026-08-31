"""Reproducible quality benchmark for SplitProof assignment strategies."""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping

from splitproof import (
    Assignment,
    Record,
    diagnose,
    hash_split,
    stratified_group_split,
)
from splitproof.hashing import stable_digest

RATIOS = {"test": 0.2, "train": 0.6, "validation": 0.2}
SEED = "splitproof-v0.2-benchmark"


def dataset() -> tuple[Record, ...]:
    """Create a fixed weighted, multi-label corpus with indivisible groups."""
    rows: list[Record] = []
    for group in range(60):
        labels = [f"intent-{group % 5}"]
        if group % 9 == 0:
            labels.append("rare-escalation")
        if group % 4 == 0:
            labels.append("long-context")
        for index in range(1 + group % 4):
            rows.append(
                Record(
                    id=f"g{group:02d}-r{index}",
                    group=f"g{group:02d}",
                    labels=tuple(labels),
                    weight=1.0 + (group % 7) * 0.25,
                    group_weight=1.0 + (group % 6) * 0.5,
                )
            )
    return tuple(rows)


def seeded_random_group(
    records: Iterable[Record], ratios: Mapping[str, float], *, seed: str
) -> tuple[Assignment, ...]:
    """Assign whole groups with a seeded random baseline."""
    buckets: dict[str, list[Record]] = defaultdict(list)
    for record in records:
        buckets[record.group or f"record:{record.id}"].append(record)
    rng = random.Random(seed)
    intervals = sorted(ratios.items())
    result: list[Assignment] = []
    for group in sorted(buckets):
        value = rng.random()
        cumulative = 0.0
        destination = intervals[-1][0]
        for split, ratio in intervals:
            cumulative += ratio
            if value < cumulative:
                destination = split
                break
        result.extend(Assignment(record.id, destination) for record in buckets[group])
    return tuple(sorted(result, key=lambda item: item.record_id))


def legacy_greedy_v2(
    records: Iterable[Record], ratios: Mapping[str, float], *, seed: str
) -> tuple[Assignment, ...]:
    """Reproduce v2 greedy after projecting each row to its first sorted label.

    The historical algorithm had no multi-label or weight model; the projection
    and ignored weights make that limitation explicit rather than retrofitting it.
    """
    buckets: dict[str, list[Record]] = defaultdict(list)
    for record in records:
        buckets[record.group or f"record:{record.id}"].append(record)
    total_labels = Counter(record.all_labels[0] for values in buckets.values() for record in values)

    def rarity(group: str) -> float:
        labels = Counter(record.all_labels[0] for record in buckets[group])
        return max(count / total_labels[label] for label, count in labels.items())

    groups = sorted(
        buckets,
        key=lambda group: (
            -rarity(group),
            -len(buckets[group]),
            stable_digest(f"group:{group}", seed=seed, domain="group-order"),
        ),
    )
    counts = Counter({name: 0 for name in ratios})
    label_counts: dict[str, Counter[str]] = {name: Counter() for name in ratios}
    size_targets = {name: ratio * sum(map(len, buckets.values())) for name, ratio in ratios.items()}
    label_targets = {
        name: {label: ratio * count for label, count in total_labels.items()}
        for name, ratio in ratios.items()
    }
    result: list[Assignment] = []
    for group in groups:
        group_labels = Counter(record.all_labels[0] for record in buckets[group])

        def cost(
            split: str,
            *,
            group_key: str = group,
            current_labels: Counter[str] = group_labels,
        ) -> tuple[float, str]:
            before_size = abs(counts[split] - size_targets[split])
            after_size = abs(counts[split] + len(buckets[group_key]) - size_targets[split])
            overflow = max(
                0.0,
                counts[split] + len(buckets[group_key]) - size_targets[split],
            )
            label_delta = 0.0
            for label, addition in current_labels.items():
                target = label_targets[split][label]
                scale = max(1.0, target)
                before = abs(label_counts[split][label] - target) / scale
                after = abs(label_counts[split][label] + addition - target) / scale
                label_delta += after - before
            size_delta = after_size - before_size + overflow * 0.25
            score = label_delta * 2.0 + size_delta / max(1.0, sum(map(len, buckets.values())))
            tie = stable_digest(f"group:{group_key}", split, seed=seed, domain="stratified-choice")
            return score, tie

        destination = min(ratios, key=cost)
        counts[destination] += len(buckets[group])
        label_counts[destination].update(group_labels)
        result.extend(Assignment(record.id, destination) for record in buckets[group])
    return tuple(sorted(result, key=lambda item: item.record_id))


def summarize(
    records: tuple[Record, ...], assignments: tuple[Assignment, ...]
) -> dict[str, object]:
    report = diagnose(records, assignments, RATIOS)
    return {
        "group_leakage": len(report.group_leakage),
        "max_count_deviation": round(report.max_ratio_deviation, 8),
        "max_group_weight_deviation": round(report.max_group_weight_deviation, 8),
        "max_label_deviation": round(report.max_label_deviation, 8),
        "max_record_weight_deviation": round(report.max_record_weight_deviation, 8),
        "objective": round(report.objective_score, 10),
    }


def run() -> dict[str, object]:
    rows = dataset()
    methods: dict[
        str,
        Callable[[Iterable[Record]], tuple[Assignment, ...]],
    ] = {
        "seeded_random_group": lambda values: seeded_random_group(values, RATIOS, seed=SEED),
        "record_hash": lambda values: hash_split(values, RATIOS, seed=SEED),
        "legacy_greedy_v2": lambda values: legacy_greedy_v2(values, RATIOS, seed=SEED),
        "greedy_only_v3": lambda values: stratified_group_split(
            values, RATIOS, seed=SEED, max_local_iterations=0
        ),
        "optimized_v3": lambda values: stratified_group_split(values, RATIOS, seed=SEED),
    }
    results: dict[str, object] = {}
    for name, method in methods.items():
        forward = method(rows)
        backward = method(reversed(rows))
        results[name] = {
            **summarize(rows, forward),
            "order_independent": forward == backward,
        }
    return {
        "benchmark_version": "1",
        "dataset": {
            "groups": len({record.group for record in rows}),
            "labels": len({label for record in rows for label in record.all_labels}),
            "records": len(rows),
            "total_record_weight": sum(record.weight for record in rows),
        },
        "ratios": RATIOS,
        "seed": SEED,
        "methods": results,
    }


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
