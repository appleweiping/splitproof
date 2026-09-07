"""Repeated cross-validation and split stability diagnostics."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from .hashing import stable_digest
from .kfold import assign_kfold
from .models import Assignment, Record


@dataclass(frozen=True, slots=True)
class StabilityReport:
    """Agreement and uncertainty summaries across repeated assignments."""

    repeats: int
    folds: int
    pairwise_agreement: float
    per_record_entropy: Mapping[str, float]
    fold_counts: Mapping[str, Mapping[int, int]]

    @property
    def mean_entropy(self) -> float:
        """Mean normalized assignment entropy across records."""
        return (
            sum(self.per_record_entropy.values()) / len(self.per_record_entropy)
            if self.per_record_entropy
            else 0.0
        )


def repeated_kfold(
    records: Iterable[Record],
    folds: int,
    repeats: int,
    *,
    seed: str | int = "0",
    stratified: bool = False,
    max_local_iterations: int | None = None,
) -> tuple[tuple[Assignment, ...], ...]:
    """Generate independently seeded, group-safe repeated k-fold assignments.

    Each repetition uses a domain-separated digest seed, so changing the number
    of repeats cannot perturb the first repetition. Input records are validated
    once by ``assign_kfold`` and returned in stable record-ID order.
    """
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    materialized = tuple(records)
    if not materialized:
        raise ValueError("at least one record is required")
    seed_text = str(seed)
    return tuple(
        assign_kfold(
            materialized,
            folds,
            seed=stable_digest(str(repeat), seed=seed_text, domain="repeat-kfold"),
            stratified=stratified,
            max_local_iterations=max_local_iterations,
        )
        for repeat in range(repeats)
    )


def stability_report(
    assignments: Iterable[Iterable[Assignment]],
    *,
    folds: int | None = None,
) -> StabilityReport:
    """Summarize repeated assignments without requiring the original records."""
    repetitions = tuple(tuple(items) for items in assignments)
    if not repetitions:
        raise ValueError("at least one assignment repetition is required")
    ids = tuple(sorted({item.record_id for repetition in repetitions for item in repetition}))
    if not ids:
        raise ValueError("assignments must contain records")
    if any(len({item.record_id for item in repetition}) != len(ids) for repetition in repetitions):
        raise ValueError("each repetition must contain every record exactly once")
    if any(item.fold is None for repetition in repetitions for item in repetition):
        raise ValueError("stability_report requires fold assignments")
    inferred = sorted(
        {item.fold for repetition in repetitions for item in repetition if item.fold is not None}
    )
    fold_count = folds if folds is not None else (len(inferred) if inferred else 1)
    if isinstance(fold_count, bool) or not isinstance(fold_count, int) or fold_count < 1:
        raise ValueError("folds must be a positive integer")
    by_repeat = [
        {item.record_id: int(item.fold) for item in repetition if item.fold is not None}
        for repetition in repetitions
    ]
    agreement_values: list[float] = []
    for left_index in range(len(by_repeat)):
        for right_index in range(left_index + 1, len(by_repeat)):
            agreement_values.append(
                sum(
                    by_repeat[left_index][identifier] == by_repeat[right_index][identifier]
                    for identifier in ids
                )
                / len(ids)
            )
    fold_counts: dict[str, dict[int, int]] = {}
    entropy: dict[str, float] = {}
    for identifier in ids:
        counts = Counter(mapping[identifier] for mapping in by_repeat)
        fold_counts[identifier] = dict(sorted(counts.items(), key=lambda item: str(item[0])))
        probabilities = [count / len(by_repeat) for count in counts.values()]
        entropy[identifier] = (
            -sum(
                probability * math.log(probability, fold_count)
                for probability in probabilities
                if probability
            )
            if fold_count > 1
            else 0.0
        )
    return StabilityReport(
        len(repetitions),
        fold_count,
        sum(agreement_values) / len(agreement_values) if agreement_values else 1.0,
        MappingProxyType(entropy),
        MappingProxyType({key: MappingProxyType(value) for key, value in fold_counts.items()}),
    )
