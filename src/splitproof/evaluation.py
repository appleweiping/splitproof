"""Leakage-aware evaluation helpers for split assignments."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from statistics import fmean, pstdev

from .constraints import validate_records
from .models import Assignment, Record

FoldEvaluator = Callable[[tuple[Record, ...], tuple[Record, ...]], float]


@dataclass(frozen=True, slots=True)
class FoldScore:
    """One evaluated fold and its train/test cardinalities."""

    repetition: int
    fold: int
    score: float | None
    train_records: int
    test_records: int
    error: str | None = None


@dataclass(frozen=True, slots=True)
class CrossValidationReport:
    """Scores with failures retained instead of silently being dropped."""

    scores: tuple[FoldScore, ...]

    @property
    def successful(self) -> tuple[FoldScore, ...]:
        """Return finite, successfully evaluated folds."""
        return tuple(item for item in self.scores if item.error is None and item.score is not None)

    @property
    def failed(self) -> tuple[FoldScore, ...]:
        """Return folds whose evaluator raised or returned a non-finite score."""
        return tuple(item for item in self.scores if item.error is not None)

    @property
    def mean_score(self) -> float | None:
        """Return the macro mean over successful folds."""
        values = [item.score for item in self.successful if item.score is not None]
        return fmean(values) if values else None

    @property
    def score_stddev(self) -> float | None:
        """Return population standard deviation over successful folds."""
        values = [item.score for item in self.successful if item.score is not None]
        return pstdev(values) if values else None

    @property
    def complete(self) -> bool:
        """Whether every requested fold produced a score."""
        return not self.failed


def evaluate_repeated_kfold(
    records: Iterable[Record],
    assignments: Iterable[Iterable[Assignment]],
    evaluator: FoldEvaluator,
    *,
    strict: bool = True,
) -> CrossValidationReport:
    """Evaluate repeated fold assignments without permitting group leakage.

    The evaluator receives immutable tuples of train and test records. Every
    record must occur exactly once in each repetition and every fold must have a
    non-empty test set. In non-strict mode evaluator exceptions are represented
    in the report; strict mode raises immediately so CI cannot hide a failure.
    """
    if not callable(evaluator):
        raise TypeError("evaluator must be callable")
    if not isinstance(strict, bool):
        raise TypeError("strict must be a boolean")
    materialized = validate_records(records)
    by_id = {record.id: record for record in materialized}
    repetitions = tuple(tuple(items) for items in assignments)
    if not repetitions:
        raise ValueError("at least one assignment repetition is required")
    rows: list[FoldScore] = []
    for repetition_index, repetition in enumerate(repetitions):
        assignment_by_id = {item.record_id: item for item in repetition}
        if len(assignment_by_id) != len(by_id) or set(assignment_by_id) != set(by_id):
            raise ValueError("each repetition must assign every record exactly once")
        folds = tuple(sorted({item.fold for item in repetition if item.fold is not None}))
        if not folds or len(folds) < 2:
            raise ValueError("each repetition must contain at least two numbered folds")
        for fold in folds:
            test = tuple(
                record for record in materialized if assignment_by_id[record.id].fold == fold
            )
            train = tuple(
                record for record in materialized if assignment_by_id[record.id].fold != fold
            )
            if not test or not train:
                raise ValueError(f"fold {fold} must have non-empty train and test sets")
            train_groups = {record.group for record in train if record.group is not None}
            test_groups = {record.group for record in test if record.group is not None}
            if train_groups & test_groups:
                raise ValueError(
                    f"group leakage detected in repetition {repetition_index}, fold {fold}"
                )
            try:
                score = float(evaluator(train, test))
                if not math.isfinite(score):
                    raise ValueError("evaluator returned a non-finite score")
                rows.append(FoldScore(repetition_index, int(fold), score, len(train), len(test)))
            except Exception as error:
                if strict:
                    raise ValueError(
                        f"evaluator failed in repetition {repetition_index}, fold {fold}: {error}"
                    ) from error
                rows.append(
                    FoldScore(
                        repetition_index,
                        int(fold),
                        None,
                        len(train),
                        len(test),
                        f"{type(error).__name__}: {error}",
                    )
                )
    return CrossValidationReport(tuple(rows))
