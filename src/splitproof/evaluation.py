"""Leakage-aware evaluation helpers for split assignments."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from statistics import fmean, pstdev
from typing import TYPE_CHECKING, Literal

from .constraints import validate_records
from .models import Assignment, Record

if TYPE_CHECKING:
    from .nested import NestedSplit, RepeatedNestedSplit

FoldEvaluator = Callable[[tuple[Record, ...], tuple[Record, ...]], float]
CandidateEvaluator = Callable[[str, tuple[Record, ...], tuple[Record, ...]], float]


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


@dataclass(frozen=True, slots=True)
class NestedFoldScore:
    """One outer fold with retained inner-selection evidence."""

    repetition: int
    outer_fold: int
    inner_scores: tuple[tuple[int, float], ...]
    selected_inner_fold: int | None
    outer_score: float | None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class NestedEvaluationReport:
    """Nested evaluation results that never hide inner or outer failures."""

    scores: tuple[NestedFoldScore, ...]
    direction: Literal["higher", "lower"]

    @property
    def successful(self) -> tuple[NestedFoldScore, ...]:
        """Return complete outer folds with a finite selected score."""

        return tuple(
            item for item in self.scores if item.error is None and item.outer_score is not None
        )

    @property
    def failed(self) -> tuple[NestedFoldScore, ...]:
        """Return outer folds with an inner or outer evaluation error."""

        return tuple(item for item in self.scores if item.error is not None)

    @property
    def mean_score(self) -> float | None:
        """Return the macro mean over successful outer folds."""

        values = [item.outer_score for item in self.successful if item.outer_score is not None]
        return fmean(values) if values else None

    @property
    def complete(self) -> bool:
        """Whether every outer fold completed selection and evaluation."""

        return not self.failed


@dataclass(frozen=True, slots=True)
class CandidateNestedFoldScore:
    """One outer fold with candidate means and the selected model name."""

    repetition: int
    outer_fold: int
    inner_scores: tuple[tuple[str, float], ...]
    selected_candidate: str | None
    outer_score: float | None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class CandidateNestedEvaluationReport:
    """Leakage-aware model-selection scores with candidate failures retained."""

    scores: tuple[CandidateNestedFoldScore, ...]
    direction: Literal["higher", "lower"]

    @property
    def successful(self) -> tuple[CandidateNestedFoldScore, ...]:
        """Return complete outer folds with a selected finite score."""

        return tuple(
            item for item in self.scores if item.error is None and item.outer_score is not None
        )

    @property
    def failed(self) -> tuple[CandidateNestedFoldScore, ...]:
        """Return outer folds that failed candidate selection or evaluation."""

        return tuple(item for item in self.scores if item.error is not None)

    @property
    def mean_score(self) -> float | None:
        """Return the macro mean over selected outer scores."""

        values = [item.outer_score for item in self.successful if item.outer_score is not None]
        return fmean(values) if values else None

    @property
    def selection_counts(self) -> Mapping[str, int]:
        """Return deterministic counts of selected candidates."""

        selected = (
            item.selected_candidate
            for item in self.successful
            if item.selected_candidate is not None
        )
        return dict(sorted(Counter(selected).items()))

    @property
    def complete(self) -> bool:
        """Whether every outer fold completed selection and scoring."""

        return not self.failed


def evaluate_nested(
    records: Iterable[Record],
    nested: NestedSplit | RepeatedNestedSplit,
    evaluator: FoldEvaluator,
    *,
    direction: Literal["higher", "lower"] = "higher",
    strict: bool = True,
) -> NestedEvaluationReport:
    """Evaluate inner selection and outer holdouts from a nested split.

    The evaluator receives immutable tuples. The best inner score (or lowest
    when ``direction='lower'``) is retained as selection evidence; the outer
    score is computed only on records excluded from that outer training set.
    In non-strict mode an inner or outer failure becomes a row-level error.
    """

    if not callable(evaluator):
        raise TypeError("evaluator must be callable")
    if direction not in {"higher", "lower"}:
        raise ValueError("direction must be 'higher' or 'lower'")
    if not isinstance(strict, bool):
        raise TypeError("strict must be a boolean")
    materialized = validate_records(records)
    by_id = {record.id: record for record in materialized}
    if not by_id:
        raise ValueError("at least one record is required")
    from .nested import NestedSplit, RepeatedNestedSplit

    repetitions: tuple[NestedSplit, ...]
    if isinstance(nested, NestedSplit):
        repetitions = (nested,)
    elif isinstance(nested, RepeatedNestedSplit):
        repetitions = nested.repetitions
    else:
        raise TypeError("nested must be NestedSplit or RepeatedNestedSplit")
    if not repetitions:
        raise ValueError("nested split must contain at least one repetition")

    rows: list[NestedFoldScore] = []
    for repetition_index, split in enumerate(repetitions):
        outer_by_id = {item.record_id: item for item in split.outer}
        if set(outer_by_id) != set(by_id) or len(outer_by_id) != len(by_id):
            raise ValueError("outer assignments must cover every record exactly once")
        for outer_fold in range(split.outer_folds):
            outer_test = tuple(
                by_id[item.record_id] for item in split.outer if item.fold == outer_fold
            )
            outer_train = tuple(
                by_id[item.record_id] for item in split.outer if item.fold != outer_fold
            )
            inner = split.inner.get(outer_fold)
            if inner is None:
                raise ValueError(f"missing inner assignments for outer fold {outer_fold}")
            inner_by_id = {item.record_id: item for item in inner}
            if set(inner_by_id) != {record.id for record in outer_train}:
                raise ValueError("inner assignments must cover the outer training records")
            inner_scores: list[tuple[int, float]] = []
            error: str | None = None
            try:
                for inner_fold in range(split.inner_folds):
                    inner_validation = tuple(
                        by_id[item.record_id] for item in inner if item.fold == inner_fold
                    )
                    inner_training = tuple(
                        record
                        for record in outer_train
                        if inner_by_id[record.id].fold != inner_fold
                    )
                    if not inner_validation or not inner_training:
                        raise ValueError(f"inner fold {inner_fold} must have non-empty partitions")
                    score = float(evaluator(inner_training, inner_validation))
                    if not math.isfinite(score):
                        raise ValueError("evaluator returned a non-finite score")
                    inner_scores.append((inner_fold, score))
                selected = (
                    max(inner_scores, key=lambda item: (item[1], -item[0]))
                    if direction == "higher"
                    else min(inner_scores, key=lambda item: (item[1], item[0]))
                )
                outer_score = float(evaluator(outer_train, outer_test))
                if not math.isfinite(outer_score):
                    raise ValueError("evaluator returned a non-finite outer score")
                rows.append(
                    NestedFoldScore(
                        repetition_index,
                        outer_fold,
                        tuple(inner_scores),
                        selected[0],
                        outer_score,
                    )
                )
            except Exception as exc:
                if strict:
                    raise ValueError(
                        "nested evaluation failed in repetition "
                        f"{repetition_index}, outer fold {outer_fold}: {exc}"
                    ) from exc
                error = f"{type(exc).__name__}: {exc}"
                rows.append(
                    NestedFoldScore(
                        repetition_index,
                        outer_fold,
                        tuple(inner_scores),
                        None,
                        None,
                        error,
                    )
                )
    return NestedEvaluationReport(tuple(rows), direction)


def evaluate_nested_candidates(
    records: Iterable[Record],
    nested: NestedSplit | RepeatedNestedSplit,
    candidates: Mapping[str, CandidateEvaluator],
    *,
    direction: Literal["higher", "lower"] = "higher",
    strict: bool = True,
) -> CandidateNestedEvaluationReport:
    """Select and score named candidates inside each nested outer fold.

    Each callback receives the candidate name plus immutable train/test tuples.
    Candidate means are computed only on inner folds of the outer training
    partition. The best candidate is then evaluated once on the untouched outer
    holdout. Ties are resolved by candidate name, and non-strict failures remain
    attached to their outer fold instead of becoming a misleading score.
    """

    if not isinstance(candidates, Mapping) or not candidates:
        raise ValueError("candidates must be a non-empty mapping")
    if any(not isinstance(name, str) or not name.strip() for name in candidates):
        raise ValueError("candidate names must be non-empty strings")
    if not all(callable(evaluator) for evaluator in candidates.values()):
        raise TypeError("candidate evaluators must be callable")
    if direction not in {"higher", "lower"}:
        raise ValueError("direction must be 'higher' or 'lower'")
    if not isinstance(strict, bool):
        raise TypeError("strict must be a boolean")
    materialized = validate_records(records)
    by_id = {record.id: record for record in materialized}
    if not by_id:
        raise ValueError("at least one record is required")
    from .nested import NestedSplit, RepeatedNestedSplit

    repetitions: tuple[NestedSplit, ...]
    if isinstance(nested, NestedSplit):
        repetitions = (nested,)
    elif isinstance(nested, RepeatedNestedSplit):
        repetitions = nested.repetitions
    else:
        raise TypeError("nested must be NestedSplit or RepeatedNestedSplit")
    if not repetitions:
        raise ValueError("nested split must contain at least one repetition")

    rows: list[CandidateNestedFoldScore] = []
    names = tuple(sorted(candidates))
    for repetition_index, split in enumerate(repetitions):
        outer_by_id = {item.record_id: item for item in split.outer}
        if set(outer_by_id) != set(by_id) or len(outer_by_id) != len(by_id):
            raise ValueError("outer assignments must cover every record exactly once")
        for outer_fold in range(split.outer_folds):
            outer_test = tuple(
                by_id[item.record_id] for item in split.outer if item.fold == outer_fold
            )
            outer_train = tuple(
                by_id[item.record_id] for item in split.outer if item.fold != outer_fold
            )
            inner = split.inner.get(outer_fold)
            if inner is None:
                raise ValueError(f"missing inner assignments for outer fold {outer_fold}")
            inner_by_id = {item.record_id: item for item in inner}
            if set(inner_by_id) != {record.id for record in outer_train}:
                raise ValueError("inner assignments must cover the outer training records")
            inner_scores: list[tuple[str, float]] = []
            try:
                for name in names:
                    values: list[float] = []
                    evaluator = candidates[name]
                    for inner_fold in range(split.inner_folds):
                        inner_validation = tuple(
                            by_id[item.record_id] for item in inner if item.fold == inner_fold
                        )
                        inner_training = tuple(
                            record
                            for record in outer_train
                            if inner_by_id[record.id].fold != inner_fold
                        )
                        if not inner_validation or not inner_training:
                            raise ValueError(
                                f"inner fold {inner_fold} must have non-empty partitions"
                            )
                        score = float(evaluator(name, inner_training, inner_validation))
                        if not math.isfinite(score):
                            raise ValueError("candidate evaluator returned a non-finite score")
                        values.append(score)
                    inner_scores.append((name, fmean(values)))
                selected = min(
                    inner_scores,
                    key=lambda item: ((-item[1] if direction == "higher" else item[1]), item[0]),
                )
                outer_score = float(candidates[selected[0]](selected[0], outer_train, outer_test))
                if not math.isfinite(outer_score):
                    raise ValueError("candidate evaluator returned a non-finite outer score")
                rows.append(
                    CandidateNestedFoldScore(
                        repetition_index, outer_fold, tuple(inner_scores), selected[0], outer_score
                    )
                )
            except Exception as exc:
                if strict:
                    raise ValueError(
                        "nested candidate evaluation failed in repetition "
                        f"{repetition_index}, outer fold {outer_fold}: {exc}"
                    ) from exc
                rows.append(
                    CandidateNestedFoldScore(
                        repetition_index,
                        outer_fold,
                        tuple(inner_scores),
                        None,
                        None,
                        f"{type(exc).__name__}: {exc}",
                    )
                )
    return CandidateNestedEvaluationReport(tuple(rows), direction)


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
