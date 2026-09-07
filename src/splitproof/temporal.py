"""Leakage-aware time-interval validation, with explicit excluded records."""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from .constraints import ConstraintError, validate_records
from .models import Record


@dataclass(frozen=True)
class TimeInterval:
    """Closed interval covering *all* information used by a sample.

    Include feature lookback and label lookahead, not just the observation
    timestamp. Endpoints must be timezone-aware and are normalized to UTC.
    Equal endpoints describe a point observation.
    """

    start: datetime
    end: datetime

    def __post_init__(self) -> None:
        for field in ("start", "end"):
            value = getattr(self, field)
            if not isinstance(value, datetime) or value.utcoffset() is None:
                raise ConstraintError("time endpoints must be timezone-aware datetimes")
            try:
                object.__setattr__(self, field, value.astimezone(timezone.utc))
            except (OverflowError, ValueError) as error:
                raise ConstraintError("time endpoint outside supported UTC range") from error
        if self.end < self.start:
            raise ConstraintError("interval end must not precede start")


@dataclass(frozen=True)
class TemporalFold:
    """Disjoint, exhaustive ID sets for one validation fold.

    ``excluded`` is intentionally not training data. Each exclusion records all
    applicable reasons: interval overlap (including gaps/embargo), validation
    group membership, or future data under past-only evaluation.
    """

    index: int
    train: tuple[str, ...]
    validation: tuple[str, ...]
    excluded: tuple[tuple[str, tuple[str, ...]], ...]


def _duration(value: timedelta, name: str) -> None:
    if not isinstance(value, timedelta) or value < timedelta(0):
        raise ConstraintError(f"{name} must be a non-negative timedelta")


def _merged_intervals(
    intervals: Iterable[TimeInterval], gap: timedelta, embargo: timedelta
) -> tuple[tuple[datetime, ...], tuple[datetime, ...]]:
    try:
        expanded = sorted((item.start - gap, item.end + embargo) for item in intervals)
    except OverflowError as error:
        raise ConstraintError("gap or embargo exceeds supported datetime range") from error
    starts: list[datetime] = []
    ends: list[datetime] = []
    for start, end in expanded:
        if ends and start <= ends[-1]:
            ends[-1] = max(ends[-1], end)
        else:
            starts.append(start)
            ends.append(end)
    return tuple(starts), tuple(ends)


def purged_holdout(
    records: Iterable[Record],
    intervals: Mapping[str, TimeInterval],
    validation_ids: Iterable[str],
    *,
    gap: timedelta = timedelta(0),
    embargo: timedelta = timedelta(0),
    past_only: bool = False,
    protect_groups: bool = True,
    index: int = 0,
) -> TemporalFold:
    """Remove training candidates overlapping validation information intervals.

    Validation intervals expand by ``gap`` before and ``embargo`` after, then
    merge. Closed boundaries mean touching intervals overlap. Group protection
    additionally excludes any training candidate sharing a non-null group with
    validation. Ungrouped records are independent. Past-only mode also excludes
    samples whose information ends at or after the earliest expanded validation
    start. Empty resulting training sets raise rather than return unusable folds.

    Time is O(n log n + v log v), memory O(n + v); no pairwise overlap matrix.
    Input objects are never modified. Results are sorted by ID.
    """
    materialized = validate_records(records)
    _duration(gap, "gap")
    _duration(embargo, "embargo")
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ConstraintError("index must be a non-negative integer")
    if not isinstance(past_only, bool) or not isinstance(protect_groups, bool):
        raise ConstraintError("past_only and protect_groups must be booleans")
    known = {record.id for record in materialized}
    if set(intervals) != known:
        raise ConstraintError("interval IDs must exactly match record IDs")
    if any(not isinstance(value, TimeInterval) for value in intervals.values()):
        raise ConstraintError("interval values must be TimeInterval objects")
    validation = tuple(validation_ids)
    if not validation or len(set(validation)) != len(validation):
        raise ConstraintError("validation IDs must be nonempty and unique")
    held = set(validation)
    if not held <= known:
        raise ConstraintError("validation contains unknown IDs")
    groups = {
        record.group for record in materialized if record.id in held and record.group is not None
    }
    starts, ends = _merged_intervals((intervals[id] for id in held), gap, embargo)
    train: list[str] = []
    excluded: list[tuple[str, tuple[str, ...]]] = []
    for record in materialized:
        if record.id in held:
            continue
        interval = intervals[record.id]
        reasons: list[str] = []
        position = bisect_left(ends, interval.start)
        if position < len(starts) and starts[position] <= interval.end:
            reasons.append("interval_overlap")
        if protect_groups and record.group is not None and record.group in groups:
            reasons.append("validation_group")
        if past_only and interval.end >= starts[0]:
            reasons.append("not_past")
        if reasons:
            excluded.append((record.id, tuple(reasons)))
        else:
            train.append(record.id)
    if not train:
        raise ConstraintError("no training records remain after temporal purging")
    return TemporalFold(index, tuple(sorted(train)), tuple(sorted(held)), tuple(sorted(excluded)))


def purged_kfold(
    records: Iterable[Record],
    intervals: Mapping[str, TimeInterval],
    folds: int,
    *,
    gap: timedelta = timedelta(0),
    embargo: timedelta = timedelta(0),
    protect_groups: bool = True,
) -> tuple[TemporalFold, ...]:
    """Create chronological validation blocks, purging each training complement.

    Each ID is validated exactly once. Equal start timestamps stay in one block,
    so validation block sizes need not be equal. Blocks partition distinct start
    timestamps equally (the first remainder blocks get one extra timestamp).
    Groups can span validation folds but never cross train/validation *within*
    one fold when group protection is enabled. This is not forward chaining:
    training can contain both earlier and later non-overlapping records.
    """
    materialized = validate_records(records)
    if isinstance(folds, bool) or not isinstance(folds, int) or folds < 2:
        raise ConstraintError("folds must be an integer of at least two")
    if set(intervals) != {record.id for record in materialized}:
        raise ConstraintError("interval IDs must exactly match record IDs")
    if any(not isinstance(value, TimeInterval) for value in intervals.values()):
        raise ConstraintError("interval values must be TimeInterval objects")
    buckets: dict[datetime, list[str]] = {}
    for id, interval in intervals.items():
        buckets.setdefault(interval.start, []).append(id)
    times = sorted(buckets)
    if folds > len(times):
        raise ConstraintError("folds cannot exceed the number of distinct start timestamps")
    base, remainder = divmod(len(times), folds)
    result: list[TemporalFold] = []
    offset = 0
    for index in range(folds):
        size = base + int(index < remainder)
        ids = [id for time in times[offset : offset + size] for id in buckets[time]]
        result.append(
            purged_holdout(
                materialized,
                intervals,
                ids,
                gap=gap,
                embargo=embargo,
                protect_groups=protect_groups,
                index=index,
            )
        )
        offset += size
    return tuple(result)
