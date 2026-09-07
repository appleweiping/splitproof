import random
from datetime import datetime, timedelta, timezone

import pytest

from splitproof import Record, TimeInterval, purged_holdout, purged_kfold
from splitproof.constraints import ConstraintError

EPOCH = datetime(2026, 1, 1, tzinfo=timezone.utc)


def interval(start: int, end: int | None = None) -> TimeInterval:
    return TimeInterval(
        EPOCH + timedelta(seconds=start), EPOCH + timedelta(seconds=start if end is None else end)
    )


def test_boundaries_and_group_exclusions() -> None:
    records = [
        Record("before"),
        Record("touch"),
        Record("test", group="g"),
        Record("group", group="g"),
        Record("after"),
        Record("embargo"),
    ]
    spans = {
        "before": interval(0, 4),
        "touch": interval(4, 5),
        "test": interval(5, 10),
        "group": interval(30),
        "after": interval(20),
        "embargo": interval(11),
    }
    fold = purged_holdout(records, spans, ["test"], embargo=timedelta(seconds=1))
    assert fold.train == ("after", "before")
    assert fold.validation == ("test",)
    assert dict(fold.excluded) == {
        "touch": ("interval_overlap",),
        "group": ("validation_group",),
        "embargo": ("interval_overlap",),
    }
    past = purged_holdout(records, spans, ["test"], past_only=True)
    assert past.train == ("before",)
    assert "not_past" in dict(past.excluded)["after"]
    unprotected = purged_holdout(records, spans, ["test"], protect_groups=False)
    assert "group" in unprotected.train


def test_disjoint_validation_spans_do_not_purge_whole_envelope() -> None:
    spans = {"a": interval(0, 1), "b": interval(5, 6), "c": interval(10, 11)}
    records = [Record(id) for id in spans]
    assert purged_holdout(records, spans, ["a", "c"]).train == ("b",)
    with pytest.raises(ConstraintError, match="no training"):
        purged_holdout(records, spans, ["a", "c"], gap=timedelta(seconds=4))


def test_overlap_algorithm_matches_independent_pairwise_oracle() -> None:
    rng = random.Random(2718)
    for _ in range(30):
        spans = {}
        for i in range(40):
            start = rng.randrange(100)
            spans[str(i)] = interval(start, start + rng.randrange(10))
        spans["safe"] = interval(-100)
        records = [Record(id) for id in spans]
        held = {"0", "1", "2", "3"}
        gap, embargo = timedelta(seconds=2), timedelta(seconds=3)
        expected = {
            id
            for id, candidate in spans.items()
            if id not in held
            and not any(
                candidate.start <= spans[test].end + embargo
                and candidate.end >= spans[test].start - gap
                for test in held
            )
        }
        actual = purged_holdout(records, spans, sorted(held), gap=gap, embargo=embargo)
        assert set(actual.train) == expected
        assert set(actual.train) | set(actual.validation) | dict(actual.excluded).keys() == set(
            spans
        )
        assert actual == purged_holdout(
            list(reversed(records)), spans, sorted(held), gap=gap, embargo=embargo
        )


def test_kfold_ties_and_exhaustive_validation() -> None:
    spans = {str(i): interval(i // 2 * 10) for i in range(12)}
    records = [Record(id) for id in spans]
    folds = purged_kfold(records, spans, 3)
    assert [fold.validation for fold in folds] == [
        ("0", "1", "2", "3"),
        ("4", "5", "6", "7"),
        ("10", "11", "8", "9"),
    ]
    assert sum(len(fold.validation) for fold in folds) == len(records)
    assert all(not fold.excluded for fold in folds)
    with pytest.raises(ConstraintError, match="distinct"):
        purged_kfold(records, spans, 7)


@pytest.mark.parametrize("validation", [[], ["a", "a"], ["unknown"]])
def test_invalid_validation(validation: list[str]) -> None:
    with pytest.raises(ConstraintError):
        purged_holdout(
            [Record("a"), Record("b")], {"a": interval(0), "b": interval(10)}, validation
        )


def test_invalid_intervals() -> None:
    with pytest.raises(ConstraintError, match="timezone"):
        TimeInterval(datetime(2026, 1, 1), EPOCH)
    with pytest.raises(ConstraintError, match="precede"):
        interval(3, 2)
    shifted = EPOCH.astimezone(timezone(timedelta(hours=3)))
    assert TimeInterval(shifted, shifted) == interval(0)
    with pytest.raises(ConstraintError, match="exactly"):
        purged_holdout([Record("a")], {}, ["a"])
    with pytest.raises(ConstraintError, match="non-negative"):
        purged_holdout([Record("a")], {"a": interval(0)}, ["a"], gap=timedelta(seconds=-1))


@pytest.mark.parametrize("folds", [True, 1, -1, 1.5])
def test_invalid_folds(folds: int) -> None:
    with pytest.raises(ConstraintError, match="integer"):
        purged_kfold([Record("a")], {"a": interval(0)}, folds)


def test_embargo_overflow() -> None:
    spans = {
        "a": TimeInterval(
            datetime.max.replace(tzinfo=timezone.utc), datetime.max.replace(tzinfo=timezone.utc)
        ),
        "b": interval(0),
    }
    with pytest.raises(ConstraintError, match="range"):
        purged_holdout([Record("a"), Record("b")], spans, ["a"], embargo=timedelta(seconds=1))
