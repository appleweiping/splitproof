from __future__ import annotations

from collections import Counter, defaultdict

import pytest

from splitproof import (
    Record,
    balanced_group_split,
    hash_split,
    stratified_group_split,
)
from splitproof.constraints import ConstraintError
from splitproof.hashing import stable_unit_interval


def records() -> list[Record]:
    return [
        Record(
            f"r{group}-{index}",
            group=f"g{group}",
            label="positive" if group % 2 else "negative",
        )
        for group in range(12)
        for index in range(2)
    ]


@pytest.mark.parametrize("assigner", [hash_split, balanced_group_split, stratified_group_split])
def test_deterministic_and_input_order_independent(assigner) -> None:  # type: ignore[no-untyped-def]
    rows = records()
    ratios = {"train": 0.6, "validation": 0.2, "test": 0.2}
    forward = assigner(rows, ratios, seed="paper-7")
    backward = assigner(reversed(rows), ratios, seed="paper-7")
    assert forward == backward


@pytest.mark.parametrize("assigner", [balanced_group_split, stratified_group_split])
def test_group_aware_algorithms_never_leak(assigner) -> None:  # type: ignore[no-untyped-def]
    rows = records()
    assignments = assigner(rows, {"train": 0.75, "test": 0.25}, seed=4)
    split_by_id = {item.record_id: item.split for item in assignments}
    observed: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        observed[row.group or ""].add(split_by_id[row.id])
    assert all(len(splits) == 1 for splits in observed.values())


def test_hash_split_is_append_stable() -> None:
    rows = records()
    ratios = {"train": 0.7, "test": 0.3}
    before = hash_split(rows, ratios, seed="fixed")
    after = hash_split([*rows, Record("new")], ratios, seed="fixed")
    assert before == tuple(item for item in after if item.record_id != "new")


def test_stratified_group_split_balances_labels() -> None:
    rows = [
        Record(f"{label}-{group}-{index}", group=f"{label}-{group}", label=label)
        for label in ("a", "b")
        for group in range(10)
        for index in range(2)
    ]
    assignments = stratified_group_split(rows, {"train": 0.5, "test": 0.5}, seed=8)
    by_id = {row.id: row for row in rows}
    label_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for item in assignments:
        label_counts[item.split][by_id[item.record_id].label or ""] += 1
    assert abs(label_counts["train"]["a"] - label_counts["test"]["a"]) <= 2
    assert abs(label_counts["train"]["b"] - label_counts["test"]["b"]) <= 2


def test_tiny_dataset_has_defined_best_effort_behavior() -> None:
    assignment = balanced_group_split([Record("only")], {"train": 0.8, "test": 0.2})
    assert len(assignment) == 1
    with pytest.raises(ConstraintError, match="minimum split counts"):
        balanced_group_split(
            [Record("only")],
            {"train": 0.8, "test": 0.2},
            minimum_counts={"train": 1, "test": 1},
        )


def test_missing_labels_are_rejected_for_stratification() -> None:
    with pytest.raises(ConstraintError, match="requires every record"):
        stratified_group_split([Record("a")], {"train": 0.8, "test": 0.2})


def test_duplicate_ids_and_invalid_ratios_are_rejected() -> None:
    with pytest.raises(ConstraintError, match="duplicate"):
        hash_split([Record("same"), Record("same")], {"train": 1.0})
    with pytest.raises(ConstraintError, match=r"sum to 1\.0"):
        hash_split([Record("a")], {"train": 0.8, "test": 0.3})
    with pytest.raises(ConstraintError, match="greater than zero"):
        hash_split([Record("a")], {"train": 1.0, "test": 0.0})


def test_different_seed_can_change_assignments() -> None:
    rows = [Record(str(index)) for index in range(100)]
    ratios = {"train": 0.5, "test": 0.5}
    assert hash_split(rows, ratios, seed="one") != hash_split(rows, ratios, seed="two")


def test_ratio_mapping_order_never_changes_assignments() -> None:
    rows = records()
    forward = hash_split(rows, {"train": 0.7, "test": 0.3}, seed="fixed")
    backward = hash_split(rows, {"test": 0.3, "train": 0.7}, seed="fixed")
    assert forward == backward


def test_unit_interval_cannot_round_up_to_one(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr("splitproof.hashing.stable_digest", lambda *args, **kwargs: "f" * 32)
    value = stable_unit_interval("record", seed="seed", domain="record-split")
    assert value == ((1 << 53) - 1) / (1 << 53)
    assert 0 <= value < 1


def test_direct_records_require_stable_string_fields() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        Record("")
    with pytest.raises(ValueError, match="group"):
        Record("id", group=[])  # type: ignore[arg-type]
