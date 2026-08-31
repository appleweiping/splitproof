from __future__ import annotations

import json

import pytest

from splitproof import (
    Assignment,
    Record,
    assign_kfold,
    balanced_group_split,
    diagnose,
    stratified_group_split,
)
from splitproof.constraints import ConstraintError
from splitproof.io import load_records


def weighted_rows(groups: int = 20) -> list[Record]:
    rows: list[Record] = []
    for group in range(groups):
        labels = ("common", "rare") if group % 7 == 0 else (f"class-{group % 3}",)
        group_weight = float(1 + group % 4)
        for index in range(1 + group % 3):
            rows.append(
                Record(
                    f"r{group}-{index}",
                    group=f"g{group}",
                    labels=labels,
                    weight=1.0 + (group % 5) * 0.5,
                    group_weight=group_weight,
                )
            )
    return rows


def test_record_loader_supports_multilabel_and_both_weight_types(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "weighted.jsonl"
    path.write_text(
        json.dumps(
            {
                "uid": "a",
                "thread": "g",
                "classes": ["question", "urgent"],
                "importance": 2.5,
                "thread_weight": 7,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (record,) = load_records(
        path,
        id_field="uid",
        group_field="thread",
        label_field="classes",
        weight_field="importance",
        group_weight_field="thread_weight",
    )
    assert record.all_labels == ("question", "urgent")
    assert record.weight == 2.5
    assert record.group_weight == 7.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"weight": True},
        {"weight": 0},
        {"weight": float("inf")},
        {"group_weight": -1},
        {"labels": ("duplicate", "duplicate")},
        {"labels": ("",)},
    ],
)
def test_record_rejects_invalid_weights_and_labels(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        Record("r", **kwargs)  # type: ignore[arg-type]


def test_scalar_label_and_positional_payload_remain_compatible() -> None:
    payload = {"text": "hello"}
    record = Record("r", "g", "positive", payload)
    assert record.label == "positive"
    assert record.all_labels == ("positive",)
    assert record.payload == payload


def test_group_weight_must_be_consistent_within_a_named_group() -> None:
    rows = [
        Record("a", group="g", group_weight=1),
        Record("b", group="g", group_weight=2),
    ]
    with pytest.raises(ConstraintError, match="one group_weight"):
        balanced_group_split(rows, {"train": 1.0})


def test_aggregate_weight_overflow_is_rejected() -> None:
    rows = [Record("a", weight=1e308), Record("b", weight=1e308)]
    with pytest.raises(ConstraintError, match="aggregate"):
        balanced_group_split(rows, {"train": 1.0})


def test_empty_dataset_is_rejected() -> None:
    with pytest.raises(ConstraintError, match="at least one record"):
        balanced_group_split([], {"train": 1.0})


def test_public_ratio_and_minimum_count_types_are_strict() -> None:
    rows = [Record("r")]
    with pytest.raises(ConstraintError, match="numbers"):
        balanced_group_split(rows, {"train": True})
    with pytest.raises(ConstraintError, match="non-empty strings"):
        balanced_group_split(rows, {1: 1.0})  # type: ignore[dict-item]
    with pytest.raises(ConstraintError, match="non-negative integers"):
        balanced_group_split(rows, {"train": 1.0}, minimum_counts={"train": True})


@pytest.mark.parametrize(
    "assignment",
    [
        ("", "train", None),
        ("r", "", None),
        ("r", "train", True),
        ("r", "train", -1),
    ],
)
def test_assignment_public_boundary_is_strict(assignment: tuple[object, object, object]) -> None:
    with pytest.raises(ValueError):
        Assignment(*assignment)  # type: ignore[arg-type]


def test_weighted_multilabel_optimizer_is_deterministic_safe_and_improving() -> None:
    rows = weighted_rows()
    ratios = {"train": 0.6, "validation": 0.2, "test": 0.2}
    greedy = stratified_group_split(rows, ratios, seed="weighted", max_local_iterations=0)
    optimized = stratified_group_split(rows, ratios, seed="weighted")
    reordered = stratified_group_split(
        reversed(rows),
        {"test": 0.2, "train": 0.6, "validation": 0.2},
        seed="weighted",
    )
    assert optimized == reordered
    assert diagnose(rows, optimized, ratios).group_leakage == ()
    assert (
        diagnose(rows, optimized, ratios).objective_score
        <= diagnose(rows, greedy, ratios).objective_score
    )
    assert {item.split for item in optimized} == set(ratios)


def test_property_matrix_preserves_groups_and_order_independence() -> None:
    ratios = {"train": 0.7, "test": 0.3}
    for groups in (2, 3, 8, 17):
        rows = weighted_rows(groups)
        for seed in ("0", "paper", "unicode-种子"):
            forward = stratified_group_split(rows, ratios, seed=seed)
            backward = stratified_group_split(reversed(rows), ratios, seed=seed)
            assert forward == backward
            assert not diagnose(rows, forward, ratios).group_leakage


def test_local_iteration_boundary_and_weighted_kfold() -> None:
    rows = weighted_rows(9)
    with pytest.raises(ValueError, match="max_local_iterations"):
        balanced_group_split(rows, {"train": 1.0}, max_local_iterations=-1)
    with pytest.raises(ConstraintError, match="integer"):
        assign_kfold(rows, True)  # type: ignore[arg-type]
    folds = assign_kfold(
        rows,
        3,
        seed="folds",
        stratified=True,
        max_local_iterations=40,
    )
    assert {item.fold for item in folds} == {0, 1, 2}
    assert folds == assign_kfold(
        reversed(rows),
        3,
        seed="folds",
        stratified=True,
        max_local_iterations=40,
    )


def test_diagnostics_explain_count_weight_group_and_label_balance() -> None:
    rows = [
        Record("a", group="g1", labels=("x", "shared"), weight=3, group_weight=4),
        Record("b", group="g2", labels=("y", "shared"), weight=1, group_weight=2),
    ]
    assigned = balanced_group_split(rows, {"train": 0.5, "test": 0.5}, seed="d")
    report = diagnose(rows, assigned, {"train": 0.5, "test": 0.5})
    assert sum(report.record_weights.values()) == 4
    assert sum(report.group_weights.values()) == 6
    assert sum(values["shared"] for values in report.label_counts.values()) == 2
    assert set(report.label_deviations) == {"shared", "x", "y"}
    assert set(report.objective_components) == {
        "count_balance",
        "record_weight_balance",
        "group_weight_balance",
        "label_balance",
        "total",
    }
    assert report.objective_score == report.objective_components["total"]
