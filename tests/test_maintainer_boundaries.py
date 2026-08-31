from __future__ import annotations

import json
import math
import os
from dataclasses import replace
from itertools import count

import pytest

import splitproof.assigners as assigners_module
from splitproof import (
    Assignment,
    Record,
    assign_kfold,
    balanced_group_split,
    create_manifest,
    diagnose,
    stratified_group_split,
    verify_manifest,
)
from splitproof.cli import main
from splitproof.constraints import ConstraintError
from splitproof.hashing import HASH_ALGORITHM, HASH_VERSION, data_fingerprint_v1, stable_digest
from splitproof.io import load_records
from splitproof.manifest import load_manifest, manifest_checksum, save_manifest
from splitproof.models import SplitManifest


def _weighted_rows(groups: int) -> list[Record]:
    return [
        Record(
            f"记录-{group}-{index}",
            group=f"组-{group}",
            labels=("共享", f"标签-{group % 3}"),
            weight=0.5 + (group % 5),
            group_weight=1.0 + (group % 4),
        )
        for group in range(groups)
        for index in range(1 + group % 2)
    ]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"weight": 0.0},
        {"group_weight": 0.0},
        {"group_weight": True},
    ],
)
def test_zero_and_boolean_weights_are_rejected_by_the_record_boundary(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="positive finite number"):
        Record("r", **kwargs)  # type: ignore[arg-type]


def test_empty_labels_are_allowed_only_for_unstratified_operations() -> None:
    rows = [Record("a", group="g-a"), Record("b", group="g-b")]
    assert len(balanced_group_split(rows, {"all": 1.0})) == 2
    assert {item.fold for item in assign_kfold(rows, 2)} == {0, 1}
    with pytest.raises(ConstraintError, match="requires every record"):
        stratified_group_split(rows, {"all": 1.0})
    with pytest.raises(ConstraintError, match="requires every record"):
        assign_kfold(rows, 2, stratified=True)


@pytest.mark.parametrize("identifiers", [(1, "1"), (True, "true")])
def test_json_scalar_normalization_cannot_hide_duplicate_ids(tmp_path, identifiers) -> None:  # type: ignore[no-untyped-def]
    source = tmp_path / "duplicates.jsonl"
    source.write_text(
        "\n".join(json.dumps({"id": value}) for value in identifiers) + "\n",
        encoding="utf-8",
    )
    rows = load_records(source)
    assert rows[0].id == rows[1].id
    with pytest.raises(ConstraintError, match="duplicate record ids"):
        balanced_group_split(rows, {"all": 1.0})


def test_extreme_ratios_unicode_and_kfold_remain_deterministic_and_group_safe() -> None:
    rows = _weighted_rows(8)
    ratios = {"几乎全部": 1.0 - 1e-12, "极少": 1e-12}
    for assigner in (balanced_group_split, stratified_group_split):
        forward = assigner(rows, ratios, seed="种子")
        backward = assigner(reversed(rows), dict(reversed(list(ratios.items()))), seed="种子")
        assert forward == backward
        assert {item.split for item in forward} == set(ratios)
        assert diagnose(rows, forward, ratios).group_leakage == ()
    folds = assign_kfold(rows, 8, seed="种子", stratified=True)
    assert folds == assign_kfold(reversed(rows), 8, seed="种子", stratified=True)
    assert {item.fold for item in folds} == set(range(8))


def test_local_search_is_monotone_and_the_objective_is_scale_free() -> None:
    rows = _weighted_rows(10)
    ratios = {"train": 0.6, "validation": 0.2, "test": 0.2}
    scores = []
    assignments = ()
    for limit in range(5):
        assignments = stratified_group_split(rows, ratios, seed="scale", max_local_iterations=limit)
        scores.append(diagnose(rows, assignments, ratios).objective_score)
    for index in range(1, len(scores)):
        assert scores[index] <= scores[index - 1]

    scaled = [
        Record(
            row.id,
            group=row.group,
            labels=row.labels,
            weight=row.weight * 1_000_000,
            group_weight=(row.group_weight or 0.0) * 1_000_000,
        )
        for row in rows
    ]
    scaled_assignments = stratified_group_split(
        scaled, ratios, seed="scale", max_local_iterations=4
    )
    assert scaled_assignments == assignments
    assert diagnose(scaled, scaled_assignments, ratios).objective_score == pytest.approx(
        scores[-1], abs=1e-14
    )


def test_pair_swaps_are_considered_at_32_groups_but_not_33(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    original_digest = stable_digest

    def observed_domains(groups: int) -> list[str]:
        domains: list[str] = []
        calls = count(1)

        def descending_score(*args, **kwargs) -> float:  # type: ignore[no-untyped-def]
            return -float(next(calls))

        def traced_digest(*parts, seed: str = "0", domain: str) -> str:  # type: ignore[no-untyped-def]
            domains.append(domain)
            return original_digest(*parts, seed=seed, domain=domain)

        monkeypatch.setattr(assigners_module, "_assignment_score", descending_score)
        monkeypatch.setattr(assigners_module, "stable_digest", traced_digest)
        balanced_group_split(
            [Record(f"r-{index}", group=f"g-{index}") for index in range(groups)],
            {"left": 0.5, "right": 0.5},
            seed="boundary",
            max_local_iterations=1,
        )
        return domains

    assert "local-swap-v3" in observed_domains(32)
    assert "local-swap-v3" not in observed_domains(33)


@pytest.mark.parametrize(
    "ratios",
    [
        {"train": math.nan},
        {"train": math.inf},
        {"train": True},
        {"train": 0.5},
    ],
)
def test_diagnostics_reject_invalid_expected_ratios(ratios: dict[str, float]) -> None:
    with pytest.raises(ConstraintError):
        diagnose([Record("r")], [Assignment("r", "train")], ratios)


def test_diagnostics_reject_aggregate_weight_overflow() -> None:
    rows = [Record("a", weight=1e308), Record("b", weight=1e308)]
    with pytest.raises(ConstraintError, match="aggregate"):
        diagnose(rows, [Assignment("a", "all"), Assignment("b", "all")], {"all": 1.0})

    valid_rows = [Record("a"), Record("b")]
    manifest = create_manifest(
        valid_rows,
        [Assignment("a", "all"), Assignment("b", "all")],
        algorithm="group",
        algorithm_version="3",
        seed=0,
        ratios={"all": 1.0},
    )
    assert any(
        error.startswith("invalid dataset: aggregate") for error in verify_manifest(manifest, rows)
    )


def test_diagnostics_reject_duplicate_ids_and_inconsistent_group_weights() -> None:
    with pytest.raises(ConstraintError, match="duplicate record ids"):
        diagnose(
            [Record("same"), Record("same")],
            [Assignment("same", "all")],
            {"all": 1.0},
        )
    with pytest.raises(ConstraintError, match="one group_weight"):
        diagnose(
            [Record("a", group="g", group_weight=1), Record("b", group="g", group_weight=2)],
            [Assignment("a", "all"), Assignment("b", "all")],
            {"all": 1.0},
        )


def test_in_memory_manifest_corruption_returns_errors_instead_of_raising() -> None:
    rows = [Record("r")]
    manifest = create_manifest(
        rows,
        [Assignment("r", "train")],
        algorithm="group",
        algorithm_version="3",
        seed="boundary",
        ratios={"train": 1.0},
    )
    malformed_ratios = replace(manifest, ratios={"train": "not-a-number"})  # type: ignore[dict-item]
    assert any(
        error.startswith("invalid manifest ratios:")
        for error in verify_manifest(malformed_ratios, rows)
    )
    malformed_metadata = replace(manifest, metadata={**manifest.metadata, "bad": math.nan})
    assert "manifest contains values that cannot be checksummed" in verify_manifest(
        malformed_metadata, rows
    )


@pytest.mark.parametrize("ratio", [True, math.nan, math.inf, 0.0])
def test_manifest_mapping_parser_rejects_boolean_nonfinite_and_zero_ratios(ratio: object) -> None:
    rows = [Record("r")]
    manifest = create_manifest(
        rows,
        [Assignment("r", "train")],
        algorithm="group",
        algorithm_version="3",
        seed=0,
        ratios={"train": 1.0},
    )
    value = manifest.to_dict()
    value["ratios"] = {"train": ratio}
    with pytest.raises(ValueError):
        SplitManifest.from_dict(value)


def test_schema_v1_disk_round_trip_and_assignment_preserving_migration(tmp_path) -> None:  # type: ignore[no-untyped-def]
    rows = [Record("a", group="g", label="x"), Record("b", group="g", label="y")]
    assignments = (Assignment("a", "train"), Assignment("b", "train"))
    legacy = SplitManifest(
        schema_version="1",
        algorithm="group",
        algorithm_version="2",
        seed="legacy",
        created_at="2026-01-01T00:00:00+00:00",
        data_fingerprint=data_fingerprint_v1(rows),
        ratios={"train": 1.0},
        assignments=assignments,
        metadata={"hash_algorithm": HASH_ALGORITHM, "hash_version": HASH_VERSION},
    )
    legacy = replace(legacy, checksum=manifest_checksum(legacy))
    path = tmp_path / "legacy.json"
    save_manifest(legacy, path)
    restored = load_manifest(path)
    assert restored == legacy
    assert verify_manifest(restored, rows) == ()

    migrated = create_manifest(
        rows,
        restored.assignments,
        algorithm=restored.algorithm,
        algorithm_version=restored.algorithm_version,
        seed=restored.seed,
        ratios=restored.ratios,
    )
    assert migrated.schema_version == "2"
    assert migrated.assignments == restored.assignments
    assert verify_manifest(migrated, rows) == ()
    changed_weights = [replace(row, weight=2.0) for row in rows]
    assert verify_manifest(restored, changed_weights) == ()
    assert "dataset fingerprint mismatch" in verify_manifest(migrated, changed_weights)


@pytest.mark.parametrize("output_role", ["assignments", "manifest"])
def test_cli_rejects_an_output_hardlinked_to_the_input(tmp_path, capsys, output_role: str) -> None:  # type: ignore[no-untyped-def]
    source = tmp_path / "source.jsonl"
    source.write_text(
        "\n".join(f'{{"id":"r-{index}","group":"g-{index}","label":"x"}}' for index in range(3))
        + "\n",
        encoding="utf-8",
    )
    alias = tmp_path / f"{output_role}-alias.json"
    try:
        os.link(source, alias)
    except OSError as error:
        pytest.skip(f"hard links unavailable: {error}")
    original = source.read_bytes()
    assignments = alias if output_role == "assignments" else tmp_path / "assignments.jsonl"
    manifest = alias if output_role == "manifest" else tmp_path / "manifest.json"
    assert (
        main(
            [
                "split",
                str(source),
                "--algorithm",
                "group",
                "--ratios",
                "train=0.5,test=0.5",
                "--assignments",
                str(assignments),
                "--manifest",
                str(manifest),
            ]
        )
        == 2
    )
    assert source.read_bytes() == original
    assert "paths must be different" in capsys.readouterr().err


def test_cli_rejects_hardlinked_assignment_and_manifest_outputs(tmp_path, capsys) -> None:  # type: ignore[no-untyped-def]
    source = tmp_path / "source.jsonl"
    source.write_text('{"id":"r"}\n', encoding="utf-8")
    assignments = tmp_path / "assignments.jsonl"
    assignments.write_text("protected\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    try:
        os.link(assignments, manifest)
    except OSError as error:
        pytest.skip(f"hard links unavailable: {error}")
    original = assignments.read_bytes()
    assert (
        main(
            [
                "split",
                str(source),
                "--algorithm",
                "group",
                "--ratios",
                "all=1",
                "--assignments",
                str(assignments),
                "--manifest",
                str(manifest),
            ]
        )
        == 2
    )
    assert assignments.read_bytes() == original
    assert manifest.read_bytes() == original
    assert "paths must be different" in capsys.readouterr().err
