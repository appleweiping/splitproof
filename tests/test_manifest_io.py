from __future__ import annotations

import json
from dataclasses import replace

import pytest

from splitproof import (
    Assignment,
    Record,
    balanced_group_split,
    create_manifest,
    hash_split,
    verify_manifest,
)
from splitproof.constraints import ConstraintError
from splitproof.hashing import stable_digest, stable_unit_interval
from splitproof.io import load_assignments, load_records, save_assignments
from splitproof.manifest import load_manifest, manifest_checksum, save_manifest
from splitproof.models import SplitManifest


def sample() -> list[Record]:
    return [Record(f"r{i}", group=f"g{i // 2}", label=str(i % 2)) for i in range(8)]


def test_versioned_hash_has_a_fixed_cross_machine_vector() -> None:
    assert stable_digest("abc", seed="s", domain="test") == "6fdd3c439301ea99e8c5cb02d2029208"
    assert stable_unit_interval("abc", seed="s", domain="test") == 0.43696953439485464


def test_manifest_round_trip_and_dataset_verification(tmp_path) -> None:  # type: ignore[no-untyped-def]
    rows = sample()
    ratios = {"train": 0.75, "test": 0.25}
    assigned = balanced_group_split(rows, ratios, seed=10)
    manifest = create_manifest(
        rows,
        assigned,
        algorithm="group",
        algorithm_version="1",
        seed=10,
        ratios=ratios,
    )
    path = tmp_path / "split.json"
    save_manifest(manifest, path)
    restored = load_manifest(path)
    assert restored == manifest
    assert verify_manifest(restored, reversed(rows)) == ()
    assert verify_manifest(restored, [*rows[:-1], Record("changed")]) == (
        "dataset fingerprint mismatch",
        "missing assignments for 1 records",
        "assignments contain 1 unknown records",
    )


def test_manifest_tampering_is_detected() -> None:
    rows = sample()
    assigned = balanced_group_split(rows, {"a": 0.5, "b": 0.5})
    manifest = create_manifest(rows, assigned, algorithm="group", algorithm_version="1", seed="0")
    tampered_assignment = replace(manifest.assignments[0], split="tampered")
    tampered = replace(manifest, assignments=(tampered_assignment, *manifest.assignments[1:]))
    errors = verify_manifest(tampered, rows)
    assert "manifest checksum mismatch" in errors


def test_json_and_jsonl_loading_and_assignment_output(tmp_path) -> None:  # type: ignore[no-untyped-def]
    values = [{"uid": 1, "thread": "x", "class": "yes"}, {"uid": 2}]
    json_path = tmp_path / "rows.json"
    jsonl_path = tmp_path / "rows.jsonl"
    json_path.write_text(json.dumps(values), encoding="utf-8")
    jsonl_path.write_text("\n".join(json.dumps(item) for item in values), encoding="utf-8")
    kwargs = {"id_field": "uid", "group_field": "thread", "label_field": "class"}
    assert load_records(json_path, **kwargs) == load_records(jsonl_path, **kwargs)
    out = tmp_path / "assignments.jsonl"
    save_assignments(balanced_group_split(load_records(json_path, **kwargs), {"all": 1.0}), out)
    assert len(out.read_text(encoding="utf-8").splitlines()) == 2


def test_manifest_parser_rejects_missing_fields() -> None:
    with pytest.raises(ValueError, match="missing fields"):
        SplitManifest.from_dict({})


def test_hash_manifest_allows_and_reports_expected_group_leakage() -> None:
    rows = [Record("a", group="g"), Record("b", group="g")]
    assigned = hash_split(rows, {"train": 0.5, "test": 0.5}, seed=0)
    assert len({item.split for item in assigned}) == 2
    manifest = create_manifest(
        rows,
        assigned,
        algorithm="hash",
        algorithm_version="2",
        seed=0,
        ratios={"train": 0.5, "test": 0.5},
    )
    assert verify_manifest(manifest, rows) == ()


def test_manifest_rejects_unknown_split_and_reserved_metadata() -> None:
    rows = [Record("r")]
    with pytest.raises(ConstraintError, match="absent from ratios"):
        create_manifest(
            rows,
            [Assignment("r", "bogus")],
            algorithm="group",
            algorithm_version="2",
            seed=0,
            ratios={"train": 1.0},
        )
    with pytest.raises(ConstraintError, match="reserved"):
        create_manifest(
            rows,
            [Assignment("r", "train")],
            algorithm="group",
            algorithm_version="2",
            seed=0,
            ratios={"train": 1.0},
            metadata={"hash_version": "fake"},
        )
    with pytest.raises(ConstraintError, match="keys must be strings"):
        create_manifest(
            rows,
            [Assignment("r", "train")],
            algorithm="group",
            algorithm_version="2",
            seed=0,
            ratios={"train": 1.0},
            metadata={1: "value"},  # type: ignore[dict-item]
        )


def test_verify_checks_reserved_metadata_and_external_assignments() -> None:
    rows = [Record("r")]
    manifest = create_manifest(
        rows,
        [Assignment("r", "train")],
        algorithm="group",
        algorithm_version="2",
        seed=0,
        ratios={"train": 1.0},
    )
    assert verify_manifest(manifest, rows, manifest.assignments) == ()
    errors = verify_manifest(manifest, rows, [Assignment("r", "tampered")])
    assert "external assignments do not match the manifest" in errors
    duplicate_errors = verify_manifest(
        manifest, rows, [Assignment("r", "train"), Assignment("r", "train")]
    )
    assert "external assignments contain duplicate record ids" in duplicate_errors
    changed = replace(manifest, metadata={"hash_algorithm": "other", "hash_version": "999"})
    changed = replace(changed, checksum=manifest_checksum(changed))
    errors = verify_manifest(changed, rows)
    assert "unsupported manifest hash algorithm" in errors
    assert "unsupported manifest hash version" in errors
    unknown_split = replace(manifest, assignments=(Assignment("r", "bogus"),))
    unknown_split = replace(unknown_split, checksum=manifest_checksum(unknown_split))
    assert "assignments use splits absent from ratios: bogus" in verify_manifest(
        unknown_split, rows
    )
    invalid_ratios = replace(manifest, ratios={"train": 0.5})
    invalid_ratios = replace(invalid_ratios, checksum=manifest_checksum(invalid_ratios))
    assert any(
        error.startswith("invalid manifest ratios:")
        for error in verify_manifest(invalid_ratios, rows)
    )
    assert any(
        error.startswith("invalid dataset:") for error in verify_manifest(manifest, rows * 2)
    )


def test_verify_reports_duplicate_ids_with_different_optional_fields() -> None:
    rows = [Record("r")]
    manifest = create_manifest(
        rows,
        [Assignment("r", "train")],
        algorithm="group",
        algorithm_version="2",
        seed=0,
        ratios={"train": 1.0},
    )
    invalid_rows = [Record("r"), Record("r", group="different", label="label")]
    errors = verify_manifest(manifest, invalid_rows)
    assert any(error.startswith("invalid dataset: duplicate record ids") for error in errors)
    assert "dataset fingerprint mismatch" in errors


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update(extra=True),
        lambda value: value.pop("checksum"),
        lambda value: value.update(ratios={"train": "1.0"}),
        lambda value: value.update(assignments=[{"record_id": "r", "split": "train"}]),
    ],
)
def test_manifest_loader_rejects_unknown_missing_and_malformed_fields(tmp_path, mutation) -> None:  # type: ignore[no-untyped-def]
    rows = [Record("r")]
    manifest = create_manifest(
        rows,
        [Assignment("r", "train")],
        algorithm="group",
        algorithm_version="2",
        seed=0,
        ratios={"train": 1.0},
    )
    value = manifest.to_dict()
    mutation(value)
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError):
        load_manifest(path)


@pytest.mark.parametrize(
    "text",
    [
        '{"id":NaN}\n',
        '{"id":"a","payload":1e400}\n',
        '{"id":"a","nested":{"value":-1e400}}\n',
        '{"id":"a","id":"b"}\n',
        '{"id":[]}\n',
        '{"id":"   "}\n',
        '{"id":"a","group":{}}\n',
    ],
)
def test_record_loader_rejects_non_finite_duplicate_and_unstable_fields(
    tmp_path, text: str
) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "bad.jsonl"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        load_records(path)


def test_assignment_loader_is_strict_and_round_trips(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "assignments.jsonl"
    values = (Assignment("a", "train"), Assignment("b", "test", 1))
    save_assignments(values, path)
    assert load_assignments(path) == values
    path.write_text('{"id":"a","id":"b","split":"train","fold":null}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_assignments(path)


def test_manifest_json_rejects_non_finite_and_duplicate_keys(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "manifest.json"
    path.write_text('{"schema_version":"1","schema_version":"1"}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_manifest(path)
    path.write_text('{"ratio":NaN}', encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite"):
        load_manifest(path)
