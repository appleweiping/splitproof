from __future__ import annotations

import json

import pytest

from splitproof.cli import main


def write_dataset(path) -> None:  # type: ignore[no-untyped-def]
    rows = [
        {"id": f"r{i}", "group": f"g{i // 2}", "label": "a" if i < 6 else "b"} for i in range(12)
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")


def test_split_verify_and_inspect_end_to_end(tmp_path, capsys) -> None:  # type: ignore[no-untyped-def]
    data = tmp_path / "data.jsonl"
    assignments = tmp_path / "assignments.jsonl"
    manifest = tmp_path / "manifest.json"
    report = tmp_path / "report.md"
    write_dataset(data)
    assert (
        main(
            [
                "split",
                str(data),
                "--ratios",
                "train=0.5,test=0.5",
                "--algorithm",
                "stratified-group",
                "--seed",
                "demo",
                "--assignments",
                str(assignments),
                "--manifest",
                str(manifest),
            ]
        )
        == 0
    )
    assert assignments.exists() and manifest.exists()
    assert main(["verify", str(data), "--manifest", str(manifest)]) == 0
    assert (
        main(
            [
                "inspect",
                str(data),
                "--manifest",
                str(manifest),
                "--format",
                "markdown",
                "--output",
                str(report),
            ]
        )
        == 0
    )
    assert "Status: **PASS**" in report.read_text(encoding="utf-8")
    assert "Verification passed" in capsys.readouterr().out


def test_verify_detects_changed_data(tmp_path, capsys) -> None:  # type: ignore[no-untyped-def]
    data = tmp_path / "data.jsonl"
    assignments = tmp_path / "assignments.jsonl"
    manifest = tmp_path / "manifest.json"
    write_dataset(data)
    assert (
        main(
            [
                "kfold",
                str(data),
                "--folds",
                "3",
                "--stratified",
                "--assignments",
                str(assignments),
                "--manifest",
                str(manifest),
            ]
        )
        == 0
    )
    data.write_text(data.read_text(encoding="utf-8").replace('"r0"', '"other"'), encoding="utf-8")
    assert main(["verify", str(data), "--manifest", str(manifest)]) == 1
    assert "fingerprint mismatch" in capsys.readouterr().err


def test_verify_external_assignments_and_inspect_integrity(tmp_path, capsys) -> None:  # type: ignore[no-untyped-def]
    data = tmp_path / "data.jsonl"
    assignments = tmp_path / "assignments.jsonl"
    manifest = tmp_path / "manifest.json"
    write_dataset(data)
    assert (
        main(
            [
                "split",
                str(data),
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
        == 0
    )
    assert (
        main(
            [
                "verify",
                str(data),
                "--manifest",
                str(manifest),
                "--assignments",
                str(assignments),
            ]
        )
        == 0
    )
    assignments.write_text(
        assignments.read_text(encoding="utf-8").replace(
            '"split": "train"', '"split": "changed"', 1
        ),
        encoding="utf-8",
    )
    assert (
        main(
            [
                "verify",
                str(data),
                "--manifest",
                str(manifest),
                "--assignments",
                str(assignments),
            ]
        )
        == 1
    )
    value = json.loads(manifest.read_text(encoding="utf-8"))
    value["seed"] = "tampered"
    manifest.write_text(json.dumps(value), encoding="utf-8")
    assert main(["inspect", str(data), "--manifest", str(manifest)]) == 1
    assert "checksum mismatch" in capsys.readouterr().err


def test_cli_hash_mode_allows_record_level_group_leakage(tmp_path) -> None:  # type: ignore[no-untyped-def]
    data = tmp_path / "data.jsonl"
    data.write_text('{"id":"a","group":"g"}\n{"id":"b","group":"g"}\n', encoding="utf-8")
    assert (
        main(
            [
                "split",
                str(data),
                "--algorithm",
                "hash",
                "--ratios",
                "train=0.5,test=0.5",
                "--seed",
                "0",
                "--assignments",
                str(tmp_path / "a.jsonl"),
                "--manifest",
                str(tmp_path / "m.json"),
            ]
        )
        == 0
    )


def test_cli_wraps_io_errors_and_rejects_duplicate_ratio_names(tmp_path, capsys) -> None:  # type: ignore[no-untyped-def]
    assert main(["verify", str(tmp_path / "missing"), "--manifest", str(tmp_path / "m")]) == 2
    assert "No such file" in capsys.readouterr().err
    with pytest.raises(SystemExit) as raised:
        main(
            [
                "split",
                str(tmp_path / "missing"),
                "--ratios",
                "train=0.1,train=0.9",
                "--assignments",
                str(tmp_path / "a"),
                "--manifest",
                str(tmp_path / "m"),
            ]
        )
    assert raised.value.code == 2


@pytest.mark.parametrize(
    ("assignments_role", "manifest_role"),
    [
        ("input", "manifest"),
        ("assignments", "input"),
        ("shared", "shared"),
    ],
)
def test_split_rejects_output_path_collisions_without_overwriting_input(
    tmp_path, capsys, assignments_role: str, manifest_role: str
) -> None:  # type: ignore[no-untyped-def]
    data = tmp_path / "data.jsonl"
    write_dataset(data)
    original = data.read_bytes()
    roles = {
        "input": data,
        "assignments": tmp_path / "assignments.jsonl",
        "manifest": tmp_path / "manifest.json",
        "shared": tmp_path / "shared.json",
    }

    assert (
        main(
            [
                "split",
                str(data),
                "--algorithm",
                "group",
                "--ratios",
                "train=0.5,test=0.5",
                "--assignments",
                str(roles[assignments_role]),
                "--manifest",
                str(roles[manifest_role]),
            ]
        )
        == 2
    )
    assert data.read_bytes() == original
    assert "paths must be different" in capsys.readouterr().err


def test_kfold_rejects_colliding_outputs_before_writing(tmp_path, capsys) -> None:  # type: ignore[no-untyped-def]
    data = tmp_path / "data.jsonl"
    shared = tmp_path / "shared.json"
    write_dataset(data)
    assert (
        main(
            [
                "kfold",
                str(data),
                "--folds",
                "3",
                "--assignments",
                str(shared),
                "--manifest",
                str(shared),
            ]
        )
        == 2
    )
    assert not shared.exists()
    assert "paths must be different" in capsys.readouterr().err


def test_verify_rejects_colliding_input_roles(tmp_path, capsys) -> None:  # type: ignore[no-untyped-def]
    data = tmp_path / "data.jsonl"
    write_dataset(data)
    assert (
        main(
            [
                "verify",
                str(data),
                "--manifest",
                str(data),
                "--assignments",
                str(tmp_path / "assignments.jsonl"),
            ]
        )
        == 2
    )
    assert "paths must be different" in capsys.readouterr().err


@pytest.mark.parametrize("collision", ["input", "manifest"])
def test_inspect_rejects_output_overwriting_an_input(tmp_path, capsys, collision: str) -> None:  # type: ignore[no-untyped-def]
    data = tmp_path / "data.jsonl"
    assignments = tmp_path / "assignments.jsonl"
    manifest = tmp_path / "manifest.json"
    write_dataset(data)
    assert (
        main(
            [
                "split",
                str(data),
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
        == 0
    )
    protected = data if collision == "input" else manifest
    original = protected.read_bytes()
    assert (
        main(
            [
                "inspect",
                str(data),
                "--manifest",
                str(manifest),
                "--output",
                str(protected),
            ]
        )
        == 2
    )
    assert protected.read_bytes() == original
    assert "paths must be different" in capsys.readouterr().err


def test_cli_supports_custom_multilabel_and_weight_fields(tmp_path, capsys) -> None:  # type: ignore[no-untyped-def]
    data = tmp_path / "weighted.jsonl"
    rows = [
        {
            "uid": f"r{index}",
            "thread": f"g{index // 2}",
            "classes": ["shared", f"class-{index % 2}"],
            "importance": 1 + index / 10,
            "thread_importance": 2 + index // 2,
        }
        for index in range(8)
    ]
    data.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    assignments = tmp_path / "assignments.jsonl"
    manifest = tmp_path / "manifest.json"
    assert (
        main(
            [
                "split",
                str(data),
                "--id-field",
                "uid",
                "--group-field",
                "thread",
                "--label-field",
                "classes",
                "--weight-field",
                "importance",
                "--group-weight-field",
                "thread_importance",
                "--algorithm",
                "stratified-group",
                "--ratios",
                "train=0.5,test=0.5",
                "--max-local-iterations",
                "20",
                "--assignments",
                str(assignments),
                "--manifest",
                str(manifest),
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "Record weight" in output
    assert "per-label weight" in output
    value = json.loads(manifest.read_text(encoding="utf-8"))
    assert value["schema_version"] == "2"
    assert value["algorithm_version"] == "3"
    assert value["metadata"]["optimizer"] == "greedy-local-v3"


def test_cli_rejects_invalid_or_irrelevant_local_search_limit(tmp_path, capsys) -> None:  # type: ignore[no-untyped-def]
    data = tmp_path / "data.jsonl"
    write_dataset(data)
    base = [
        "--assignments",
        str(tmp_path / "assignments.jsonl"),
        "--manifest",
        str(tmp_path / "manifest.json"),
    ]
    assert (
        main(
            [
                "split",
                str(data),
                "--algorithm",
                "group",
                "--max-local-iterations",
                "-1",
                *base,
            ]
        )
        == 2
    )
    assert "max_local_iterations" in capsys.readouterr().err
    assert (
        main(
            [
                "split",
                str(data),
                "--algorithm",
                "hash",
                "--max-local-iterations",
                "0",
                *base,
            ]
        )
        == 2
    )
    assert "only valid for group algorithms" in capsys.readouterr().err
