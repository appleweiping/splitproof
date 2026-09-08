from __future__ import annotations

import json
from pathlib import Path

import pytest

from splitproof import (
    hash_split,
    hash_split_stream,
    iter_records,
    verify_hash_split_stream,
    write_hash_split_stream,
)
from splitproof.cli import main
from splitproof.models import Record


def _records() -> tuple[Record, ...]:
    return tuple(Record(str(index), payload={"id": str(index)}) for index in range(1, 8))


def _write_jsonl(path: Path, records: tuple[Record, ...] | None = None) -> None:
    rows = records or _records()
    path.write_text(
        "".join(json.dumps({"id": record.id, "text": record.id}) + "\n" for record in rows),
        encoding="utf-8",
    )


def test_hash_split_stream_matches_hash_split_set_and_rejects_duplicates() -> None:
    records = _records()
    streamed = tuple(hash_split_stream(reversed(records), {"train": 0.7, "test": 0.3}, seed="s"))
    expected = hash_split(records, {"train": 0.7, "test": 0.3}, seed="s")
    assert {(item.record_id, item.split) for item in streamed} == {
        (item.record_id, item.split) for item in expected
    }
    with pytest.raises(ValueError, match="duplicate"):
        tuple(hash_split_stream((records[0], records[0]), {"train": 1.0}))
    with pytest.raises(TypeError, match="Record"):
        tuple(hash_split_stream((object(),), {"train": 1.0}))  # type: ignore[arg-type]


def test_hash_split_stream_commits_bounded_duplicate_store(tmp_path: Path) -> None:
    records = tuple(Record(str(index), payload={"id": str(index)}) for index in range(512))
    assignments = tmp_path / "assignments.jsonl"
    report = write_hash_split_stream(records, {"train": 1.0}, assignments)
    assert report.records == 512
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report.to_dict()), encoding="utf-8")
    assert verify_hash_split_stream(assignments, report_path).records == 512


def test_write_hash_split_stream_is_atomic_and_reports_digest(tmp_path: Path) -> None:
    destination = tmp_path / "assignments.jsonl"
    report = write_hash_split_stream(_records(), {"train": 0.5, "test": 0.5}, destination, seed=4)
    assert report.records == 7
    assert sum(report.counts.values()) == 7
    assert len(report.assignment_digest) == 64
    assert len(destination.read_text(encoding="utf-8").splitlines()) == 7
    destination.write_text("original\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        write_hash_split_stream(
            (_records()[0], _records()[0]),
            {"train": 1.0},
            destination,
        )
    assert destination.read_text(encoding="utf-8") == "original\n"


def test_verify_hash_split_stream_detects_tampering(tmp_path: Path) -> None:
    destination = tmp_path / "assignments.jsonl"
    report_path = tmp_path / "report.json"
    report = write_hash_split_stream(_records(), {"train": 1.0}, destination, seed="x")
    report_path.write_text(json.dumps(report.to_dict()), encoding="utf-8")
    assert verify_hash_split_stream(destination, report_path).records == report.records
    destination.write_text(
        destination.read_text(encoding="utf-8").replace('"train"', '"test"', 1), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="does not match"):
        verify_hash_split_stream(destination, report_path)


def test_iter_records_streams_jsonl_and_cli_emits_report(tmp_path: Path) -> None:
    source = tmp_path / "records.jsonl"
    _write_jsonl(source)
    iterator = iter_records(source)
    assert next(iterator).id == "1"
    assert [record.id for record in iterator] == ["2", "3", "4", "5", "6", "7"]
    assignments = tmp_path / "assignments.jsonl"
    report = tmp_path / "report.json"
    assert (
        main(
            [
                "hash-stream",
                str(source),
                "--ratios",
                "train=0.5,test=0.5",
                "--assignments",
                str(assignments),
                "--report",
                str(report),
            ]
        )
        == 0
    )
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["records"] == 7
    assert payload["algorithm"] == "record-hash-stream-v1"
    assert main(["hash-stream-verify", str(assignments), str(report)]) == 0


def test_streaming_cli_rejects_non_jsonl(tmp_path: Path) -> None:
    source = tmp_path / "records.json"
    source.write_text("[]", encoding="utf-8")
    assert (
        main(
            [
                "hash-stream",
                str(source),
                "--ratios",
                "train=1",
                "--assignments",
                str(tmp_path / "out.jsonl"),
            ]
        )
        == 2
    )


def test_verify_hash_split_stream_rejects_bad_report_and_rows(tmp_path: Path) -> None:
    assignments = tmp_path / "assignments.jsonl"
    report = tmp_path / "report.json"
    assignments.write_text(
        '{"id":"a","split":"train"}\n{"id":"a","split":"train"}\n', encoding="utf-8"
    )
    report.write_text(json.dumps({"algorithm": "wrong"}), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported"):
        verify_hash_split_stream(assignments, report)
    report.write_text(json.dumps({"algorithm": "record-hash-stream-v1"}), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        verify_hash_split_stream(assignments, report)
