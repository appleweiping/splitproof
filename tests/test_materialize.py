from __future__ import annotations

import json

import pytest

from splitproof import Record, create_manifest, hash_split, partition_records, write_materialized


def _records() -> tuple[Record, ...]:
    return tuple(
        Record(str(index), group=f"g{index // 2}", payload={"id": str(index), "text": f"t{index}"})
        for index in range(4)
    )


def test_partition_and_materialize_preserve_payload_order(tmp_path) -> None:  # type: ignore[no-untyped-def]
    records = _records()
    assignments = hash_split(records, {"train": 0.5, "test": 0.5}, seed="materialize")
    manifest = create_manifest(
        records,
        assignments,
        algorithm="hash",
        algorithm_version="3",
        seed="materialize",
        ratios={"train": 0.5, "test": 0.5},
    )
    grouped = partition_records(records, manifest)
    assert sum(len(items) for items in grouped.values()) == len(records)
    paths = write_materialized(records, manifest, tmp_path / "out")
    assert [path.name for path in paths] == ["test.jsonl", "train.jsonl"]
    assert all(
        json.loads(line)["text"].startswith("t")
        for path in paths
        for line in path.read_text().splitlines()
    )


def test_materialize_rejects_tampering_and_overwrite(tmp_path) -> None:  # type: ignore[no-untyped-def]
    records = _records()
    assignments = hash_split(records, {"train": 1.0}, seed="materialize")
    manifest = create_manifest(
        records,
        assignments,
        algorithm="hash",
        algorithm_version="3",
        seed="materialize",
        ratios={"train": 1.0},
    )
    changed = (*records[:-1], Record("changed", payload={"id": "changed"}))
    with pytest.raises(ValueError, match="unverified"):
        partition_records(changed, manifest)
    output = tmp_path / "out"
    write_materialized(records, manifest, output)
    with pytest.raises(FileExistsError, match="overwrite"):
        write_materialized(records, manifest, output)
