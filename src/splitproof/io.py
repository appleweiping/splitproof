"""Read and write rows as JSON Lines, JSON arrays, CSV/TSV or Parquet."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

FORMATS = ".jsonl, .json, .csv, .tsv or .parquet"


def _pyarrow() -> Any:
    try:
        import pyarrow.parquet as parquet
    except ImportError:  # pragma: no cover - depends on the environment
        raise ValueError(
            "reading Parquet needs pyarrow: pip install 'splitproof[parquet]'"
        ) from None
    return parquet


def read_rows(path: str | Path) -> list[dict[str, Any]]:
    """Load a dataset as a list of dicts, choosing the format by file extension."""
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix in {".csv", ".tsv"}:
        with source.open(encoding="utf-8", newline="") as stream:
            return list(csv.DictReader(stream, delimiter="\t" if suffix == ".tsv" else ","))
    if suffix in {".jsonl", ".ndjson"}:
        rows = []
        with source.open(encoding="utf-8") as stream:
            for number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"{source}:{number}: invalid JSON ({error.msg})") from error
                if not isinstance(value, dict):
                    raise ValueError(f"{source}:{number}: expected a JSON object")
                rows.append(value)
        return rows
    if suffix == ".json":
        value = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
            raise ValueError(f"{source}: expected a JSON array of objects")
        return value
    if suffix == ".parquet":
        if not source.exists():
            raise FileNotFoundError(f"{source}: no such file")
        rows = _pyarrow().read_table(source).to_pylist()
        return [dict(row) for row in rows]
    raise ValueError(f"{source}: unsupported file type (use {FORMATS})")


def write_rows(rows: Iterable[dict[str, Any]], path: str | Path) -> None:
    """Write rows in the format implied by the file extension."""
    target = Path(path)
    suffix = target.suffix.lower()
    materialized = list(rows)
    if suffix in {".csv", ".tsv"}:
        fields: dict[str, None] = {}
        for row in materialized:
            fields.update(dict.fromkeys(row))
        with target.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=list(fields), delimiter="\t" if suffix == ".tsv" else ","
            )
            writer.writeheader()
            writer.writerows(materialized)
    elif suffix in {".jsonl", ".ndjson"}:
        with target.open("w", encoding="utf-8") as stream:
            for row in materialized:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    elif suffix == ".json":
        target.write_text(json.dumps(materialized, ensure_ascii=False, indent=1) + "\n", "utf-8")
    elif suffix == ".parquet":
        parquet = _pyarrow()
        import pyarrow

        parquet.write_table(pyarrow.Table.from_pylist(materialized), target)
    else:
        raise ValueError(f"{target}: unsupported file type (use {FORMATS})")
