"""JSON and JSONL record input/output."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .jsonutil import strict_dumps, strict_loads
from .models import Assignment, Record


def load_records(
    path: str | Path,
    *,
    id_field: str = "id",
    group_field: str = "group",
    label_field: str = "label",
    weight_field: str = "weight",
    group_weight_field: str = "group_weight",
) -> tuple[Record, ...]:
    """Load records from a JSON array or newline-delimited JSON objects."""
    source = Path(path)
    text = source.read_text(encoding="utf-8")
    if source.suffix.lower() == ".jsonl":
        values: list[Any] = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                values.append(strict_loads(line))
            except ValueError as error:
                raise ValueError(f"line {line_number}: {error}") from error
    else:
        decoded = strict_loads(text)
        if not isinstance(decoded, list):
            raise ValueError("JSON dataset root must be an array")
        values = decoded
    records: list[Record] = []
    for index, value in enumerate(values, start=1):
        if not isinstance(value, Mapping):
            raise ValueError(f"record {index} must be a JSON object")
        records.append(
            Record.from_mapping(
                value,
                id_field=id_field,
                group_field=group_field,
                label_field=label_field,
                weight_field=weight_field,
                group_weight_field=group_weight_field,
            )
        )
    return tuple(records)


def save_assignments(assignments: Iterable[Assignment], path: str | Path) -> None:
    """Write assignments as deterministic JSONL."""
    lines = [
        strict_dumps(
            {"id": item.record_id, "split": item.split, "fold": item.fold},
            ensure_ascii=False,
            sort_keys=True,
        )
        for item in sorted(assignments, key=lambda value: value.record_id)
    ]
    Path(path).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def load_assignments(path: str | Path) -> tuple[Assignment, ...]:
    """Load strict assignment JSONL with exact fields and stable scalar types."""
    result: list[Assignment] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = strict_loads(line)
        except ValueError as error:
            raise ValueError(f"line {line_number}: {error}") from error
        if not isinstance(value, Mapping):
            raise ValueError(f"assignment line {line_number} must be a JSON object")
        fields = {"id", "split", "fold"}
        if set(value) != fields:
            raise ValueError(f"assignment line {line_number} has invalid fields")
        record_id, split, fold = value["id"], value["split"], value["fold"]
        if not isinstance(record_id, str) or not record_id.strip():
            raise ValueError(f"assignment line {line_number} has an invalid id")
        if not isinstance(split, str) or not split:
            raise ValueError(f"assignment line {line_number} has an invalid split")
        if fold is not None and (isinstance(fold, bool) or not isinstance(fold, int) or fold < 0):
            raise ValueError(f"assignment line {line_number} has an invalid fold")
        result.append(Assignment(record_id, split, fold))
    return tuple(result)
