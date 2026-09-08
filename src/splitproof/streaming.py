"""Bounded-memory record-hash assignment for JSONL datasets."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

from .assigners import _choose_by_ratio
from .constraints import validate_ratios
from .hashing import stable_unit_interval
from .jsonutil import strict_dumps
from .models import Assignment, Record


@dataclass(frozen=True, slots=True)
class StreamSplitReport:
    """Audit metadata emitted by a bounded-memory hash split."""

    algorithm: str
    records: int
    counts: Mapping[str, int]
    assignment_digest: str

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "algorithm": self.algorithm,
            "records": self.records,
            "counts": dict(sorted(self.counts.items())),
            "assignment_digest": self.assignment_digest,
        }


def hash_split_stream(
    records: Iterable[Record],
    ratios: Mapping[str, float],
    *,
    seed: str | int = "0",
) -> Iterator[Assignment]:
    """Yield append-stable hash assignments without materializing record payloads.

    The iterator retains only seen IDs while consuming input. It intentionally
    supports record-hash assignment only; group balancing requires a complete
    group inventory and remains available through :func:`balanced_group_split`.
    """

    checked = validate_ratios(ratios)
    seen: set[str] = set()
    for record in records:
        if not isinstance(record, Record):
            raise TypeError("records must contain Record values")
        if record.id in seen:
            raise ValueError(f"duplicate record ID: {record.id!r}")
        seen.add(record.id)
        destination = _choose_by_ratio(
            stable_unit_interval(record.id, seed=str(seed), domain="record-split"), checked
        )
        yield Assignment(record.id, destination)


def write_hash_split_stream(
    records: Iterable[Record],
    ratios: Mapping[str, float],
    destination: str | Path,
    *,
    seed: str | int = "0",
) -> StreamSplitReport:
    """Atomically write input-order assignments and return their digest report."""

    target = Path(destination).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    digest = hashlib.sha256()
    counts: Counter[str] = Counter()
    total = 0
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            for assignment in hash_split_stream(records, ratios, seed=seed):
                line = (
                    strict_dumps(
                        {
                            "id": assignment.record_id,
                            "split": assignment.split,
                            "fold": assignment.fold,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )
                stream.write(line)
                digest.update(line.encode("utf-8"))
                counts[assignment.split] += 1
                total += 1
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    except BaseException:
        with suppress(OSError):
            os.unlink(temporary)
        raise
    return StreamSplitReport("record-hash-stream-v1", total, dict(counts), digest.hexdigest())


def verify_hash_split_stream(assignments: str | Path, report: str | Path) -> StreamSplitReport:
    """Authenticate a streamed assignment JSONL file against its report."""
    try:
        expected = json.loads(Path(report).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read stream report: {error}") from error
    if not isinstance(expected, Mapping) or expected.get("algorithm") != "record-hash-stream-v1":
        raise ValueError("unsupported stream report")
    digest = hashlib.sha256()
    counts: Counter[str] = Counter()
    seen: set[str] = set()
    total = 0
    try:
        with Path(assignments).open("rb") as stream:
            for raw in stream:
                digest.update(raw)
                try:
                    value = json.loads(raw.decode("utf-8"))
                except (UnicodeError, json.JSONDecodeError) as error:
                    raise ValueError(f"invalid assignment JSON at row {total + 1}") from error
                if not isinstance(value, Mapping) or not isinstance(value.get("id"), str):
                    raise ValueError(f"assignment row {total + 1} must contain a string id")
                record_id = value["id"]
                split = value.get("split")
                if not isinstance(split, str) or not split:
                    raise ValueError(f"assignment row {total + 1} must contain a split")
                if record_id in seen:
                    raise ValueError(f"duplicate assignment ID: {record_id!r}")
                seen.add(record_id)
                counts[split] += 1
                total += 1
    except OSError as error:
        raise ValueError(f"cannot read assignments: {error}") from error
    actual = StreamSplitReport("record-hash-stream-v1", total, dict(counts), digest.hexdigest())
    if (
        expected.get("records") != actual.records
        or expected.get("counts") != actual.to_dict()["counts"]
        or expected.get("assignment_digest") != actual.assignment_digest
    ):
        raise ValueError("stream assignment report does not match the assignment file")
    return actual
