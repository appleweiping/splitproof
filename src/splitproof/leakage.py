"""Deterministic cross-split duplicate and leakage auditing."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .models import Assignment, Record

_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True, slots=True)
class LeakageFinding:
    """One normalized payload value observed in more than one split."""

    field: str
    record_id: str
    other_record_id: str
    split: str
    other_split: str
    value_digest: str

    def to_dict(self) -> dict[str, str]:
        return {
            "field": self.field,
            "record_id": self.record_id,
            "other_record_id": self.other_record_id,
            "split": self.split,
            "other_split": self.other_split,
            "value_digest": self.value_digest,
        }


@dataclass(frozen=True, slots=True)
class LeakageReport:
    """Bounded, auditable result of a cross-split leakage scan."""

    records: int
    assigned: int
    fields: tuple[str, ...]
    findings: tuple[LeakageFinding, ...]
    truncated: bool = False

    @property
    def valid(self) -> bool:
        """Return whether no cross-split duplicate was found."""

        return not self.findings and not self.truncated

    @property
    def pairs(self) -> int:
        """Return the number of reported leaking record pairs."""

        return len(self.findings)

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": {
                "records": self.records,
                "assigned": self.assigned,
                "fields": list(self.fields),
                "pairs": self.pairs,
                "truncated": self.truncated,
                "valid": self.valid,
            },
            "findings": [finding.to_dict() for finding in self.findings],
        }


def audit_leakage(
    records: Iterable[Record],
    assignments: Iterable[Assignment] | Mapping[str, str],
    *,
    fields: Sequence[str] = ("text", "content", "question", "answer"),
    min_length: int = 1,
    max_findings: int = 10_000,
) -> LeakageReport:
    """Find normalized exact duplicates that cross split boundaries.

    The scan hashes canonicalized payload values and retains only one small
    representative per value, so it does not copy the full dataset into the
    report. Values are compared within each named field; a ``text`` value is
    never compared with a ``content`` value. Missing fields are ignored.
    ``max_findings`` is a safety bound for pathological duplicate corpora.
    """

    if not fields or any(not isinstance(field, str) or not field.strip() for field in fields):
        raise ValueError("fields must contain non-empty names")
    normalized_fields = tuple(dict.fromkeys(field.strip() for field in fields))
    if min_length < 0:
        raise ValueError("min_length must be non-negative")
    if max_findings < 1:
        raise ValueError("max_findings must be positive")
    rows = tuple(records)
    split_by_id = _assignment_map(assignments)
    if len({row.id for row in rows}) != len(rows):
        raise ValueError("records contain duplicate IDs")
    findings: list[LeakageFinding] = []
    seen: dict[tuple[str, str], list[tuple[str, str]]] = {}
    truncated = False
    for row in sorted(rows, key=lambda item: item.id):
        split = split_by_id.get(row.id)
        if split is None:
            continue
        for field in normalized_fields:
            value = row.payload.get(field)
            normalized = _normalize(value)
            if normalized is None or len(normalized) < min_length:
                continue
            digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
            bucket = seen.setdefault((field, digest), [])
            for previous_id, previous_split in bucket:
                if previous_split == split:
                    continue
                if len(findings) >= max_findings:
                    truncated = True
                    break
                findings.append(
                    LeakageFinding(
                        field=field,
                        record_id=previous_id,
                        other_record_id=row.id,
                        split=previous_split,
                        other_split=split,
                        value_digest=digest,
                    )
                )
            if truncated:
                break
            bucket.append((row.id, split))
        if truncated:
            break
    findings.sort(key=lambda item: (item.field, item.record_id, item.other_record_id))
    return LeakageReport(len(rows), len(split_by_id), normalized_fields, tuple(findings), truncated)


def _assignment_map(assignments: Iterable[Assignment] | Mapping[str, str]) -> dict[str, str]:
    if isinstance(assignments, Mapping):
        result = dict(assignments)
    else:
        result = {}
        for assignment in assignments:
            if assignment.record_id in result:
                raise ValueError(f"duplicate assignment for {assignment.record_id!r}")
            result[assignment.record_id] = assignment.split
    if any(not isinstance(identifier, str) or not identifier.strip() for identifier in result):
        raise ValueError("assignment IDs must be non-empty strings")
    if any(not isinstance(split, str) or not split.strip() for split in result.values()):
        raise ValueError("assignment split names must be non-empty strings")
    return result


def _normalize(value: Any) -> str | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, str):
        return _WHITESPACE.sub(" ", value).strip().casefold()
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError):
        return None
