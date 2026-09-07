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
_TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)


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


@dataclass(frozen=True, slots=True)
class NearDuplicateFinding:
    """One cross-split pair whose shingle similarity crosses a threshold."""

    field: str
    record_id: str
    other_record_id: str
    split: str
    other_split: str
    similarity: float

    def to_dict(self) -> dict[str, object]:
        return {
            "field": self.field,
            "record_id": self.record_id,
            "other_record_id": self.other_record_id,
            "split": self.split,
            "other_split": self.other_split,
            "similarity": self.similarity,
        }


@dataclass(frozen=True, slots=True)
class NearDuplicateReport:
    """Bounded report for approximate cross-split duplicate detection."""

    records: int
    assigned: int
    field: str
    threshold: float
    findings: tuple[NearDuplicateFinding, ...]
    truncated: bool = False

    @property
    def valid(self) -> bool:
        return not self.findings and not self.truncated

    def to_dict(self) -> dict[str, object]:
        return {
            "summary": {
                "records": self.records,
                "assigned": self.assigned,
                "field": self.field,
                "threshold": self.threshold,
                "pairs": len(self.findings),
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


def audit_near_duplicates(
    records: Iterable[Record],
    assignments: Iterable[Assignment] | Mapping[str, str],
    *,
    field: str = "text",
    threshold: float = 0.8,
    min_tokens: int = 3,
    max_pairs: int = 10_000,
) -> NearDuplicateReport:
    """Find cross-split near duplicates using deterministic token shingles.

    The inverted shingle index avoids comparing unrelated records and retains
    no source text in the report. Similarity is ordinary Jaccard similarity of
    normalized token trigrams; the threshold is inclusive. This is a review
    diagnostic, not a semantic-equivalence claim.
    """

    if not isinstance(field, str) or not field.strip():
        raise ValueError("field must be a non-empty name")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise TypeError("threshold must be a real number")
    if not 0 <= float(threshold) <= 1:
        raise ValueError("threshold must be between zero and one")
    if isinstance(min_tokens, bool) or not isinstance(min_tokens, int) or min_tokens < 1:
        raise ValueError("min_tokens must be a positive integer")
    if isinstance(max_pairs, bool) or not isinstance(max_pairs, int) or max_pairs < 1:
        raise ValueError("max_pairs must be a positive integer")
    rows = tuple(records)
    split_by_id = _assignment_map(assignments)
    candidates: dict[str, tuple[str, str, frozenset[str]]] = {}
    inverted: dict[str, list[str]] = {}
    for row in sorted(rows, key=lambda item: item.id):
        split = split_by_id.get(row.id)
        if split is None:
            continue
        value = row.payload.get(field)
        if not isinstance(value, str):
            continue
        tokens = tuple(_TOKEN_RE.findall(value.casefold()))
        if len(tokens) < min_tokens:
            continue
        shingles = frozenset(
            " ".join(tokens[index : index + 3]) for index in range(len(tokens) - 2)
        )
        if not shingles:
            continue
        candidates[row.id] = (row.id, split, shingles)
        for shingle in shingles:
            inverted.setdefault(shingle, []).append(row.id)
    findings: list[NearDuplicateFinding] = []
    compared: set[tuple[str, str]] = set()
    truncated = False
    for record_id in sorted(candidates):
        _, split, shingles = candidates[record_id]
        related = sorted(
            {item for shingle in shingles for item in inverted[shingle] if item < record_id}
        )
        for other_id in related:
            pair = (other_id, record_id)
            if pair in compared:
                continue
            compared.add(pair)
            _, other_split, other_shingles = candidates[other_id]
            if split == other_split:
                continue
            similarity = len(shingles & other_shingles) / len(shingles | other_shingles)
            if similarity < float(threshold):
                continue
            if len(findings) >= max_pairs:
                truncated = True
                break
            findings.append(
                NearDuplicateFinding(field, other_id, record_id, other_split, split, similarity)
            )
        if truncated:
            break
    findings.sort(key=lambda item: (item.record_id, item.other_record_id))
    return NearDuplicateReport(
        len(rows), len(split_by_id), field, float(threshold), tuple(findings), truncated
    )


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
