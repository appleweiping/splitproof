"""Versioned stable hashing for persistent split decisions."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable

from .models import Record

HASH_ALGORITHM = "blake2b-128"
HASH_VERSION = "1"


def canonical_key(parts: Iterable[str | None]) -> bytes:
    """Encode key parts without delimiter ambiguity using canonical JSON."""
    return json.dumps(
        list(parts), ensure_ascii=False, separators=(",", ":"), sort_keys=False
    ).encode("utf-8")


def stable_digest(*parts: str | None, seed: str = "0", domain: str) -> str:
    """Return a stable hex digest with explicit seed and purpose separation."""
    digest = hashlib.blake2b(digest_size=16, person=b"splitproof-v1")
    digest.update(canonical_key((domain, seed, *parts)))
    return digest.hexdigest()


def stable_unit_interval(*parts: str | None, seed: str, domain: str) -> float:
    """Map a stable digest to the half-open interval [0, 1)."""
    integer = int(stable_digest(*parts, seed=seed, domain=domain), 16)
    # A binary64 float has 53 bits of integer precision. Taking the high bits
    # before conversion avoids rounding the maximum 128-bit values to 1.0.
    return (integer >> 75) / (1 << 53)


def data_fingerprint_v1(records: Iterable[Record]) -> str:
    """Return the schema-v1 identity/group/single-label fingerprint."""
    rows = sorted(
        ((record.id, record.group, record.label) for record in records),
        key=canonical_key,
    )
    return stable_digest(
        json.dumps(rows, ensure_ascii=False, separators=(",", ":")),
        seed="",
        domain="dataset",
    )


def data_fingerprint(records: Iterable[Record]) -> str:
    """Fingerprint v2 split inputs, including labels and both weight types."""
    rows = sorted(
        (
            (
                record.id,
                record.group,
                record.all_labels,
                record.weight,
                record.group_weight,
            )
            for record in records
        ),
        key=lambda row: json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False),
    )
    return stable_digest(
        json.dumps(rows, ensure_ascii=False, separators=(",", ":"), allow_nan=False),
        seed="",
        domain="dataset-v2",
    )
