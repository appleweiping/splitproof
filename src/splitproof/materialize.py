"""Materialize verified split assignments into deterministic JSONL datasets."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType

from .jsonutil import strict_dumps
from .manifest import verify_manifest
from .models import Record, SplitManifest

_SAFE_SPLIT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def partition_records(
    records: Sequence[Record], manifest: SplitManifest
) -> Mapping[str, tuple[Record, ...]]:
    """Return records grouped by a verified manifest split without mutation.

    The dataset fingerprint, coverage, assignment IDs, and manifest checksum are
    checked before any partition is returned. Record order follows the input
    sequence, which keeps materialized files stable while preserving source order.
    """

    errors = verify_manifest(manifest, records)
    if errors:
        raise ValueError("cannot materialize unverified manifest: " + "; ".join(errors))
    assignments = {item.record_id: item.split for item in manifest.assignments}
    grouped: dict[str, list[Record]] = {name: [] for name in manifest.ratios}
    for record in records:
        split = assignments.get(record.id)
        if split is None:
            raise ValueError(f"manifest has no split for record {record.id!r}")
        grouped.setdefault(split, []).append(record)
    return MappingProxyType({name: tuple(rows) for name, rows in sorted(grouped.items())})


def write_materialized(
    records: Sequence[Record], manifest: SplitManifest, output_dir: str | Path
) -> tuple[Path, ...]:
    """Write verified split payloads as ``<split>.jsonl`` files.

    Existing files are never overwritten. The original record payload is emitted
    unchanged; split metadata remains in the authenticated manifest and can be
    joined through record IDs.
    """

    destination = Path(output_dir)
    if destination.exists() and not destination.is_dir():
        raise ValueError(f"materialization path is not a directory: {destination}")
    grouped = partition_records(records, manifest)
    destination.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for split, items in grouped.items():
        if not _SAFE_SPLIT.fullmatch(split):
            raise ValueError(f"split name is not safe for a filename: {split!r}")
        path = destination / f"{split}.jsonl"
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing materialized file: {path}")
        text = "".join(
            strict_dumps(dict(record.payload), ensure_ascii=False, sort_keys=True) + "\n"
            for record in items
        )
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(text, encoding="utf-8", newline="\n")
        temporary.replace(path)
        paths.append(path)
    return tuple(paths)
