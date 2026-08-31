"""Creation, persistence, and verification of split manifests."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from .constraints import ConstraintError, validate_ratios, validate_records
from .diagnostics import diagnose
from .hashing import HASH_ALGORITHM, HASH_VERSION, data_fingerprint, stable_digest
from .jsonutil import strict_dumps, strict_loads
from .models import Assignment, Record, SplitManifest

SCHEMA_VERSION = "1"
RESERVED_METADATA = frozenset({"hash_algorithm", "hash_version"})


def manifest_checksum(manifest: SplitManifest) -> str:
    """Checksum all semantic manifest fields except the checksum itself."""
    body = strict_dumps(
        manifest.to_dict(include_checksum=False),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return stable_digest(body, seed="", domain="manifest")


def create_manifest(
    records: Iterable[Record],
    assignments: Iterable[Assignment],
    *,
    algorithm: str,
    algorithm_version: str,
    seed: str | int,
    ratios: Mapping[str, float] | None = None,
    metadata: Mapping[str, object] | None = None,
) -> SplitManifest:
    """Create a checksummed manifest after validating assignment coverage."""
    rows = validate_records(records)
    assigned = tuple(assignments)
    checked_ratios = validate_ratios(ratios) if ratios is not None else None
    report = diagnose(rows, assigned, checked_ratios)
    coverage_invalid = bool(report.missing_ids or report.unexpected_ids)
    leakage_invalid = bool(report.group_leakage) and algorithm != "hash"
    if coverage_invalid or leakage_invalid:
        raise ConstraintError(
            "cannot create manifest with coverage or group-leakage errors: "
            f"missing={len(report.missing_ids)}, unexpected={len(report.unexpected_ids)}, "
            f"leaking_groups={len(report.group_leakage)}"
        )
    if len({item.record_id for item in assigned}) != len(assigned):
        raise ConstraintError("a manifest cannot assign one record more than once")
    if checked_ratios is not None:
        unknown_splits = sorted({item.split for item in assigned} - checked_ratios.keys())
        if unknown_splits:
            raise ConstraintError(
                "assignments use splits absent from ratios: " + ", ".join(unknown_splits)
            )
    supplied_metadata = dict(metadata or {})
    if any(not isinstance(key, str) for key in supplied_metadata):
        raise ConstraintError("metadata keys must be strings")
    reserved = sorted(RESERVED_METADATA & supplied_metadata.keys())
    if reserved:
        raise ConstraintError("metadata cannot override reserved fields: " + ", ".join(reserved))
    manifest = SplitManifest(
        schema_version=SCHEMA_VERSION,
        algorithm=algorithm,
        algorithm_version=algorithm_version,
        seed=str(seed),
        created_at=datetime.now(timezone.utc).isoformat(),
        data_fingerprint=data_fingerprint(rows),
        ratios=dict(checked_ratios or report.ratios),
        assignments=assigned,
        metadata={
            "hash_algorithm": HASH_ALGORITHM,
            "hash_version": HASH_VERSION,
            **supplied_metadata,
        },
    )
    return replace(manifest, checksum=manifest_checksum(manifest))


def save_manifest(manifest: SplitManifest, path: str | Path) -> None:
    """Save an indented UTF-8 JSON manifest."""
    Path(path).write_text(
        strict_dumps(manifest.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_manifest(path: str | Path) -> SplitManifest:
    """Load and structurally parse a manifest."""
    value = strict_loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("manifest root must be a JSON object")
    manifest = SplitManifest.from_dict(value)
    return replace(manifest, ratios=validate_ratios(manifest.ratios))


def verify_manifest(
    manifest: SplitManifest,
    records: Iterable[Record],
    assignments: Iterable[Assignment] | None = None,
) -> tuple[str, ...]:
    """Return human-readable verification errors; an empty tuple means valid."""
    rows = tuple(records)
    errors: list[str] = []
    try:
        validate_records(rows)
    except ConstraintError as error:
        errors.append(f"invalid dataset: {error}")
    try:
        checked_ratios = validate_ratios(manifest.ratios)
    except ConstraintError as error:
        errors.append(f"invalid manifest ratios: {error}")
        checked_ratios = dict(manifest.ratios)
    if manifest.schema_version != SCHEMA_VERSION:
        errors.append(f"unsupported schema version {manifest.schema_version!r}")
    if manifest.checksum != manifest_checksum(manifest):
        errors.append("manifest checksum mismatch")
    if manifest.metadata.get("hash_algorithm") != HASH_ALGORITHM:
        errors.append("unsupported manifest hash algorithm")
    if manifest.metadata.get("hash_version") != HASH_VERSION:
        errors.append("unsupported manifest hash version")
    if manifest.data_fingerprint != data_fingerprint(rows):
        errors.append("dataset fingerprint mismatch")
    report = diagnose(rows, manifest.assignments, checked_ratios)
    if report.missing_ids:
        errors.append(f"missing assignments for {len(report.missing_ids)} records")
    if report.unexpected_ids:
        errors.append(f"assignments contain {len(report.unexpected_ids)} unknown records")
    if report.group_leakage and manifest.algorithm != "hash":
        errors.append(f"group leakage detected for {len(report.group_leakage)} groups")
    if len({item.record_id for item in manifest.assignments}) != len(manifest.assignments):
        errors.append("duplicate assignment record ids")
    declared_splits = set(manifest.ratios)
    unknown_splits = sorted({item.split for item in manifest.assignments} - declared_splits)
    if unknown_splits:
        errors.append("assignments use splits absent from ratios: " + ", ".join(unknown_splits))
    if assignments is not None:
        external = tuple(assignments)
        if len({item.record_id for item in external}) != len(external):
            errors.append("external assignments contain duplicate record ids")
        expected = tuple(sorted(manifest.assignments, key=lambda item: item.record_id))
        observed = tuple(sorted(external, key=lambda item: item.record_id))
        if observed != expected:
            errors.append("external assignments do not match the manifest")
    return tuple(errors)
