"""Core records with frozen fields and intentionally mutable mapping values."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any


def _stable_scalar(value: Any, field_name: str, *, allow_none: bool) -> str | None:
    if value is None:
        if allow_none:
            return None
        raise ValueError(f"{field_name} must not be null")
    if isinstance(value, str):
        result = value
    elif isinstance(value, bool):
        result = "true" if value else "false"
    elif isinstance(value, int):
        result = str(value)
    elif isinstance(value, float) and math.isfinite(value):
        result = json.dumps(value, allow_nan=False)
    else:
        raise ValueError(f"{field_name} must be a finite JSON scalar")
    if field_name == "record id" and not result.strip():
        raise ValueError("record id must not be empty")
    return result


@dataclass(frozen=True, slots=True)
class Record:
    """A dataset row reduced to the fields relevant to splitting.

    ``payload`` is deliberately excluded from split decisions. It can be used by
    callers to retain application data while SplitProof handles stable identity.
    """

    id: str
    group: str | None = None
    label: str | None = None
    payload: Mapping[str, Any] = field(default_factory=dict, compare=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("record id must be a non-empty string")
        if self.group is not None and not isinstance(self.group, str):
            raise ValueError("record group must be a string or null")
        if self.label is not None and not isinstance(self.label, str):
            raise ValueError("record label must be a string or null")
        object.__setattr__(self, "payload", dict(self.payload))

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        id_field: str = "id",
        group_field: str = "group",
        label_field: str = "label",
    ) -> Record:
        """Create a record from a JSON-compatible mapping."""
        if id_field not in value:
            raise ValueError(f"record is missing required field {id_field!r}")
        identifier = _stable_scalar(value[id_field], "record id", allow_none=False)
        assert identifier is not None
        group = _stable_scalar(value.get(group_field), "record group", allow_none=True)
        label = _stable_scalar(value.get(label_field), "record label", allow_none=True)
        return cls(
            id=identifier,
            group=group,
            label=label,
            payload=dict(value),
        )


@dataclass(frozen=True, slots=True)
class Assignment:
    """The partition (and optional fold) assigned to one record."""

    record_id: str
    split: str
    fold: int | None = None


@dataclass(frozen=True, slots=True)
class SplitDiagnostics:
    """Measured properties of an assignment."""

    counts: Mapping[str, int]
    ratios: Mapping[str, float]
    label_counts: Mapping[str, Mapping[str, int]]
    max_ratio_deviation: float
    group_leakage: tuple[str, ...]
    missing_ids: tuple[str, ...]
    unexpected_ids: tuple[str, ...]

    @property
    def valid(self) -> bool:
        """Whether coverage and group-isolation invariants hold."""
        return not (self.group_leakage or self.missing_ids or self.unexpected_ids)


@dataclass(frozen=True, slots=True)
class SplitManifest:
    """Portable record of an exact split operation."""

    schema_version: str
    algorithm: str
    algorithm_version: str
    seed: str
    created_at: str
    data_fingerprint: str
    ratios: Mapping[str, float]
    assignments: tuple[Assignment, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    checksum: str | None = None

    def to_dict(self, *, include_checksum: bool = True) -> dict[str, Any]:
        """Return a JSON-serializable dictionary with stable assignment order."""
        result: dict[str, Any] = {
            "schema_version": self.schema_version,
            "algorithm": self.algorithm,
            "algorithm_version": self.algorithm_version,
            "seed": self.seed,
            "created_at": self.created_at,
            "data_fingerprint": self.data_fingerprint,
            "ratios": dict(sorted(self.ratios.items())),
            "assignments": [
                asdict(item) for item in sorted(self.assignments, key=lambda item: item.record_id)
            ],
            "metadata": dict(self.metadata),
        }
        if include_checksum:
            result["checksum"] = self.checksum
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SplitManifest:
        """Parse a manifest dictionary, rejecting unknown or absent fields."""
        required = {
            "schema_version",
            "algorithm",
            "algorithm_version",
            "seed",
            "created_at",
            "data_fingerprint",
            "ratios",
            "assignments",
            "metadata",
            "checksum",
        }
        missing = sorted(required - value.keys())
        if missing:
            raise ValueError(f"manifest missing fields: {', '.join(missing)}")
        unknown = sorted(value.keys() - required)
        if unknown:
            raise ValueError(f"manifest has unknown fields: {', '.join(unknown)}")
        ratios_value = value["ratios"]
        assignments_value = value["assignments"]
        metadata_value = value["metadata"]
        if not isinstance(ratios_value, Mapping):
            raise ValueError("manifest ratios must be an object")
        if not isinstance(assignments_value, list):
            raise ValueError("manifest assignments must be an array")
        if not isinstance(metadata_value, Mapping):
            raise ValueError("manifest metadata must be an object")
        string_fields = (
            "schema_version",
            "algorithm",
            "algorithm_version",
            "seed",
            "created_at",
            "data_fingerprint",
        )
        for field_name in string_fields:
            if not isinstance(value[field_name], str) or not value[field_name]:
                raise ValueError(f"manifest {field_name} must be a non-empty string")
        if not isinstance(value["checksum"], str) or not value["checksum"]:
            raise ValueError("manifest checksum must be a non-empty string")
        ratios: dict[str, float] = {}
        for name, ratio in ratios_value.items():
            if not isinstance(name, str):
                raise ValueError("manifest ratio names must be strings")
            if isinstance(ratio, bool) or not isinstance(ratio, (int, float)):
                raise ValueError(f"manifest ratio {name!r} must be a number")
            ratios[name] = float(ratio)

        assignments: list[Assignment] = []
        assignment_fields = {"record_id", "split", "fold"}
        for index, item in enumerate(assignments_value):
            if not isinstance(item, Mapping):
                raise ValueError(f"manifest assignment {index} must be an object")
            missing_assignment = assignment_fields - item.keys()
            unknown_assignment = item.keys() - assignment_fields
            if missing_assignment or unknown_assignment:
                raise ValueError(f"manifest assignment {index} has invalid fields")
            record_id = item["record_id"]
            split = item["split"]
            fold = item["fold"]
            if not isinstance(record_id, str) or not record_id.strip():
                raise ValueError(f"manifest assignment {index} has an invalid record_id")
            if not isinstance(split, str) or not split:
                raise ValueError(f"manifest assignment {index} has an invalid split")
            if fold is not None and (
                isinstance(fold, bool) or not isinstance(fold, int) or fold < 0
            ):
                raise ValueError(f"manifest assignment {index} has an invalid fold")
            assignments.append(Assignment(record_id, split, fold))
        return cls(
            schema_version=value["schema_version"],
            algorithm=value["algorithm"],
            algorithm_version=value["algorithm_version"],
            seed=value["seed"],
            created_at=value["created_at"],
            data_fingerprint=value["data_fingerprint"],
            ratios=ratios,
            assignments=tuple(assignments),
            metadata=dict(metadata_value),
            checksum=value["checksum"],
        )
