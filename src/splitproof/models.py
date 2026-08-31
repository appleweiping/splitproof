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


def _positive_weight(value: Any, field_name: str, *, allow_none: bool) -> float | None:
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a positive finite number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{field_name} must be a positive finite number")
    return result


def _label_values(value: Any, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    values = value if isinstance(value, list) else [value]
    labels: list[str] = []
    for item in values:
        label = _stable_scalar(item, field_name, allow_none=False)
        assert label is not None
        if not label:
            raise ValueError(f"{field_name} values must not be empty")
        labels.append(label)
    if len(labels) != len(set(labels)):
        raise ValueError(f"{field_name} must not contain duplicate values")
    return tuple(sorted(labels))


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
    labels: tuple[str, ...] = ()
    weight: float = 1.0
    group_weight: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("record id must be a non-empty string")
        if self.group is not None and not isinstance(self.group, str):
            raise ValueError("record group must be a string or null")
        if self.label is not None and (not isinstance(self.label, str) or not self.label):
            raise ValueError("record label must be a non-empty string or null")
        labels_value: object = self.labels
        if not isinstance(labels_value, tuple) or not all(
            isinstance(label, str) and bool(label) for label in labels_value
        ):
            raise ValueError("record labels must be non-empty strings")
        if len(labels_value) != len(set(labels_value)):
            raise ValueError("record labels must not contain duplicates")
        normalized_labels = tuple(
            sorted(set(labels_value) | ({self.label} if self.label else set()))
        )
        normalized_weight = _positive_weight(self.weight, "record weight", allow_none=False)
        normalized_group_weight = _positive_weight(
            self.group_weight, "record group_weight", allow_none=True
        )
        assert normalized_weight is not None
        object.__setattr__(self, "labels", normalized_labels)
        object.__setattr__(self, "weight", normalized_weight)
        object.__setattr__(self, "group_weight", normalized_group_weight)
        object.__setattr__(self, "payload", dict(self.payload))

    @property
    def all_labels(self) -> tuple[str, ...]:
        """Return the normalized label set used for stratification."""
        return self.labels

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        id_field: str = "id",
        group_field: str = "group",
        label_field: str = "label",
        weight_field: str = "weight",
        group_weight_field: str = "group_weight",
    ) -> Record:
        """Create a record from a JSON-compatible mapping."""
        if id_field not in value:
            raise ValueError(f"record is missing required field {id_field!r}")
        identifier = _stable_scalar(value[id_field], "record id", allow_none=False)
        assert identifier is not None
        group = _stable_scalar(value.get(group_field), "record group", allow_none=True)
        raw_label = value.get(label_field)
        if isinstance(raw_label, list):
            label = None
            labels = _label_values(raw_label, "record labels")
        else:
            label = _stable_scalar(raw_label, "record label", allow_none=True)
            labels = ()
        weight = _positive_weight(value.get(weight_field, 1.0), "record weight", allow_none=False)
        group_weight = _positive_weight(
            value.get(group_weight_field), "record group_weight", allow_none=True
        )
        assert weight is not None
        return cls(
            id=identifier,
            group=group,
            label=label,
            labels=labels,
            weight=weight,
            group_weight=group_weight,
            payload=dict(value),
        )


@dataclass(frozen=True, slots=True)
class Assignment:
    """The partition (and optional fold) assigned to one record."""

    record_id: str
    split: str
    fold: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.record_id, str) or not self.record_id.strip():
            raise ValueError("assignment record_id must be a non-empty string")
        if not isinstance(self.split, str) or not self.split:
            raise ValueError("assignment split must be a non-empty string")
        if self.fold is not None and (
            isinstance(self.fold, bool) or not isinstance(self.fold, int) or self.fold < 0
        ):
            raise ValueError("assignment fold must be a non-negative integer or null")


@dataclass(frozen=True, slots=True)
class SplitDiagnostics:
    """Measured properties of an assignment."""

    counts: Mapping[str, int]
    ratios: Mapping[str, float]
    record_weights: Mapping[str, float]
    record_weight_ratios: Mapping[str, float]
    group_weights: Mapping[str, float]
    group_weight_ratios: Mapping[str, float]
    label_counts: Mapping[str, Mapping[str, int]]
    label_weights: Mapping[str, Mapping[str, float]]
    label_deviations: Mapping[str, float]
    max_ratio_deviation: float
    max_record_weight_deviation: float
    max_group_weight_deviation: float
    max_label_deviation: float
    objective_score: float
    objective_components: Mapping[str, float]
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
            normalized_ratio = float(ratio)
            if not math.isfinite(normalized_ratio) or normalized_ratio <= 0:
                raise ValueError(f"manifest ratio {name!r} must be finite and greater than zero")
            ratios[name] = normalized_ratio

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
