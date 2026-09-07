"""Auditable comparisons between persisted split manifests."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

from .models import Assignment, SplitManifest


@dataclass(frozen=True, slots=True)
class ManifestComparison:
    """Deterministic record-level drift between two manifests."""

    same_data: bool
    same_algorithm: bool
    added: tuple[str, ...]
    removed: tuple[str, ...]
    moved: tuple[tuple[str, str, int | None, str, int | None], ...]
    transitions: dict[str, int]

    @property
    def changed(self) -> bool:
        return bool(self.added or self.removed or self.moved or not self.same_data)

    def to_dict(self) -> dict[str, Any]:
        return {
            "changed": self.changed,
            "same_data": self.same_data,
            "same_algorithm": self.same_algorithm,
            "added": list(self.added),
            "removed": list(self.removed),
            "moved": [
                {
                    "record_id": record_id,
                    "before_split": before_split,
                    "before_fold": before_fold,
                    "after_split": after_split,
                    "after_fold": after_fold,
                }
                for record_id, before_split, before_fold, after_split, after_fold in self.moved
            ],
            "transitions": dict(self.transitions),
        }


def compare_manifests(before: SplitManifest, after: SplitManifest) -> ManifestComparison:
    """Compare assignments while keeping dataset and algorithm drift explicit."""

    if not isinstance(before, SplitManifest) or not isinstance(after, SplitManifest):
        raise TypeError("before and after must be SplitManifest values")
    before_map = _assignment_map(before)
    after_map = _assignment_map(after)
    moved: list[tuple[str, str, int | None, str, int | None]] = []
    transitions: Counter[str] = Counter()
    for record_id in sorted(before_map.keys() & after_map.keys()):
        left, right = before_map[record_id], after_map[record_id]
        if (left.split, left.fold) != (right.split, right.fold):
            moved.append((record_id, left.split, left.fold, right.split, right.fold))
            transitions[f"{left.split}->{right.split}"] += 1
    return ManifestComparison(
        before.data_fingerprint == after.data_fingerprint,
        (before.algorithm, before.algorithm_version) == (after.algorithm, after.algorithm_version),
        tuple(sorted(after_map.keys() - before_map.keys())),
        tuple(sorted(before_map.keys() - after_map.keys())),
        tuple(moved),
        dict(sorted(transitions.items())),
    )


def _assignment_map(manifest: SplitManifest) -> dict[str, Assignment]:
    result: dict[str, Assignment] = {}
    for assignment in manifest.assignments:
        if assignment.record_id in result:
            raise ValueError(f"manifest contains duplicate assignment id {assignment.record_id!r}")
        result[assignment.record_id] = assignment
    return result
