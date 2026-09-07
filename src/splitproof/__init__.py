"""SplitProof: reproducible, group-aware dataset partitioning."""

from .assigners import balanced_group_split, hash_split, stratified_group_split
from .diagnostics import diagnose
from .kfold import assign_kfold
from .manifest import create_manifest, load_manifest, save_manifest, verify_manifest
from .models import Assignment, Record, SplitDiagnostics, SplitManifest
from .nested import NestedSplit, nested_group_kfold
from .repeat import StabilityReport, repeated_kfold, stability_report
from .temporal import TemporalFold, TimeInterval, purged_holdout, purged_kfold

__all__ = [
    "Assignment",
    "NestedSplit",
    "Record",
    "SplitDiagnostics",
    "SplitManifest",
    "StabilityReport",
    "TemporalFold",
    "TimeInterval",
    "assign_kfold",
    "balanced_group_split",
    "create_manifest",
    "diagnose",
    "hash_split",
    "load_manifest",
    "nested_group_kfold",
    "purged_holdout",
    "purged_kfold",
    "repeated_kfold",
    "save_manifest",
    "stability_report",
    "stratified_group_split",
    "verify_manifest",
]

__version__ = "0.2.0"
