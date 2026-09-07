"""SplitProof: reproducible, group-aware dataset partitioning."""

from .assigners import balanced_group_split, hash_split, stratified_group_split
from .diagnostics import diagnose
from .evaluation import (
    CrossValidationReport,
    FoldScore,
    NestedEvaluationReport,
    NestedFoldScore,
    evaluate_nested,
    evaluate_repeated_kfold,
)
from .kfold import assign_kfold
from .manifest import create_manifest, load_manifest, save_manifest, verify_manifest
from .models import Assignment, Record, SplitDiagnostics, SplitManifest
from .nested import (
    NestedSplit,
    RepeatedNestedSplit,
    nested_group_kfold,
    repeated_nested_group_kfold,
)
from .repeat import (
    HoldoutStabilityReport,
    StabilityReport,
    holdout_stability_report,
    repeated_group_holdout,
    repeated_kfold,
    stability_report,
)
from .service import SplitService, create_server
from .temporal import TemporalFold, TimeInterval, purged_holdout, purged_kfold

__all__ = [
    "Assignment",
    "CrossValidationReport",
    "FoldScore",
    "HoldoutStabilityReport",
    "NestedEvaluationReport",
    "NestedFoldScore",
    "NestedSplit",
    "Record",
    "RepeatedNestedSplit",
    "SplitDiagnostics",
    "SplitManifest",
    "SplitService",
    "StabilityReport",
    "TemporalFold",
    "TimeInterval",
    "assign_kfold",
    "balanced_group_split",
    "create_manifest",
    "create_server",
    "diagnose",
    "evaluate_nested",
    "evaluate_repeated_kfold",
    "hash_split",
    "holdout_stability_report",
    "load_manifest",
    "nested_group_kfold",
    "purged_holdout",
    "purged_kfold",
    "repeated_group_holdout",
    "repeated_kfold",
    "repeated_nested_group_kfold",
    "save_manifest",
    "stability_report",
    "stratified_group_split",
    "verify_manifest",
]

__version__ = "0.2.0"
