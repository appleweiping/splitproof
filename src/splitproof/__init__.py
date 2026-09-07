"""SplitProof: reproducible, group-aware dataset partitioning."""

from .assigners import balanced_group_split, hash_split, stratified_group_split
from .comparison import ManifestComparison, compare_manifests
from .diagnostics import diagnose
from .evaluation import (
    CandidateNestedEvaluationReport,
    CandidateNestedFoldScore,
    CrossValidationReport,
    FoldScore,
    NestedEvaluationReport,
    NestedFoldScore,
    evaluate_nested,
    evaluate_nested_candidates,
    evaluate_repeated_kfold,
)
from .io import iter_records
from .kfold import assign_kfold
from .leakage import (
    LeakageFinding,
    LeakageReport,
    NearDuplicateFinding,
    NearDuplicateReport,
    audit_leakage,
    audit_near_duplicates,
)
from .manifest import (
    create_manifest,
    load_manifest,
    migrate_manifest,
    save_manifest,
    verify_manifest,
)
from .materialize import partition_records, write_materialized
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
from .streaming import StreamSplitReport, hash_split_stream, write_hash_split_stream
from .temporal import TemporalFold, TimeInterval, purged_holdout, purged_kfold

__all__ = [
    "Assignment",
    "CandidateNestedEvaluationReport",
    "CandidateNestedFoldScore",
    "CrossValidationReport",
    "FoldScore",
    "HoldoutStabilityReport",
    "LeakageFinding",
    "LeakageReport",
    "ManifestComparison",
    "NearDuplicateFinding",
    "NearDuplicateReport",
    "NestedEvaluationReport",
    "NestedFoldScore",
    "NestedSplit",
    "Record",
    "RepeatedNestedSplit",
    "SplitDiagnostics",
    "SplitManifest",
    "SplitService",
    "StabilityReport",
    "StreamSplitReport",
    "TemporalFold",
    "TimeInterval",
    "assign_kfold",
    "audit_leakage",
    "audit_near_duplicates",
    "balanced_group_split",
    "compare_manifests",
    "create_manifest",
    "create_server",
    "diagnose",
    "evaluate_nested",
    "evaluate_nested_candidates",
    "evaluate_repeated_kfold",
    "hash_split",
    "hash_split_stream",
    "holdout_stability_report",
    "iter_records",
    "load_manifest",
    "migrate_manifest",
    "nested_group_kfold",
    "partition_records",
    "purged_holdout",
    "purged_kfold",
    "repeated_group_holdout",
    "repeated_kfold",
    "repeated_nested_group_kfold",
    "save_manifest",
    "stability_report",
    "stratified_group_split",
    "verify_manifest",
    "write_hash_split_stream",
    "write_materialized",
]

__version__ = "0.2.0"
