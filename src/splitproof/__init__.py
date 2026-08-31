"""SplitProof: reproducible, group-aware dataset partitioning."""

from .assigners import balanced_group_split, hash_split, stratified_group_split
from .diagnostics import diagnose
from .kfold import assign_kfold
from .manifest import create_manifest, load_manifest, save_manifest, verify_manifest
from .models import Assignment, Record, SplitDiagnostics, SplitManifest

__all__ = [
    "Assignment",
    "Record",
    "SplitDiagnostics",
    "SplitManifest",
    "assign_kfold",
    "balanced_group_split",
    "create_manifest",
    "diagnose",
    "hash_split",
    "load_manifest",
    "save_manifest",
    "stratified_group_split",
    "verify_manifest",
]

__version__ = "0.1.0"
