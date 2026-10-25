"""splitproof: train/test splits for text data that keep near-duplicates together."""

from .assign import assign_groups
from .audit import LeakageReport, leakage_report
from .neardup import find_cross_duplicates, near_duplicate_groups, normalize_text
from .splitters import NearDupKFold, assign_splits, merge_groups

__all__ = [
    "LeakageReport",
    "NearDupKFold",
    "assign_groups",
    "assign_splits",
    "find_cross_duplicates",
    "leakage_report",
    "merge_groups",
    "near_duplicate_groups",
    "normalize_text",
]

__version__ = "0.3.0"
