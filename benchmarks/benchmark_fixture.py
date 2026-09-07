"""Benchmark split algorithms on the checked-in support-message fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
import tracemalloc
from pathlib import Path

from splitproof import (
    evaluate_nested,
    holdout_stability_report,
    nested_group_kfold,
    repeated_group_holdout,
    repeated_kfold,
    stability_report,
)
from splitproof.io import load_records


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    source = root / "examples" / "support_messages.jsonl"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=root / "benchmarks/results/fixture.json")
    args = parser.parse_args()
    payload = source.read_bytes()
    records = load_records(source)
    tracemalloc.start()
    started = time.perf_counter()
    folds = repeated_kfold(records, folds=2, repeats=3, seed="fixture", stratified=True)
    fold_report = stability_report(folds)
    holdouts = repeated_group_holdout(records, {"train": 0.75, "test": 0.25}, 3, seed="fixture")
    holdout_report = holdout_stability_report(holdouts)
    nested = nested_group_kfold(records, outer_folds=2, inner_folds=2, seed="fixture")
    nested_report = evaluate_nested(
        records,
        nested,
        lambda train, test: len(train) / len(test),
    )
    elapsed = time.perf_counter() - started
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    result = {
        "kind": "fixture-real",
        "source": str(source.relative_to(root)).replace("\\", "/"),
        "source_sha256": hashlib.sha256(payload).hexdigest(),
        "source_bytes": len(payload),
        "records": len(records),
        "kfold_repeats": fold_report.repeats,
        "kfold_agreement": fold_report.pairwise_agreement,
        "holdout_repeats": holdout_report.repeats,
        "holdout_splits": list(holdout_report.splits),
        "nested_outer_folds": len(nested_report.successful),
        "nested_outer_mean": nested_report.mean_score,
        "elapsed_seconds": elapsed,
        "peak_python_bytes": peak,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
