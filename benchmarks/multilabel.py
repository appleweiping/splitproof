"""Compare 5-fold splitters on a multi-label text dataset (GoEmotions).

Data: the train split of google-research-datasets/go_emotions ("simplified"
config): 43,410 Reddit comments, 28 labels, 1-3 labels per comment.

Splitters:

- ``KFold(shuffle=True)`` from scikit-learn, the usual default;
- ``MultilabelStratifiedKFold`` from iterative-stratification;
- ``StratifiedGroupKFold`` from scikit-learn, given splitproof's near-duplicate
  groups and each comment's first label (it only accepts one label per row);
- ``NearDupKFold`` from splitproof.

For each, the script reports how far each fold's label shares drift from the
whole dataset, the spread of fold sizes, how many test comments have a
near-duplicate (or an identical comment) in the training folds, and the time
taken. Run from a source checkout:

    pip install -e ".[bench]"
    python benchmarks/multilabel.py
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import hub

from splitproof import NearDupKFold, find_cross_duplicates

FOLDS = 5
THRESHOLD = 0.8


def fold_ids(splits: Iterable[tuple[np.ndarray, np.ndarray]], n_rows: int) -> np.ndarray:
    folds = np.full(n_rows, -1)
    for fold, (_, test) in enumerate(splits):
        folds[test] = fold
    assert (folds >= 0).all()
    return folds


def evaluate(folds: np.ndarray, indicator: np.ndarray, texts: list[str]) -> dict[str, float]:
    overall = indicator.mean(axis=0)
    relative = []
    sizes = []
    leaked = exact = 0
    for fold in range(FOLDS):
        mask = folds == fold
        sizes.append(int(mask.sum()))
        relative.append(np.abs(indicator[mask].mean(axis=0) - overall) / overall)
        train = [texts[i] for i in np.flatnonzero(~mask)]
        test = [texts[i] for i in np.flatnonzero(mask)]
        pairs = find_cross_duplicates(train, test, THRESHOLD)
        leaked += len(pairs)
        exact += sum(1 for pair in pairs if pair.similarity >= 1.0)
    deviation = np.array(relative)
    return {
        "label_share_rel_dev_mean": float(deviation.mean()),
        "label_share_rel_dev_max": float(deviation.max()),
        "fold_size_min": min(sizes),
        "fold_size_max": max(sizes),
        "test_rows_with_near_dup_in_train": leaked,
        "test_rows_with_identical_in_train": exact,
    }


def timed(make: Callable[[], np.ndarray]) -> tuple[np.ndarray, float]:
    started = time.perf_counter()
    result = make()
    return result, time.perf_counter() - started


def main() -> None:
    from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
    from sklearn.model_selection import KFold, StratifiedGroupKFold

    data = hub.columns("go_emotions_train")
    texts: list[str] = data["text"]
    label_lists: list[list[int]] = data["labels"]
    n_rows, n_labels = len(texts), 28
    indicator = np.zeros((n_rows, n_labels), dtype=np.int8)
    for row, labels in enumerate(label_lists):
        indicator[row, labels] = 1
    first_label = np.array([labels[0] for labels in label_lists])
    print(f"go_emotions train: {n_rows} comments, {n_labels} labels")

    ours = NearDupKFold(FOLDS, threshold=THRESHOLD, random_state=0)
    groups, grouping_time = timed(lambda: ours.near_duplicate_groups(texts))
    n_groups = int(groups.max()) + 1
    print(f"near-duplicate grouping: {n_groups} groups, {grouping_time:.1f} s")

    runs: dict[str, tuple[np.ndarray, float]] = {
        "KFold(shuffle=True)": timed(
            lambda: fold_ids(KFold(FOLDS, shuffle=True, random_state=0).split(texts), n_rows)
        ),
        "MultilabelStratifiedKFold": timed(
            lambda: fold_ids(
                MultilabelStratifiedKFold(FOLDS, shuffle=True, random_state=0).split(
                    np.zeros(n_rows), indicator
                ),
                n_rows,
            )
        ),
        "StratifiedGroupKFold (first label)": timed(
            lambda: fold_ids(
                StratifiedGroupKFold(FOLDS, shuffle=True, random_state=0).split(
                    texts, first_label, groups
                ),
                n_rows,
            )
        ),
        "splitproof NearDupKFold": timed(lambda: ours.fold_ids(texts, indicator)),
    }

    results: dict[str, Any] = {
        "rows": n_rows,
        "labels": n_labels,
        "near_dup_groups": n_groups,
        "grouping_seconds": grouping_time,
    }
    print(
        "\n| splitter | mean label-share deviation | worst label-share deviation "
        "| fold sizes | test rows with near-dup in train (identical) | split time |"
    )
    print("|---|---:|---:|---:|---:|---:|")
    for name, (folds, seconds) in runs.items():
        row = evaluate(folds, indicator, texts)
        row["split_seconds"] = seconds
        results[name] = row
        note = (
            f" + {grouping_time:.1f} s grouping"
            if name.startswith(("Stratified", "splitproof"))
            else ""
        )
        print(
            f"| {name} | {row['label_share_rel_dev_mean']:.2%} "
            f"| {row['label_share_rel_dev_max']:.1%} "
            f"| {row['fold_size_min']}-{row['fold_size_max']} "
            f"| {row['test_rows_with_near_dup_in_train']} "
            f"({row['test_rows_with_identical_in_train']}) "
            f"| {seconds:.1f} s{note} |"
        )

    output = Path(__file__).resolve().parent / "results" / "multilabel.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {output.name}")


if __name__ == "__main__":
    main()
