"""How much of a test split already sits in the training split?

1. SMS Spam Collection (ucirvine/sms_spam, 5,574 messages): random 80/20 split
   versus a splitproof split, five seeds each.
2. The same data under 5-fold cross-validation with a TF-IDF + logistic
   regression classifier: shuffled KFold versus NearDupKFold.
3. The official train/test split of dair-ai/emotion.

Run from a source checkout:

    pip install -e ".[bench]"
    python benchmarks/leakage.py

Results are printed and written to benchmarks/results/leakage.json.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import hub

from splitproof import NearDupKFold, assign_splits, leakage_report

THRESHOLD = 0.8
SEEDS = range(5)


def leak(train: list[str], test: list[str]) -> tuple[float, float]:
    report = leakage_report(train, test, THRESHOLD)
    return report.fraction, report.exact / report.n_test


def split_comparison(texts: list[str], labels: list[int]) -> dict[str, dict[str, float]]:
    results: dict[str, list[tuple[float, float]]] = {"random": [], "splitproof": []}
    for seed in SEEDS:
        order = np.random.RandomState(seed).permutation(len(texts))
        cut = int(0.8 * len(texts))
        train = [texts[i] for i in order[:cut]]
        test = [texts[i] for i in order[cut:]]
        results["random"].append(leak(train, test))
        names = assign_splits(texts, labels, threshold=THRESHOLD, random_state=seed)
        train = [text for text, name in zip(texts, names) if name == "train"]
        test = [text for text, name in zip(texts, names) if name == "test"]
        results["splitproof"].append(leak(train, test))
    return {
        method: {
            "near_dup_leak_mean": float(np.mean([near for near, _ in values])),
            "near_dup_leak_min": float(np.min([near for near, _ in values])),
            "near_dup_leak_max": float(np.max([near for near, _ in values])),
            "exact_leak_mean": float(np.mean([exact for _, exact in values])),
        }
        for method, values in results.items()
    }


def cross_validation(texts: list[str], labels: list[int]) -> dict[str, dict[str, float]]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import KFold, cross_validate
    from sklearn.pipeline import make_pipeline

    model = make_pipeline(
        TfidfVectorizer(sublinear_tf=True, min_df=2), LogisticRegression(C=10, max_iter=2000)
    )
    splitters = {
        "KFold(shuffle=True)": KFold(5, shuffle=True, random_state=0),
        "NearDupKFold": NearDupKFold(5, threshold=THRESHOLD),
    }
    results = {}
    for name, cv in splitters.items():
        scores = cross_validate(model, texts, labels, cv=cv, scoring=["f1", "accuracy"])
        results[name] = {
            "spam_f1_mean": float(scores["test_f1"].mean()),
            "spam_f1_std": float(scores["test_f1"].std()),
            "accuracy_mean": float(scores["test_accuracy"].mean()),
        }
    return results


def main() -> None:
    started = time.perf_counter()
    sms = hub.columns("sms_spam")
    texts, labels = sms["sms"], sms["label"]
    exact_unique = len({" ".join(text.lower().split()) for text in texts})
    print(f"sms_spam: {len(texts)} messages, {exact_unique} distinct after lowercasing/whitespace")

    splits = split_comparison(texts, labels)
    print("\n80/20 split, 5 seeds: share of test messages with a near-duplicate in train")
    print("| split | near-duplicate (Jaccard >= 0.8) | identical after normalization |")
    print("|---|---:|---:|")
    for method, row in splits.items():
        print(
            f"| {method} | {row['near_dup_leak_mean']:.1%} "
            f"({row['near_dup_leak_min']:.1%}-{row['near_dup_leak_max']:.1%}) "
            f"| {row['exact_leak_mean']:.1%} |"
        )

    scores = cross_validation(texts, labels)
    print("\n5-fold CV, TF-IDF + logistic regression")
    print("| cv | spam F1 | accuracy |")
    print("|---|---:|---:|")
    for name, row in scores.items():
        print(
            f"| {name} | {row['spam_f1_mean']:.4f} +/- {row['spam_f1_std']:.4f} "
            f"| {row['accuracy_mean']:.4f} |"
        )

    train = hub.columns("emotion_train")
    test = hub.columns("emotion_test")
    report = leakage_report(train["text"], test["text"], THRESHOLD)
    conflicts = sum(
        1
        for pair in report.pairs
        if pair.similarity >= 1.0 and train["label"][pair.left] != test["label"][pair.right]
    )
    print(
        f"\ndair-ai/emotion official split: {report.leaked} of {report.n_test} test rows "
        f"({report.fraction:.1%}) have a near-duplicate in train, {report.exact} identical; "
        f"{conflicts} identical pairs carry different labels"
    )

    output = Path(__file__).resolve().parent / "results" / "leakage.json"
    output.parent.mkdir(exist_ok=True)
    summary = {
        "threshold": THRESHOLD,
        "sms_spam_rows": len(texts),
        "split_80_20": splits,
        "cross_validation": scores,
        "emotion_official": {
            "n_train": report.n_train,
            "n_test": report.n_test,
            "leaked": report.leaked,
            "exact": report.exact,
            "exact_with_different_label": conflicts,
        },
    }
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(
        f"\nwrote {output.relative_to(Path.cwd()) if output.is_relative_to(Path.cwd()) else output}"
    )
    print(f"total time {time.perf_counter() - started:.0f} s")


if __name__ == "__main__":
    main()
