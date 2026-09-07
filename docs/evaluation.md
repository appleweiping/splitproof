# Leakage-aware evaluation

`evaluate_repeated_kfold()` connects SplitProof assignments to a model or
metric callback without hiding split failures. The callback receives immutable
`(train_records, test_records)` tuples for every repetition/fold. SplitProof
checks complete coverage, non-empty partitions, and group isolation before the
callback runs.

```python
from splitproof import evaluate_repeated_kfold, repeated_kfold

assignments = repeated_kfold(records, folds=5, repeats=3, seed="release-1")
report = evaluate_repeated_kfold(
    records,
    assignments,
    lambda train, test: evaluate_model(train, test),
)
print(report.mean_score, report.score_stddev)
```

Use `strict=False` when producing an exploratory report that must retain failed
folds; production CI should keep the default `strict=True` so exceptions cannot
be mistaken for a score.
