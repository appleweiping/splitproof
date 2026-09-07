# Repeated cross-validation

`repeated_kfold` creates independently seeded, group-safe k-fold assignments for
variance analysis. The seed derivation is domain-separated, so the first
repetition is stable when the requested number of repetitions changes.

`stability_report` summarizes pairwise assignment agreement, normalized per-row
fold entropy, and fold counts. It accepts manifests or in-memory assignments and
does not need access to payload data.

```python
from splitproof import repeated_kfold, stability_report

repetitions = repeated_kfold(records, folds=5, repeats=10, seed="experiment-7")
report = stability_report(repetitions)
print(report.pairwise_agreement, report.mean_entropy)
```
