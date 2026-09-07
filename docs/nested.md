# Nested cross-validation

`nested_group_kfold` constructs an outer group-aware k-fold assignment and an
independent inner assignment for each outer training partition. Inner folds are
never built from the outer validation records, preventing model-selection
information from crossing the outer evaluation boundary.

```python
from splitproof import nested_group_kfold

result = nested_group_kfold(records, outer_folds=5, inner_folds=3, seed="release-1")
for outer_fold in range(result.outer_folds):
    train = result.outer_training(outer_fold)
    validation = result.outer_validation(outer_fold)
```

Both levels preserve intact groups and can use the existing multi-label
stratification. Seeds are namespaced by level and outer fold. This returns
assignments, not fitted models or scores; the caller must fit preprocessing and
models inside each outer training partition. A fold count larger than the number
of available groups fails through the existing constraints.

For variance estimates across both levels, `repeated_nested_group_kfold`
namespaces every repetition and exposes `outer_stability()` plus
`inner_stability(outer_fold)`. Inner stability is computed over records common
to that outer fold's training partitions, so changing an outer assignment
cannot silently be mistaken for a model-selection regression.

`evaluate_nested()` executes the generated split: it scores every inner fold,
records the selected fold according to a higher/lower direction, then scores the
outer holdout without mixing records. With `strict=False`, evaluator failures
remain attached to the affected outer fold so a report can distinguish an
incomplete run from a clean low score.

For actual model selection, `evaluate_nested_candidates()` accepts a mapping of
candidate names to callbacks. Each callback receives its candidate name and
immutable train/test tuples. The callback is run on every inner fold, the best
candidate mean is selected with deterministic name tie-breaking, and only that
candidate is evaluated on the untouched outer holdout:

```python
from splitproof import evaluate_nested_candidates

report = evaluate_nested_candidates(
    records,
    result,
    {"linear": score_linear, "tree": score_tree},
    direction="higher",
)
print(report.selection_counts, report.mean_score)
```

The report retains per-fold candidate means, selected names, and failures. This
keeps preprocessing/model selection inside the outer training partition instead
of accidentally tuning against the final holdout.
