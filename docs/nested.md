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
