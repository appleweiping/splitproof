# Purged temporal validation

Group-aware random splitting is insufficient when a record's features or labels
cover a time interval. Overlapping information can leak across a validation
boundary even if record IDs and groups are disjoint.

```python
from datetime import datetime, timedelta, timezone
from splitproof import Record, TimeInterval, purged_holdout

origin = datetime(2026, 1, 1, tzinfo=timezone.utc)
records = [Record(str(i)) for i in range(10)]
intervals = {
    str(i): TimeInterval(origin + timedelta(days=i), origin + timedelta(days=i)) for i in range(10)
}
fold = purged_holdout(records, intervals, ["7", "8"], past_only=True)
assert fold.train == ("0", "1", "2", "3", "4", "5", "6")
```

An interval must cover **all information used**: feature lookback as well as
label lookahead. An observation timestamp alone cannot prove absence of leakage.
Timezone-aware endpoints are normalized to UTC. Intervals are closed: sharing
an endpoint counts as overlap. Zero-width point intervals are allowed.

For every holdout, the algorithm expands validation intervals by a nonnegative
gap before and embargo after, merges overlapping expansions, and removes any
training candidate overlapping that union. It does not remove records in the
gaps between disjoint validation intervals. A binary search over merged intervals
avoids a pairwise training-by-validation matrix.

By default, any candidate sharing a non-null group with a validation record is
also excluded. `None` does not make all ungrouped records one group. Past-only
mode additionally requires the entire training interval to end strictly before
the earliest expanded validation start. Every excluded ID includes its reasons;
exclusions must never be silently treated as training records.

## Chronological cross-validation

`purged_kfold(records, intervals, folds)` partitions distinct start timestamps
into chronological validation blocks. Tied starts remain together. Every record
is validated once, but group membership may span validation blocks: protection
is enforced between train and validation within each fold. Block sizes balance
distinct timestamps, not records or labels. This is not a stratified or
forward-chaining splitter. Use `purged_holdout(..., past_only=True)` for an
explicit forward evaluation window.

```console
splitproof temporal-kfold data.jsonl --folds 5 --start-field start --end-field end --gap-seconds 60 --embargo-seconds 120 --output folds.json
```

Input fields are ISO timestamp strings with timezones. Output is a versioned
temporal-fold report with train, validation, and excluded IDs, not the older
single-assignment manifest format. The existing `verify` command does not verify
this report. Re-run the deterministic temporal algorithm against the original
dataset to reproduce it. Empty training sets and invalid intervals fail before
output is written. This feature does not establish whole-repository parity.
