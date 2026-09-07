# Exact group optimization

`exact_group_split` is an opt-in exhaustive solver for small grouped datasets.
It enumerates every assignment of indivisible groups, uses the same count,
weight, and optional label-balance objective as the normal group solver, and
chooses equal-score results by a stable digest. `max_groups` defaults to 12;
larger inputs fail instead of silently falling back to a heuristic.

The CLI exposes this as:

```console
splitproof split records.jsonl --algorithm exact-group --max-groups 12 \
  --ratios train=0.8,test=0.2 --assignments assignments.jsonl --manifest split.json
```

The HTTP service accepts `{"operation":"split","algorithm":"exact",...}`.
This mode is intended for audit fixtures and small benchmark studies, not
large production corpora.
