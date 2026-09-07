# Near-duplicate leakage audits

Exact duplicate scans miss small edits, punctuation changes, and copied context
with one changed answer. `near-duplicate-audit` adds a deterministic review
pass using token trigrams and Jaccard similarity:

```bash
splitproof near-duplicate-audit records.jsonl \
  --assignments assignments.jsonl --field text \
  --threshold 0.80 --output near-duplicates.json
```

Only records sharing a shingle are compared. Reports contain IDs, split names,
and similarity scores, never source text. `--max-pairs` bounds pathological
duplicate corpora. This is an exact-token diagnostic, not a semantic model;
review findings before changing a split.
