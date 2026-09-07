# Cross-split leakage audits

`splitproof leakage-audit` scans application payload fields for normalized exact
duplicates that occur in different split assignments. It is intentionally an
exact, deterministic diagnostic: it does not claim semantic similarity and it
never includes the original payload text in reports. Values are case-folded and
whitespace-collapsed for string fields, then represented by SHA-256 digests.

```bash
splitproof leakage-audit records.jsonl \
  --assignments assignments.jsonl \
  --fields text,question \
  --output leakage.json
```

Exit status is `1` when cross-split pairs are found and `0` when the scan is
clean. `--max-findings` bounds report size for duplicated corpora. Missing
assignments and fields are skipped so the audit can be used on partial exports;
use `splitproof verify` first when complete coverage is required.

The Python API is `audit_leakage(records, assignments, fields=...)`. The report
contains record IDs, split names, field names, and value digests, but no source
text, which makes it suitable for CI artifacts and review.
