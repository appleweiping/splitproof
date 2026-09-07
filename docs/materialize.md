# Materializing verified splits

`splitproof materialize` turns an authenticated manifest into one deterministic
JSONL file per split:

```text
splitproof materialize data.jsonl --manifest split.json \
  --output-dir materialized
```

The command rechecks the dataset fingerprint, manifest checksum, assignment
coverage, and group constraints before writing. Each output contains the
original record payload in source order; split membership stays in the manifest
so downstream jobs can join it by the record ID. Existing files are never
overwritten, and split names that could escape the output directory are rejected.
