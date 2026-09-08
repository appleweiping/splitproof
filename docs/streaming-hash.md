# Streaming hash assignment

Group-aware and stratified algorithms need a complete group inventory to make
their balance objective meaningful. Record-hash assignment does not: each ID
maps independently to a ratio interval. `hash_split_stream` and the
`hash-stream` CLI expose that distinction explicitly.

## CLI

```console
splitproof hash-stream records.jsonl \
  --ratios train=0.8,test=0.2 --seed release-2026 \
  --assignments assignments.jsonl --report stream-report.json
```

The command accepts JSONL only and reads one record at a time. It validates
record IDs and strict JSON, rejects duplicate IDs, uses the same versioned
BLAKE2b hash interval as `hash_split`, and writes assignments atomically. The
assignment file preserves input order so the writer does not retain all rows;
the report includes split counts and a SHA-256 digest of the exact output
stream. It never emits a manifest because a manifest's dataset fingerprint and
diagnostics require a complete record inventory.

Authenticate the emitted artifact later, even when the source corpus is no
longer available:

```bash
splitproof hash-stream-verify assignments.jsonl stream-report.json
```

`verify_hash_split_stream` hashes exact UTF-8 lines, rejects malformed or
duplicate assignment IDs, and checks the declared counts and digest.

## Python API

```python
from splitproof import hash_split_stream, iter_records, write_hash_split_stream

assignments = hash_split_stream(iter_records("records.jsonl"), {"train": 0.8, "test": 0.2})
for assignment in assignments:
    consume(assignment.record_id, assignment.split)

report = write_hash_split_stream(
    iter_records("records.jsonl"), {"train": 0.8, "test": 0.2}, "assignments.jsonl"
)
```

Use the regular `splitproof split --algorithm hash` path when a checksummed
manifest, minimum-count constraints, or sorted output is required.
