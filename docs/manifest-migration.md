# Manifest schema migration

SplitProof can migrate a schema-v1 manifest to schema v2 without changing its
assignments:

```console
splitproof migrate-manifest records.jsonl \
  --manifest legacy-split.json \
  --output split-v2.json
```

Migration is data-bound and refuses to operate on an unverified manifest. The
source records are re-read, the v1 fingerprint is checked, and then the v2
fingerprint (which includes normalized multi-label and weight fields) and
checksum are regenerated. The algorithm, seed, ratios, assignments, creation
timestamp, and user metadata remain unchanged; versioned hash metadata is
updated explicitly.

The same operation is available through `SplitService` as
`{"operation":"migrate_manifest", "records":[...], "manifest":"..."}`.
