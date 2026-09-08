# Payload-aware dataset fingerprints

SplitProof fingerprints the normalized fields that affect split decisions by
default. Applications that also need to detect content drift can opt into
top-level payload fields when creating a manifest:

```python
manifest = create_manifest(
    records,
    assignments,
    algorithm="group",
    algorithm_version="3",
    seed="release-1",
    fingerprint_fields=("text", "source_id"),
)
```

The selected fields are recorded in the reserved `metadata.fingerprint_fields`
list, sorted deterministically, and included in both manifest creation and
verification. A missing payload field contributes JSON `null`, so adding or
removing a field is visible rather than silently ignored. Core split fields
(`id`, `group`, `label`, `labels`, `weight`, and `group_weight`) cannot be
listed because they are already part of the versioned v2 fingerprint.

The CLI exposes the same contract with repeated options:

```bash
splitproof split data.jsonl --algorithm group \
  --fingerprint-field text --fingerprint-field source_id \
  --assignments assignments.jsonl --manifest manifest.json
```

This does not hash binary files or dereference URLs. It fingerprints the JSON
value present in each record payload; external artifacts should be represented
by a stable digest field in the dataset before invoking SplitProof.
