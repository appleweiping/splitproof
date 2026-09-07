# Manifest comparison

`compare_manifests()` audits drift between two persisted split decisions. It
keeps dataset fingerprint changes separate from assignment changes and reports
added/removed record IDs, moved records (including fold changes), and directed
split transition counts.

```bash
splitproof compare before.manifest.json after.manifest.json \
  --output split-drift.json
```

The command exits `0` when no drift is found and `1` when the reports differ.
The JSON result is deterministic and does not infer that a changed assignment
is a regression; it is evidence for a reviewer to interpret alongside the
dataset and experiment metadata.
