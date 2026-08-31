# Architecture

SplitProof separates persistent data structures, deterministic decisions, and presentation.

## Modules

- `models.py` defines dataclasses with frozen core fields. Record payloads are shallow-copied,
  while payload, diagnostics, manifest ratio, and metadata mappings remain intentionally mutable.
- `hashing.py` owns canonical encoding and all persistent hashing.
- `constraints.py` validates IDs, ratios, labels, and explicit minimum counts.
- `assigners.py` implements record-hash and greedy group split algorithms.
- `kfold.py` applies the group objective to cross-validation folds.
- `diagnostics.py` measures coverage, distribution drift, and group leakage independently of
  assignment generation.
- `manifest.py` fingerprints data and checksums every semantic manifest field.
- `io.py`, `reporting.py`, and `cli.py` form the boundary with files and users.

The assignment core accepts typed `Record` objects and has no filesystem or clock dependency.
Only manifest creation records a timestamp; timestamps never participate in assignment.

## Invariants

Record IDs are globally unique. A non-null group is indivisible in group-aware operations.
Every input record must appear exactly once in a manifest, and no unknown ID may appear.
Stratified operations reject missing labels. Persistent hashes have an algorithm version, seed,
and purpose-specific domain.

Record-level hash splitting intentionally does not preserve groups, so its manifests record but
do not reject group leakage. Every other built-in group-aware algorithm treats leakage as a
verification failure. Assignment split names must be declared by the manifest ratios.

## Greedy group allocation

Groups are ordered deterministically by decreasing size. Stratified allocation first considers
the fraction of each label concentrated in a group, which schedules hard-to-place groups early.
For every candidate destination, the objective estimates the change in size deviation and,
when enabled, normalized label deviation. A stable digest resolves exact ties.

This design is intentionally testable and explainable. Altering its ordering, objective, or tie
break changes assignment semantics and therefore requires a new algorithm version.
