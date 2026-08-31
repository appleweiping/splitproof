# Changelog

All notable changes are documented here. The format follows Keep a Changelog.

## [Unreleased]

## [0.2.0] - 2026-08-31

### Added

- Positive record weights and consistent, once-per-group effective weights.
- Multi-label weighted stratification for group splits and k-fold assignment.
- A shared, explainable balance objective covering counts, record weights, group weights, and
  per-label weights.
- Bounded deterministic local improvement after greedy placement, including pair swaps for at
  most 32 groups and a greedy-only comparison mode.
- Count/weight/label deviation and objective-component diagnostics in JSON and Markdown.
- A reproducible quality benchmark comparing seeded random groups, record hashing, the released
  v2 greedy method, v3 greedy-only placement, and the complete v3 optimizer.
- Maintainer boundary coverage for exact local-search monotonicity and the 32-group swap cutoff,
  scale-free weights, extreme ratios, empty labels, normalized duplicate IDs, schema-v1 disk
  migration, and hard-linked CLI paths.

### Changed

- New manifests use schema 2, fingerprint version 2, and assignment algorithm version 3.
- Schema-v2 fingerprints cover normalized label sets and both weight types.
- Schema-v1 manifests remain readable and verifiable under their historical fingerprint contract.
- CLI field mapping now includes `--weight-field` and `--group-weight-field`.

### Fixed

- `diagnose()` rejects invalid expected ratios instead of producing a non-finite objective.
- `verify_manifest()` reports malformed in-memory ratios or unchecksummable values instead of
  raising while it is collecting verification errors.

## [0.1.0] - 2026-08-31

### Added

- Stable hash, balanced group, and stratified group split algorithms.
- Mapping-order-independent, strictly half-open record-hash ratio intervals.
- Group-aware stratified and non-stratified k-fold assignment.
- Strict JSON and checksummed manifests with data fingerprint, schema, and coverage verification.
- Filesystem path-collision checks that protect source data and generated evidence from overwrite.
- Optional external-assignment verification and integrity-protected manifest inspection.
- Algorithm-appropriate group-leakage diagnostics and verification semantics.
- JSON/JSONL CLI workflows and JSON/Markdown diagnostics.
