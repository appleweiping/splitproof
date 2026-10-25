# Changelog

## 0.3.0 (unreleased)

Rewritten around one job: keeping near-duplicate texts on the same side of a split.

- `NearDupKFold`, a scikit-learn compatible splitter (`split`, `get_n_splits`) that groups
  near-duplicate texts with MinHash + LSH and assigns whole groups to folds while balancing
  single- or multi-label distributions. scikit-learn is optional.
- `assign_splits` for named train/validation/test splits, `leakage_report` for auditing an
  existing split.
- CLI: `splitproof audit` and `splitproof split` for JSONL, JSON, CSV/TSV and Parquet files.
- Benchmarks on SMS Spam, dair-ai/emotion and GoEmotions (`benchmarks/`).
- Removed the checksummed manifest, weights and the hash/greedy split modes of 0.2. They are
  still in the git history under the `v0.2.0` tag.

## 0.2.0 (2026-08-31)

Weighted multi-label stratification, k-fold assignment and checksummed split manifests.

## 0.1.0 (2026-08-31)

First version: hash, group and stratified group splits with a JSON manifest.
