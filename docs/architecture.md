# Architecture

SplitProof separates persistent data structures, deterministic decisions, and presentation.

## Modules

- `models.py` defines dataclasses with frozen core fields. Record payloads are shallow-copied,
  while payload, diagnostics, manifest ratio, and metadata mappings remain intentionally mutable.
- `hashing.py` owns canonical encoding and all persistent hashing.
- `constraints.py` validates IDs, ratios, labels, positive record weights, consistent group
  weights, and explicit minimum counts.
- `scoring.py` defines the shared scale-free balance objective.
- `assigners.py` implements record-hash and greedy-plus-local group split algorithms.
- `kfold.py` applies the same weighted objective to cross-validation folds.
- `diagnostics.py` measures coverage, count/weight/label drift, objective contributions, and group
  leakage independently of assignment generation.
- `manifest.py` fingerprints data and checksums every semantic manifest field.
- `io.py`, `reporting.py`, and `cli.py` form the boundary with files and users.

The assignment core accepts typed `Record` objects and has no filesystem or clock dependency.
Only manifest creation records a timestamp; timestamps never participate in assignment.

## Invariants

Record IDs are globally unique. Record and group weights are finite and strictly positive. All
explicit `group_weight` values for one named group agree; one effective group weight is counted
per indivisible group. A non-null group is indivisible in group-aware operations. Every input
record must appear exactly once in a manifest, and no unknown ID may appear. Stratified operations
reject records without labels, while each record may carry any nonempty set of distinct labels.
Persistent hashes have an algorithm version, seed, and purpose-specific domain.

Record-level hash splitting intentionally does not preserve groups, so its manifests record but
do not reject group leakage. Every other built-in group-aware algorithm treats leakage as a
verification failure. Assignment split names must be declared by the manifest ratios.

## Weighted objective and optimization

For every split, the optimizer measures squared error from the requested ratio after normalizing
by the corpus total. The additive objective gives equal weight to record count, total record
weight, and effective group weight. In stratified mode, twice the mean per-label record-weight
error is added. Because each term is scale-free, a large numeric weight does not erase the count
or label signals. `diagnose()` reports every contribution separately.

Groups are ordered deterministically by decreasing weighted label rarity, effective group weight,
record weight, and size. The greedy phase places each group at the destination with the lowest
whole-assignment objective. It guarantees nonempty destinations whenever at least one indivisible
group exists per requested split. A versioned stable digest resolves equal scores.

The local phase evaluates deterministic single-group moves. When there are at most 32 groups it
also evaluates pair swaps. Only a candidate whose objective is lower by more than `1e-15` is
accepted; the best score wins and a stable digest breaks ties. Search stops when no candidate
improves the score or after `min(200, 2 * number_of_groups)` iterations by default. Consequently,
the procedure terminates, never makes the greedy result worse, and is independent of input and
ratio mapping order. `max_local_iterations=0` provides a reproducible greedy-only baseline.

This design is intentionally testable and explainable. Altering its ordering, objective, or tie
break changes assignment semantics and therefore requires a new algorithm version.

## Complexity

Let `G` be the number of indivisible groups, `S` the number of splits, `L` the total number of
group/label incidences, and `I` the local-iteration limit. One complete objective evaluation costs
`O(G + L)`. Because v3 deliberately recomputes that explainable objective for each candidate,
greedy placement costs `O(G * S * (G + L))` and single-move search costs
`O(I * G * S * (G + L))`.

For `G <= 32`, one iteration additionally considers at most `G * (G - 1) / 2` pair swaps, for
`O(I * G^2 * (G + L))` work. At `G = 33` and above that branch is not entered. The default
`I = min(200, 2 * G)` makes the search finite; working memory is `O(G + S + L)`. The v0.2
implementation is a deterministic heuristic, not a claim of global optimality; an opt-in exact
solver remains a possible future complement for small instances.
