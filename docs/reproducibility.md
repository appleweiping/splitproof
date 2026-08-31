# Reproducibility contract

## Canonical keys and hashes

Keys are encoded as compact UTF-8 JSON arrays. This avoids delimiter collisions: `['ab', 'c']`
cannot be confused with `['a', 'bc']`. SplitProof uses a 128-bit BLAKE2b digest with fixed
`splitproof-v1` personalization. Both the seed and an untruncated purpose domain are part of the
encoded key, separating record decisions, group ordering, fold choices, dataset fingerprints,
and manifest checksums.

`HASH_VERSION` describes this encoding contract. Changing the digest size, canonical encoding,
seed placement, or domains requires incrementing it. Python's built-in `hash()` is deliberately
never used for persisted results because it is randomized between processes.

Record-hash assignment takes the high 53 digest bits and divides by `2**53`, producing an exact
binary64 value in `[0, 1)` without a rounded `1.0` endpoint. Ratio intervals are constructed in
sorted split-name order. Changes to either rule require a new assignment algorithm version.

## Dataset fingerprint

Schema v2 fingerprints sorted `(id, group, normalized label set, record weight, group weight)`
records. Therefore file order, label-array order, and JSON object key order do not matter. Any
change that can affect a v3 group assignment or its balance evidence changes the fingerprint.
Record-hash destinations ignore weights, but schema-v2 hash manifests still protect their weight
diagnostics. Payload fields remain outside the split contract. A manifest prevents accidental
re-splitting and split-input drift; it does not authenticate all dataset bytes.

## Manifest checksum

The checksum covers every recognized manifest field except itself, serialized as canonical JSON
with sorted object keys. Unknown fields are rejected rather than silently omitted. It detects
accidental edits but is not a digital signature: someone who modifies a
manifest can calculate a new checksum. For adversarial integrity, sign the manifest with your
organization's artifact-signing system.

## Versioning policy

An implementation correction that can move a record increments `algorithm_version`. Older
versions remain readable and verifiable even if they are no longer offered for new assignment.
Schema changes increment `schema_version` and need an explicit migration path.

SplitProof v0.2 writes schema `2`, assignment algorithm `3`, and fingerprint version `2` in
reserved manifest metadata. Schema-v1 manifests remain structurally readable and are verified
with their original `(id, group, scalar label)` fingerprint. Their historical contract cannot
detect changes to v2-only labels or weights; migrate by regenerating a schema-v2 manifest from
the reviewed legacy assignments. Unsupported future schemas fail verification rather than being
silently interpreted under v2 rules.
The two supported migration paths are detailed in [Migrating schema-v1 manifests](migration-v2.md).

Verification treats malformed in-memory manifests like malformed files: invalid ratios are
reported without feeding them into diagnostics, and non-finite or otherwise unserializable values
are reported as unchecksummable. This does not weaken strict file parsing; persisted JSON still
rejects such values before a manifest object is created.

## Local-search determinism

The v3 greedy and local phases use the objective and termination rules documented in
[Architecture](architecture.md). Candidate enumeration is sorted, equal scores use domain-separated
BLAKE2 digests, and only strict improvements are applied. These details, including the 32-group
pair-swap boundary and automatic iteration limit, are part of algorithm version 3. Changing them
requires a new algorithm version and benchmark artifact.

The fixed comparison in `benchmarks/results-v0.2.json` contains no clock measurements or machine
metadata. Regeneration is therefore byte-stable on supported Python versions. The seeded random
baseline is for quality context only; persistent product decisions use SplitProof's versioned
hashing and optimizer paths.
