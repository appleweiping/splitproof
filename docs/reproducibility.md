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

The fingerprint sorts `(id, group, label)` tuples before hashing. Therefore file order and JSON
object key order do not matter. Payload fields are outside the split contract. A manifest is for
preventing accidental re-splitting and group/label drift, not for authenticating all dataset
bytes.

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
