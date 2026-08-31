# Migrating schema-v1 manifests

Schema v1 remains readable and verifiable. Its fingerprint covers only record ID, group, and the
legacy scalar label, so successful legacy verification says nothing about v0.2 multi-label or
weight fields. Choose one of the following explicit migrations.

## Preserve existing assignments

Use this route when a published evaluation split must not move.

1. Verify the schema-v1 manifest against the original normalized dataset.
2. Load the legacy assignments from that verified manifest.
3. Review and add normalized label arrays, record weights, and group weights.
4. Call `create_manifest()` with the unchanged assignments, the legacy algorithm name/version,
   and the reviewed v0.2 records.
5. Save the new manifest and verify it together with the external assignments file.

The resulting schema-v2 checksum and fingerprint protect the new fields while assignment IDs and
destinations remain byte-for-byte unchanged. Keep the v1 manifest as provenance; do not overwrite
it in place.

## Adopt assignment algorithm v3

Use this route when improved weighted and multi-label balance justifies a new split release.

1. Keep the verified v1 manifest and assignments as the baseline.
2. Run `stratified_group_split()` or the CLI with the reviewed v0.2 records, ratios, and seed.
3. Compare old and new assignments and archive both diagnostic reports.
4. Publish the v3 assignments under a new split/release identifier.
5. Verify the generated schema-v2 manifest and downstream external assignment file before use.

Algorithm v3 may move groups because its objective and local search differ from historical
versions. A schema migration must therefore never be presented as a no-op algorithm upgrade.
