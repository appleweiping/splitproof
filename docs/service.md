# Split service boundary

`SplitService` exposes deterministic split generation and diagnostics over a
small loopback-first JSON API. The Python server is created with
`create_server()` and accepts `POST /v1/dispatch` requests for `split`,
`kfold`, `temporal_kfold`, `diagnose`, `repeat_holdout`, and `verify`
operations.

```python
from splitproof import create_server

server = create_server(host="127.0.0.1", port=8080)
server.serve_forever()
```

The endpoint reuses the same group-aware and hash-based assigners as the public
library. Bind it behind authentication before exposing it beyond a trusted
machine. In addition to `split` and `diagnose`, the `repeat_holdout`
operation returns group-safe repeated assignments and per-record allocation
rates, using the same deterministic seed and minimum-count controls as the
`repeat-holdout` CLI.

The `kfold` operation exposes the same deterministic group-aware and optional
stratified assignment as the `kfold` CLI. It returns one `fold-<n>` assignment
per input record and accepts `folds`, `seed`, `stratified`, and
`max_local_iterations` request fields.

The `temporal_kfold` operation accepts ISO-8601 `start` and `end` fields from
each record payload and returns purged chronological validation folds. It
supports `gap_seconds`, `embargo_seconds`, `protect_groups`, and custom
`start_field`/`end_field` names, matching the leakage-aware temporal CLI.

The `verify` operation accepts a manifest path plus the same `records` array as
the other operations. It returns a deterministic list of checksum, fingerprint,
coverage, leakage, fold, and optional external-assignment errors.
