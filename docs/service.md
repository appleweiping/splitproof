# Split service boundary

`SplitService` exposes deterministic split generation and diagnostics over a
small loopback-first JSON API. The Python server is created with
`create_server()` and accepts `POST /v1/dispatch` requests for `split` and
`diagnose` operations.

```python
from splitproof import create_server

server = create_server(host="127.0.0.1", port=8080)
server.serve_forever()
```

The endpoint reuses the same group-aware and hash-based assigners as the public
library. Bind it behind authentication before exposing it beyond a trusted
machine.
