from __future__ import annotations

import json
import threading
import urllib.request

from splitproof import SplitService, create_server


def test_split_service_dispatch_and_http() -> None:
    request = {
        "operation": "split",
        "records": [{"id": "a"}, {"id": "b"}, {"id": "c"}, {"id": "d"}],
        "ratios": {"train": 0.5, "test": 0.5},
        "algorithm": "hash",
        "seed": "service",
    }
    result = SplitService().dispatch(request)
    assert len(result["assignments"]) == 4
    server = create_server()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        http_request = urllib.request.Request(
            f"http://127.0.0.1:{server.server_port}/v1/dispatch",
            data=json.dumps(request).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(http_request, timeout=5) as response:
            payload = json.loads(response.read())
        assert payload["assignments"] == result["assignments"]
    finally:
        server.shutdown()
        server.server_close()


def test_split_service_diagnoses_and_rejects_algorithm() -> None:
    service = SplitService()
    assignments = service.dispatch(
        {
            "operation": "split",
            "records": [{"id": "a"}, {"id": "b"}],
            "ratios": {"train": 0.5, "test": 0.5},
            "algorithm": "hash",
        }
    )["assignments"]
    report = service.dispatch(
        {
            "operation": "diagnose",
            "records": [{"id": "a"}, {"id": "b"}],
            "assignments": assignments,
            "ratios": {"train": 0.5, "test": 0.5},
        }
    )
    assert "diagnostics" in report
    try:
        service.dispatch(
            {"operation": "split", "records": [{"id": "a"}], "ratios": {"x": 1}, "algorithm": "bad"}
        )
    except ValueError as error:
        assert "algorithm" in str(error)
    else:  # pragma: no cover
        raise AssertionError("invalid algorithm was accepted")
