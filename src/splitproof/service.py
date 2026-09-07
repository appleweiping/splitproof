"""Loopback HTTP/JSON integration boundary for reproducible splits."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .assigners import balanced_group_split, hash_split, stratified_group_split
from .diagnostics import diagnose
from .models import Assignment, Record


class SplitService:
    """Dispatch split and diagnostics requests using public SplitProof APIs."""

    def dispatch(self, request: Mapping[str, Any]) -> dict[str, Any]:
        """Execute a JSON-compatible request."""
        if not isinstance(request, Mapping):
            raise ValueError("request must be an object")
        operation = request.get("operation")
        records_value = request.get("records")
        if not isinstance(records_value, list):
            raise ValueError("records must be an array")
        records = tuple(Record.from_mapping(item) for item in records_value)
        if operation == "split":
            ratios = request.get("ratios")
            if not isinstance(ratios, Mapping):
                raise ValueError("ratios must be an object")
            algorithm = request.get("algorithm", "balanced")
            seed = request.get("seed", "0")
            if algorithm == "hash":
                assignments = hash_split(records, ratios, seed=seed)
            elif algorithm == "stratified":
                assignments = stratified_group_split(records, ratios, seed=seed)
            elif algorithm == "balanced":
                assignments = balanced_group_split(records, ratios, seed=seed)
            else:
                raise ValueError("algorithm must be hash, balanced, or stratified")
            return {"operation": operation, "assignments": [asdict(item) for item in assignments]}
        if operation == "diagnose":
            assignments_value = request.get("assignments")
            if not isinstance(assignments_value, list):
                raise ValueError("assignments must be an array")
            assignments = tuple(Assignment(**item) for item in assignments_value)
            ratios = request.get("ratios")
            report = diagnose(records, assignments, ratios if isinstance(ratios, Mapping) else None)
            return {"operation": operation, "diagnostics": asdict(report)}
        raise ValueError("operation must be split or diagnose")


def create_server(
    service: SplitService | None = None,
    *,
    host: str = "127.0.0.1",
    port: int = 0,
) -> ThreadingHTTPServer:
    """Create a threaded loopback-first JSON server."""
    if not isinstance(port, int) or isinstance(port, bool) or not 0 <= port <= 65535:
        raise ValueError("port must be an integer between 0 and 65535")
    target = service or SplitService()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            if self.path != "/v1/dispatch":
                self._write(HTTPStatus.NOT_FOUND, {"error": "unknown endpoint"})
                return
            try:
                size = int(self.headers.get("Content-Length", "-1"))
                if size < 0 or size > 4 * 1024 * 1024:
                    raise ValueError("Content-Length must be between 0 and 4194304")
                response = target.dispatch(json.loads(self.rfile.read(size).decode("utf-8")))
            except (UnicodeError, json.JSONDecodeError, TypeError, ValueError) as error:
                self._write(HTTPStatus.BAD_REQUEST, {"error": str(error)})
                return
            self._write(HTTPStatus.OK, response)

        def log_message(self, format: str, *args: object) -> None:
            return

        def _write(self, status: HTTPStatus, payload: Mapping[str, Any]) -> None:
            encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    return server
