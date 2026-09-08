"""Loopback HTTP/JSON integration boundary for reproducible splits."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict
from datetime import datetime, timedelta
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .assigners import balanced_group_split, exact_group_split, hash_split, stratified_group_split
from .diagnostics import diagnose
from .kfold import assign_kfold
from .manifest import load_manifest, migrate_manifest
from .models import Assignment, Record
from .repeat import holdout_stability_report, repeated_group_holdout
from .temporal import TimeInterval, purged_kfold


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
            elif algorithm == "exact":
                max_groups = request.get("max_groups", 12)
                if isinstance(max_groups, bool) or not isinstance(max_groups, int):
                    raise ValueError("max_groups must be an integer")
                stratified = request.get("stratified", False)
                if not isinstance(stratified, bool):
                    raise ValueError("stratified must be a boolean")
                assignments = exact_group_split(
                    records,
                    ratios,
                    seed=seed,
                    stratified=stratified,
                    max_groups=max_groups,
                )
            elif algorithm == "balanced":
                assignments = balanced_group_split(records, ratios, seed=seed)
            else:
                raise ValueError("algorithm must be hash, balanced, stratified, or exact")
            return {"operation": operation, "assignments": [asdict(item) for item in assignments]}
        if operation == "kfold":
            folds = request.get("folds", 5)
            if isinstance(folds, bool) or not isinstance(folds, int):
                raise ValueError("folds must be an integer")
            seed = request.get("seed", "0")
            if not isinstance(seed, (str, int)) or isinstance(seed, bool):
                raise ValueError("seed must be a string or integer")
            stratified = request.get("stratified", False)
            if not isinstance(stratified, bool):
                raise ValueError("stratified must be a boolean")
            max_local_iterations = request.get("max_local_iterations")
            if max_local_iterations is not None and (
                isinstance(max_local_iterations, bool) or not isinstance(max_local_iterations, int)
            ):
                raise ValueError("max_local_iterations must be an integer or omitted")
            assignments = assign_kfold(
                records,
                folds,
                seed=seed,
                stratified=stratified,
                max_local_iterations=max_local_iterations,
            )
            return {
                "operation": operation,
                "folds": folds,
                "assignments": [asdict(item) for item in assignments],
            }
        if operation == "temporal_kfold":
            start_field = request.get("start_field", "start")
            end_field = request.get("end_field", "end")
            if not isinstance(start_field, str) or not start_field.strip():
                raise ValueError("start_field must be a non-empty string")
            if not isinstance(end_field, str) or not end_field.strip():
                raise ValueError("end_field must be a non-empty string")
            folds = request.get("folds", 5)
            if isinstance(folds, bool) or not isinstance(folds, int):
                raise ValueError("folds must be an integer")
            gap_seconds = request.get("gap_seconds", 0)
            embargo_seconds = request.get("embargo_seconds", 0)
            for name, value in (("gap_seconds", gap_seconds), ("embargo_seconds", embargo_seconds)):
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise ValueError(f"{name} must be a non-negative integer")
            protect_groups = request.get("protect_groups", True)
            if not isinstance(protect_groups, bool):
                raise ValueError("protect_groups must be a boolean")
            intervals: dict[str, TimeInterval] = {}
            for record in records:
                start_value = record.payload.get(start_field)
                end_value = record.payload.get(end_field)
                if not isinstance(start_value, str) or not isinstance(end_value, str):
                    raise ValueError(
                        f"record {record.id!r} requires ISO timestamp strings for start and end"
                    )
                try:
                    start = datetime.fromisoformat(start_value.replace("Z", "+00:00"))
                    end = datetime.fromisoformat(end_value.replace("Z", "+00:00"))
                    intervals[record.id] = TimeInterval(start, end)
                except (TypeError, ValueError) as error:
                    raise ValueError(f"record {record.id!r} has invalid ISO timestamp") from error
            temporal_folds = purged_kfold(
                records,
                intervals,
                folds,
                gap=timedelta(seconds=gap_seconds),
                embargo=timedelta(seconds=embargo_seconds),
                protect_groups=protect_groups,
            )
            return {
                "operation": operation,
                "schema_version": 1,
                "algorithm": "purged-time-kfold-v1",
                "gap_seconds": gap_seconds,
                "embargo_seconds": embargo_seconds,
                "protect_groups": protect_groups,
                "folds": [asdict(fold) for fold in temporal_folds],
            }
        if operation == "diagnose":
            assignments_value = request.get("assignments")
            if not isinstance(assignments_value, list):
                raise ValueError("assignments must be an array")
            assignments = tuple(Assignment(**item) for item in assignments_value)
            ratios = request.get("ratios")
            report = diagnose(records, assignments, ratios if isinstance(ratios, Mapping) else None)
            return {"operation": operation, "diagnostics": asdict(report)}
        if operation == "repeat_holdout":
            ratios = request.get("ratios")
            if not isinstance(ratios, Mapping):
                raise ValueError("ratios must be an object")
            repeats = request.get("repeats", 3)
            if isinstance(repeats, bool) or not isinstance(repeats, int):
                raise ValueError("repeats must be an integer")
            seed = request.get("seed", "0")
            stratified = request.get("stratified", False)
            if not isinstance(stratified, bool):
                raise ValueError("stratified must be a boolean")
            minimum_counts = request.get("minimum_counts")
            if minimum_counts is not None and (
                not isinstance(minimum_counts, Mapping)
                or not all(
                    isinstance(name, str) and isinstance(count, int) and not isinstance(count, bool)
                    for name, count in minimum_counts.items()
                )
            ):
                raise ValueError("minimum_counts must map split names to integer counts")
            repetitions = repeated_group_holdout(
                records,
                ratios,
                repeats,
                seed=seed,
                stratified=stratified,
                minimum_counts=minimum_counts,
            )
            holdout_report = holdout_stability_report(repetitions)
            return {
                "operation": operation,
                "repeats": holdout_report.repeats,
                "splits": list(holdout_report.splits),
                "allocation_rates": {
                    record_id: dict(rates)
                    for record_id, rates in holdout_report.allocation_rates.items()
                },
                "assignments": [
                    [asdict(item) for item in repetition] for repetition in repetitions
                ],
            }
        if operation == "migrate_manifest":
            manifest_path = request.get("manifest")
            if not isinstance(manifest_path, str) or not manifest_path.strip():
                raise ValueError("manifest must be a non-empty path string")
            migrated = migrate_manifest(load_manifest(manifest_path), records)
            return {"operation": operation, "manifest": migrated.to_dict()}
        raise ValueError(
            "operation must be split, kfold, temporal_kfold, diagnose, "
            "repeat_holdout, or migrate_manifest"
        )


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
