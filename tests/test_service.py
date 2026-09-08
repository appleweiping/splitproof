from __future__ import annotations

import json
import threading
import urllib.request
from dataclasses import replace

import pytest

from splitproof import SplitService, create_server
from splitproof.hashing import HASH_ALGORITHM, HASH_VERSION, data_fingerprint_v1
from splitproof.manifest import create_manifest, manifest_checksum, save_manifest
from splitproof.models import Assignment, SplitManifest


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


def test_split_service_repeat_holdout_reports_stability() -> None:
    service = SplitService()
    response = service.dispatch(
        {
            "operation": "repeat_holdout",
            "records": [
                {
                    "id": str(index),
                    "group": f"g{index // 2}",
                    "label": "a" if index % 2 else "b",
                }
                for index in range(8)
            ],
            "ratios": {"train": 0.75, "test": 0.25},
            "repeats": 3,
            "seed": "service",
            "stratified": True,
            "minimum_counts": {"train": 1},
        }
    )
    assert response["repeats"] == 3
    assert response["splits"] == ["test", "train"]
    assert set(response["allocation_rates"]) == {str(index) for index in range(8)}
    assert len(response["assignments"]) == 3


def test_split_service_dispatches_exact_group_algorithm() -> None:
    response = SplitService().dispatch(
        {
            "operation": "split",
            "records": [
                {"id": "a", "group": "one"},
                {"id": "b", "group": "two"},
                {"id": "c", "group": "three"},
            ],
            "ratios": {"train": 0.5, "test": 0.5},
            "algorithm": "exact",
            "max_groups": 3,
        }
    )
    assert {item["split"] for item in response["assignments"]} == {"train", "test"}


def test_split_service_dispatches_stratified_exact_group_algorithm() -> None:
    response = SplitService().dispatch(
        {
            "operation": "split",
            "records": [
                {"id": "a", "group": "one", "label": "x"},
                {"id": "b", "group": "two", "label": "y"},
                {"id": "c", "group": "three", "label": "x"},
                {"id": "d", "group": "four", "label": "y"},
            ],
            "ratios": {"train": 0.5, "test": 0.5},
            "algorithm": "exact",
            "stratified": True,
            "max_groups": 4,
        }
    )
    assert len(response["assignments"]) == 4


def test_split_service_dispatches_stratified_kfold() -> None:
    response = SplitService().dispatch(
        {
            "operation": "kfold",
            "records": [
                {"id": "a", "group": "one", "label": "x"},
                {"id": "b", "group": "two", "label": "y"},
                {"id": "c", "group": "three", "label": "x"},
                {"id": "d", "group": "four", "label": "y"},
            ],
            "folds": 2,
            "seed": "service",
            "stratified": True,
        }
    )
    assert response["folds"] == 2
    assert len(response["assignments"]) == 4
    assert {item["split"] for item in response["assignments"]} == {"fold-0", "fold-1"}


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("folds", True, "folds"),
        ("seed", None, "seed"),
        ("stratified", 1, "stratified"),
        ("max_local_iterations", True, "max_local_iterations"),
    ],
)
def test_split_service_kfold_validates_optional_fields(
    field: str, value: object, message: str
) -> None:
    request: dict[str, object] = {
        "operation": "kfold",
        "records": [{"id": "a"}, {"id": "b"}],
        field: value,
    }
    with pytest.raises(ValueError, match=message):
        SplitService().dispatch(request)


def test_split_service_dispatches_temporal_kfold() -> None:
    response = SplitService().dispatch(
        {
            "operation": "temporal_kfold",
            "records": [
                {
                    "id": "a",
                    "group": "one",
                    "start": "2026-01-01T00:00:00Z",
                    "end": "2026-01-01T00:00:00Z",
                },
                {
                    "id": "b",
                    "group": "two",
                    "start": "2026-01-02T00:00:00Z",
                    "end": "2026-01-02T00:00:00Z",
                },
                {
                    "id": "c",
                    "group": "three",
                    "start": "2026-01-03T00:00:00Z",
                    "end": "2026-01-03T00:00:00Z",
                },
                {
                    "id": "d",
                    "group": "four",
                    "start": "2026-01-04T00:00:00Z",
                    "end": "2026-01-04T00:00:00Z",
                },
            ],
            "folds": 2,
            "gap_seconds": 0,
            "embargo_seconds": 0,
        }
    )
    assert response["algorithm"] == "purged-time-kfold-v1"
    assert response["protect_groups"] is True
    assert len(response["folds"]) == 2
    assert {item for fold in response["folds"] for item in fold["validation"]} == {
        "a",
        "b",
        "c",
        "d",
    }


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("start_field", 1, "start_field"),
        ("end_field", "", "end_field"),
        ("folds", True, "folds"),
        ("gap_seconds", -1, "gap_seconds"),
        ("embargo_seconds", True, "embargo_seconds"),
        ("protect_groups", 1, "protect_groups"),
    ],
)
def test_split_service_temporal_kfold_validates_request(
    field: str, value: object, message: str
) -> None:
    request: dict[str, object] = {
        "operation": "temporal_kfold",
        "records": [],
        field: value,
    }
    with pytest.raises(ValueError, match=message):
        SplitService().dispatch(request)


def test_split_service_temporal_kfold_rejects_bad_timestamp() -> None:
    with pytest.raises(ValueError, match="invalid ISO timestamp"):
        SplitService().dispatch(
            {
                "operation": "temporal_kfold",
                "records": [{"id": "a", "start": "not-a-time", "end": "2026-01-01T00:00:00Z"}],
                "folds": 2,
            }
        )


def test_split_service_migrates_a_verified_v1_manifest(tmp_path) -> None:
    records = [{"id": "a", "group": "g"}, {"id": "b", "group": "g"}]
    rows = tuple(__import__("splitproof").io.load_records(_write_records(tmp_path, records)))
    legacy = SplitManifest(
        schema_version="1",
        algorithm="group",
        algorithm_version="2",
        seed="legacy",
        created_at="2026-01-01T00:00:00+00:00",
        data_fingerprint=data_fingerprint_v1(rows),
        ratios={"train": 1.0},
        assignments=(Assignment("a", "train"), Assignment("b", "train")),
        metadata={"hash_algorithm": HASH_ALGORITHM, "hash_version": HASH_VERSION},
    )
    legacy = replace(legacy, checksum=manifest_checksum(legacy))
    path = tmp_path / "legacy.json"
    save_manifest(legacy, path)
    response = SplitService().dispatch(
        {"operation": "migrate_manifest", "records": records, "manifest": str(path)}
    )
    assert response["manifest"]["schema_version"] == "2"


def test_split_service_verifies_manifest_and_external_assignments(tmp_path) -> None:
    records = [{"id": "a"}, {"id": "b"}]
    rows = tuple(__import__("splitproof").io.load_records(_write_records(tmp_path, records)))
    assignments = (Assignment("a", "train"), Assignment("b", "test"))
    manifest = create_manifest(
        rows,
        assignments,
        algorithm="hash",
        algorithm_version="3",
        seed="service",
        ratios={"train": 0.5, "test": 0.5},
    )
    path = tmp_path / "manifest.json"
    save_manifest(manifest, path)
    response = SplitService().dispatch(
        {
            "operation": "verify",
            "records": records,
            "manifest": str(path),
            "assignments": [
                {"record_id": "a", "split": "train"},
                {"record_id": "b", "split": "test"},
            ],
        }
    )
    assert response["verified"] is True
    assert response["errors"] == []


def test_split_service_materializes_verified_manifest(tmp_path) -> None:
    records = [{"id": "a", "group": "g1"}, {"id": "b", "group": "g2"}]
    rows = tuple(__import__("splitproof").io.load_records(_write_records(tmp_path, records)))
    assignments = (Assignment("a", "train"), Assignment("b", "test"))
    manifest = create_manifest(
        rows,
        assignments,
        algorithm="hash",
        algorithm_version="3",
        seed="service",
        ratios={"train": 0.5, "test": 0.5},
    )
    manifest_path = tmp_path / "manifest.json"
    save_manifest(manifest, manifest_path)
    output_dir = tmp_path / "materialized"
    response = SplitService().dispatch(
        {
            "operation": "materialize",
            "records": records,
            "manifest": str(manifest_path),
            "output_dir": str(output_dir),
        }
    )
    assert response["operation"] == "materialize"
    assert sorted(path.name for path in output_dir.glob("*.jsonl")) == ["test.jsonl", "train.jsonl"]
    assert sorted(response["files"]) == sorted(str(path) for path in output_dir.glob("*.jsonl"))
    assert (
        output_dir.joinpath("train.jsonl").read_text(encoding="utf-8").strip()
        == '{"group": "g1", "id": "a"}'
    )


@pytest.mark.parametrize(
    "field,value,message",
    [("manifest", "", "manifest"), ("output_dir", None, "output_dir")],
)
def test_split_service_materialize_validates_paths(
    tmp_path, field: str, value: object, message: str
) -> None:
    request: dict[str, object] = {
        "operation": "materialize",
        "records": [],
        "manifest": str(tmp_path / "manifest.json"),
        "output_dir": str(tmp_path / "out"),
    }
    request[field] = value
    with pytest.raises(ValueError, match=message):
        SplitService().dispatch(request)


def _write_records(tmp_path, records):  # type: ignore[no-untyped-def]
    path = tmp_path / "records.jsonl"
    path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    "payload",
    [
        {"operation": "repeat_holdout", "records": [], "ratios": []},
        {"operation": "repeat_holdout", "records": [], "ratios": {}, "repeats": True},
        {"operation": "repeat_holdout", "records": [], "ratios": {}, "stratified": 1},
        {
            "operation": "repeat_holdout",
            "records": [],
            "ratios": {},
            "minimum_counts": {"train": True},
        },
        {"operation": "unknown", "records": []},
    ],
)
def test_split_service_repeat_holdout_validates_request(payload: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        SplitService().dispatch(payload)


def test_split_service_rejects_invalid_port() -> None:
    with pytest.raises(ValueError, match="port"):
        create_server(port=65536)
