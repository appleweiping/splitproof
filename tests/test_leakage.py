from splitproof import Assignment, Record, audit_leakage


def _records() -> tuple[Record, ...]:
    return (
        Record("a", payload={"text": "Hello   WORLD", "question": "q1"}),
        Record("b", payload={"text": " hello world ", "question": "q2"}),
        Record("c", payload={"text": "unique", "question": "q1"}),
    )


def test_audit_reports_cross_split_normalized_duplicates() -> None:
    report = audit_leakage(
        _records(),
        (Assignment("a", "train"), Assignment("b", "test"), Assignment("c", "train")),
        fields=("text", "question"),
    )
    assert not report.valid
    assert report.pairs == 1
    assert {item.field for item in report.findings} == {"text"}
    assert report.findings[0].record_id == "a"
    assert report.to_dict()["summary"]["valid"] is False


def test_same_split_duplicates_are_not_leakage() -> None:
    report = audit_leakage(
        _records(),
        {"a": "train", "b": "train", "c": "train"},
        fields=("text",),
    )
    assert report.valid
    assert report.pairs == 0


def test_findings_are_bounded_and_missing_assignments_are_ignored() -> None:
    report = audit_leakage(
        _records(),
        {"a": "train", "b": "test"},
        fields=("text",),
        max_findings=1,
    )
    assert report.pairs == 1
    assert not report.valid
    assert report.assigned == 2


def test_json_values_and_truncation_are_deterministic() -> None:
    rows = (
        Record("a", payload={"text": {"x": [1, 2]}}),
        Record("b", payload={"text": {"x": [1, 2]}}),
        Record("c", payload={"text": {"x": [1, 2]}}),
    )
    report = audit_leakage(
        rows, {"a": "train", "b": "test", "c": "dev"}, fields=("text",), max_findings=1
    )
    assert report.truncated
    assert report.pairs == 1
    assert report.findings[0].value_digest


def test_invalid_assignment_and_limits_are_rejected() -> None:
    rows = _records()
    for assignments, expected in (({"": "train"}, "IDs"), ({"a": ""}, "split")):
        try:
            audit_leakage(rows, assignments, fields=("text",))
        except ValueError as error:
            assert expected in str(error)
        else:
            raise AssertionError("expected invalid assignment error")
    for kwargs in ({"min_length": -1}, {"max_findings": 0}):
        try:
            audit_leakage(rows, {}, fields=("text",), **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("expected invalid limit error")


def test_invalid_options_are_rejected() -> None:
    try:
        audit_leakage(_records(), {}, fields=())
    except ValueError as error:
        assert "fields" in str(error)
    else:
        raise AssertionError("expected invalid fields error")
