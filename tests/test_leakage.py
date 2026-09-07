from splitproof import Assignment, Record, audit_leakage, audit_near_duplicates


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


def test_near_duplicate_audit_uses_shingles_and_redacts_source() -> None:
    rows = (
        Record("a", payload={"text": "alpha beta gamma delta epsilon"}),
        Record("b", payload={"text": "alpha beta gamma delta zeta"}),
        Record("c", payload={"text": "unrelated words here now"}),
    )
    report = audit_near_duplicates(
        rows,
        {"a": "train", "b": "test", "c": "train"},
        threshold=0.5,
    )
    assert not report.valid
    assert report.findings[0].record_id == "a"
    assert report.findings[0].other_record_id == "b"
    assert 0.5 <= report.findings[0].similarity <= 1
    assert "alpha" not in str(report.to_dict())


def test_near_duplicate_options_and_bounds() -> None:
    rows = (Record("a", payload={"text": "one two three"}),)
    for kwargs in ({"threshold": 2}, {"min_tokens": 0}, {"max_pairs": 0}):
        try:
            audit_near_duplicates(rows, {"a": "train"}, **kwargs)
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError("expected invalid near-duplicate option")


def test_near_duplicate_skips_missing_non_text_and_same_split_candidates() -> None:
    rows = (
        Record("a", payload={"text": "one two three four"}),
        Record("b", payload={"text": "one two three five"}),
        Record("c", payload={"text": 42}),
        Record("d", payload={}),
    )
    report = audit_near_duplicates(rows, {"a": "train", "b": "train", "c": "test", "d": "test"})
    assert report.valid
    duplicate = Record("a", payload={"text": "x"})
    try:
        audit_leakage((duplicate, duplicate), {"a": "train"})
    except ValueError as error:
        assert "duplicate IDs" in str(error)
    else:
        raise AssertionError("expected duplicate record IDs")
