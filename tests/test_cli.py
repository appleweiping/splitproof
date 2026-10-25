import json

import pytest

from splitproof.cli import main
from splitproof.io import read_rows, write_rows

ROWS = [
    {"text": "Win a free cruise! Text WIN to 80082 now", "label": "spam", "thread": None},
    {"text": "WIN a FREE cruise!! text win to 80082 now", "label": "spam", "thread": None},
    {"text": "are we still on for lunch tomorrow?", "label": "ham", "thread": "t1"},
    {"text": "yes, see you at noon", "label": "ham", "thread": "t1"},
    {"text": "can you send me the slides from class", "label": "ham", "thread": "t2"},
    {"text": "sure, uploading them tonight", "label": "ham", "thread": "t2"},
    {"text": "Claim your prize: call 0906 now", "label": "spam", "thread": None},
    {"text": "running ten minutes late, sorry", "label": "ham", "thread": None},
    {"text": "did you finish the problem set", "label": "ham", "thread": None},
    {"text": "URGENT you have won a 1000 pound voucher", "label": "spam", "thread": None},
]


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_split_then_audit_round_trip(tmp_path, capsys):
    source = tmp_path / "sms.jsonl"
    output = tmp_path / "sms.split.jsonl"
    write_jsonl(source, ROWS)
    code = main(
        [
            "split",
            str(source),
            "-o",
            str(output),
            "--label-field",
            "label",
            "--group-field",
            "thread",
            "--ratios",
            "train=0.7,test=0.3",
        ]
    )
    assert code == 0
    printed = capsys.readouterr().out
    assert "10 rows -> 7 groups" in printed
    rows = read_rows(output)
    by_text = {row["text"]: row["split"] for row in rows}
    assert by_text[ROWS[0]["text"]] == by_text[ROWS[1]["text"]]
    assert by_text[ROWS[2]["text"]] == by_text[ROWS[3]["text"]]
    assert main(["audit", str(output)]) == 0
    assert "test rows with a near-duplicate in train: 0" in capsys.readouterr().out


def test_audit_two_files_reports_leak_and_fails(tmp_path, capsys):
    train, test = tmp_path / "train.jsonl", tmp_path / "test.jsonl"
    write_jsonl(train, ROWS[:6])
    write_jsonl(test, [ROWS[1], ROWS[7]])
    assert main(["audit", str(train), str(test)]) == 1
    out = capsys.readouterr().out
    assert "test rows with a near-duplicate in train: 1 (50.0%)" in out
    assert main(["audit", str(train), str(test), "--max-leak", "0.5"]) == 0
    capsys.readouterr()
    assert main(["audit", str(train), str(test), "--json"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["leaked"] == 1 and report["pairs"][0]["test"] == 0


def test_folds_and_csv_output(tmp_path, capsys):
    source = tmp_path / "data.json"
    source.write_text(json.dumps(ROWS), encoding="utf-8")
    output = tmp_path / "folds.csv"
    assert main(["split", str(source), "-o", str(output), "--folds", "3"]) == 0
    rows = read_rows(output)
    assert {row["fold"] for row in rows} == {"0", "1", "2"}


def test_errors_exit_with_code_two(tmp_path, capsys):
    source = tmp_path / "data.jsonl"
    write_jsonl(source, [{"body": "no text field"}])
    assert main(["split", str(source), "-o", str(tmp_path / "out.jsonl")]) == 2
    assert "no string field 'text'" in capsys.readouterr().err
    assert main(["audit", str(source)]) == 2
    with pytest.raises(SystemExit):
        main(["split", str(source), "-o", "x.jsonl", "--ratios", "train=0.5,test=0.6"])


def test_io_formats(tmp_path):
    rows = [{"text": "a", "n": 1}, {"text": "b", "n": 2}]
    for name in ("rows.jsonl", "rows.json", "rows.tsv"):
        write_rows(rows, tmp_path / name)
        loaded = read_rows(tmp_path / name)
        assert [row["text"] for row in loaded] == ["a", "b"]
    with pytest.raises(ValueError):
        read_rows(tmp_path / "rows.txt")
    pytest.importorskip("pyarrow")
    write_rows(rows, tmp_path / "rows.parquet")
    assert read_rows(tmp_path / "rows.parquet") == rows
