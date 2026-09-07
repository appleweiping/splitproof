import json
from pathlib import Path

import pytest

from splitproof.cli import main


def test_temporal_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    source = tmp_path / "records.json"
    source.write_text(
        json.dumps(
            [
                {
                    "id": str(i),
                    "start": f"2026-01-0{i + 1}T00:00:00Z",
                    "end": f"2026-01-0{i + 1}T00:00:00Z",
                }
                for i in range(4)
            ]
        ),
        encoding="utf-8",
    )
    args = ["temporal-kfold", str(source), "--folds", "2"]
    assert main(args) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["folds"][0]["validation"] == ["0", "1"]
    assert result["folds"][0]["train"] == ["2", "3"]
    destination = tmp_path / "folds.json"
    assert main([*args, "--output", str(destination)]) == 0
    assert json.loads(destination.read_text(encoding="utf-8")) == result
    before = source.read_bytes()
    assert main([*args, "--output", str(source)]) == 2
    assert source.read_bytes() == before


def test_temporal_cli_invalid_inputs_preserve_output(tmp_path: Path) -> None:
    source, output = tmp_path / "records.json", tmp_path / "output.json"
    source.write_text('[{"id":"a","start":"2026-01-01"}]', encoding="utf-8")
    output.write_text("keep", encoding="utf-8")
    assert main(["temporal-kfold", str(source), "--output", str(output)]) == 2
    assert output.read_text(encoding="utf-8") == "keep"


def test_repeat_cli_reports_stability(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    source = tmp_path / "records.json"
    source.write_text(
        json.dumps([{"id": str(index), "group": f"g{index // 2}"} for index in range(8)]),
        encoding="utf-8",
    )
    output = tmp_path / "repeat.json"
    assert (
        main(["repeat", str(source), "--folds", "2", "--repeats", "2", "--output", str(output)])
        == 0
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["repeats"] == 2 and len(report["assignments"]) == 2
