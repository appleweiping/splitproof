from __future__ import annotations

import json
from pathlib import Path

from benchmarks.compare_methods import run


def test_benchmark_is_reproducible_and_optimizer_improves_greedy() -> None:
    first = run()
    second = run()
    assert first == second
    committed = json.loads(
        (Path(__file__).parents[1] / "benchmarks" / "results-v0.2.json").read_text(encoding="utf-8")
    )
    assert first == committed
    methods = first["methods"]
    assert isinstance(methods, dict)
    assert all(value["order_independent"] for value in methods.values())
    assert methods["optimized_v3"]["group_leakage"] == 0
    assert methods["optimized_v3"]["objective"] <= methods["greedy_only_v3"]["objective"]
