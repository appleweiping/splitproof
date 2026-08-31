"""Strict JSON helpers shared by all persistence boundaries."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from typing import Any, NoReturn


def _reject_constant(value: str) -> NoReturn:
    raise ValueError(f"non-finite number {value!r} is not valid JSON")


def _parse_finite_float(value: str) -> float:
    """Parse a JSON number without allowing binary64 overflow to infinity."""
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"number {value!r} is outside the supported finite range")
    return result


def _reject_duplicate_keys(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def strict_loads(text: str) -> Any:
    """Parse RFC-compatible JSON, rejecting duplicate keys and non-finite numbers."""
    return json.loads(
        text,
        parse_constant=_reject_constant,
        parse_float=_parse_finite_float,
        object_pairs_hook=_reject_duplicate_keys,
    )


def strict_dumps(value: Any, **kwargs: Any) -> str:
    """Serialize JSON while refusing NaN and infinities at any depth."""
    return json.dumps(value, allow_nan=False, **kwargs)
