"""Shared deterministic balance objective used by optimizers and diagnostics."""

from __future__ import annotations

from collections.abc import Mapping


def _squared_ratio_error(
    observed: Mapping[str, float], total: float, ratios: Mapping[str, float]
) -> float:
    if total <= 0:
        return 0.0
    return sum(
        ((observed.get(split, 0.0) - ratio * total) / total) ** 2 for split, ratio in ratios.items()
    )


def balance_components(
    *,
    ratios: Mapping[str, float],
    counts: Mapping[str, float],
    total_count: float,
    record_weights: Mapping[str, float],
    total_record_weight: float,
    group_weights: Mapping[str, float],
    total_group_weight: float,
    label_weights: Mapping[str, Mapping[str, float]],
    label_totals: Mapping[str, float],
    include_labels: bool,
) -> dict[str, float]:
    """Return additive, scale-free objective components.

    Count, record-weight, and effective-group-weight errors receive equal weight.
    The mean per-label weighted error receives weight two when stratification is
    enabled. All terms are squared deviations from requested split ratios.
    """
    count_balance = _squared_ratio_error(counts, total_count, ratios)
    record_weight_balance = _squared_ratio_error(record_weights, total_record_weight, ratios)
    group_weight_balance = _squared_ratio_error(group_weights, total_group_weight, ratios)
    label_balance = 0.0
    if include_labels and label_totals:
        label_balance = (
            2.0
            * sum(
                _squared_ratio_error(label_weights.get(label, {}), total, ratios)
                for label, total in label_totals.items()
            )
            / len(label_totals)
        )
    result = {
        "count_balance": count_balance,
        "record_weight_balance": record_weight_balance,
        "group_weight_balance": group_weight_balance,
        "label_balance": label_balance,
    }
    result["total"] = sum(result.values())
    return result
