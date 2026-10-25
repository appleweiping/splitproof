"""Assign indivisible groups of rows to splits.

The objective is the squared difference between each split's share and its
target ratio, summed over splits, plus the same quantity averaged over labels
when labels are given. Groups are placed greedily (largest first, then the ones
holding the rarest labels), after which single-group moves are applied while
they lower the objective. For small problems, pairwise swaps are tried too.
Everything is computed incrementally, so tens of thousands of groups take
about a second.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence

import numpy as np

_EPSILON = 1e-12
_SWAP_LIMIT = 200


class _State:
    """Running per-split counts and the incremental objective."""

    def __init__(
        self,
        ratios: Sequence[float],
        total: int,
        label_totals: Sequence[int],
    ) -> None:
        self.splits = len(ratios)
        self.count_target = [ratio * total for ratio in ratios]
        self.count = [0.0] * self.splits
        self.count_weight = 1.0 / (total * total)
        self.label_target = [[ratio * value for ratio in ratios] for value in label_totals]
        self.label_count = [[0.0] * self.splits for _ in label_totals]
        n_labels = len(label_totals)
        self.label_weight = [1.0 / (n_labels * value * value) for value in label_totals]

    def delta(self, split: int, size: int, labels: dict[int, int]) -> float:
        """Objective change if ``size`` rows with ``labels`` are added to ``split``.

        Negative sizes and counts describe a removal.
        """
        offset = self.count[split] - self.count_target[split]
        result = size * (2.0 * offset + size) * self.count_weight
        for label, amount in labels.items():
            offset = self.label_count[label][split] - self.label_target[label][split]
            result += amount * (2.0 * offset + amount) * self.label_weight[label]
        return result

    def apply(self, split: int, size: int, labels: dict[int, int]) -> None:
        self.count[split] += size
        for label, amount in labels.items():
            self.label_count[label][split] += amount


def _negated(labels: dict[int, int]) -> dict[int, int]:
    return {label: -amount for label, amount in labels.items()}


def _difference(left: dict[int, int], right: dict[int, int]) -> dict[int, int]:
    result = dict(left)
    for label, amount in right.items():
        result[label] = result.get(label, 0) - amount
    return result


def assign_groups(
    groups: Sequence[Hashable] | np.ndarray,
    ratios: Sequence[float],
    labels: Sequence[Sequence[Hashable]] | None = None,
    *,
    seed: int = 0,
    max_passes: int = 10,
) -> np.ndarray:
    """Return the split index (``0 .. len(ratios)-1``) chosen for every row.

    ``groups[i]`` names the group of row ``i``; rows with the same group always
    get the same split. ``labels[i]`` is the (possibly empty) collection of
    labels of row ``i``; pass ``None`` to balance only the number of rows.
    ``seed`` only changes how otherwise equivalent choices are broken, so
    different seeds give different but equally balanced assignments.
    """
    targets = [float(value) for value in ratios]
    if not targets or any(not np.isfinite(value) or value <= 0 for value in targets):
        raise ValueError("ratios must be positive numbers")
    if abs(sum(targets) - 1.0) > 1e-6:
        raise ValueError(f"ratios must sum to 1, got {sum(targets):.6g}")
    group_array = np.asarray(groups)
    n_rows = len(group_array)
    if n_rows == 0:
        raise ValueError("cannot split an empty dataset")
    if labels is not None and len(labels) != n_rows:
        raise ValueError("labels and groups must have the same length")
    _, group_of_row = np.unique(group_array, return_inverse=True)
    group_of_row = group_of_row.ravel()
    n_groups = int(np.max(group_of_row)) + 1
    sizes = np.bincount(group_of_row, minlength=n_groups).tolist()

    label_ids: dict[Hashable, int] = {}
    group_labels: list[dict[int, int]] = [{} for _ in range(n_groups)]
    if labels is not None:
        for row, row_labels in enumerate(labels):
            bucket = group_labels[group_of_row[row]]
            for label in set(row_labels):
                label_id = label_ids.setdefault(label, len(label_ids))
                bucket[label_id] = bucket.get(label_id, 0) + 1
    label_totals = [0] * len(label_ids)
    for bucket in group_labels:
        for label, amount in bucket.items():
            label_totals[label] += amount

    state = _State(targets, n_rows, label_totals)
    tie = np.random.RandomState(seed).permutation(n_groups).tolist()

    def rarity(group: int) -> float:
        bucket = group_labels[group]
        return max((amount / label_totals[label] for label, amount in bucket.items()), default=0.0)

    order = sorted(range(n_groups), key=lambda group: (-sizes[group], -rarity(group), tie[group]))
    split_count = len(targets)
    placed_in = [0] * split_count
    destination = [-1] * n_groups

    for position, group in enumerate(order):
        empty = [split for split in range(split_count) if placed_in[split] == 0]
        remaining = n_groups - position
        candidates = empty if empty and remaining <= len(empty) else range(split_count)
        best_split = min(
            candidates,
            key=lambda split: (
                state.delta(split, sizes[group], group_labels[group]),
                (tie[group] + split) % split_count,
            ),
        )
        state.apply(best_split, sizes[group], group_labels[group])
        placed_in[best_split] += 1
        destination[group] = best_split

    keep_nonempty = n_groups >= split_count
    for _ in range(max_passes):
        improved = False
        for group in order:
            source = destination[group]
            if keep_nonempty and placed_in[source] <= 1:
                continue
            size, bucket = sizes[group], group_labels[group]
            removal = state.delta(source, -size, _negated(bucket))
            best_split, best_delta = source, -_EPSILON
            for split in range(split_count):
                if split == source:
                    continue
                change = removal + state.delta(split, size, bucket)
                if change < best_delta:
                    best_split, best_delta = split, change
            if best_split != source:
                state.apply(source, -size, _negated(bucket))
                state.apply(best_split, size, bucket)
                placed_in[source] -= 1
                placed_in[best_split] += 1
                destination[group] = best_split
                improved = True
        if n_groups <= _SWAP_LIMIT:
            improved = _swap_pass(order, destination, sizes, group_labels, state) or improved
        if not improved:
            break

    return np.asarray(destination, dtype=np.int64)[group_of_row]


def _swap_pass(
    order: list[int],
    destination: list[int],
    sizes: list[int],
    group_labels: list[dict[int, int]],
    state: _State,
) -> bool:
    """Swap pairs of groups in different splits while that lowers the objective."""
    improved = False
    for index, left in enumerate(order):
        for right in order[index + 1 :]:
            left_split, right_split = destination[left], destination[right]
            if left_split == right_split:
                continue
            size_change = sizes[right] - sizes[left]
            label_change = _difference(group_labels[right], group_labels[left])
            change = state.delta(left_split, size_change, label_change) + state.delta(
                right_split, -size_change, _negated(label_change)
            )
            if change < -_EPSILON:
                state.apply(left_split, size_change, label_change)
                state.apply(right_split, -size_change, _negated(label_change))
                destination[left], destination[right] = right_split, left_split
                improved = True
    return improved
