from collections import Counter

import numpy as np
import pytest

from splitproof import assign_groups


def test_groups_never_straddle_splits():
    rng = np.random.RandomState(3)
    groups = rng.randint(0, 40, size=400)
    split = assign_groups(groups, [0.6, 0.2, 0.2])
    for group in np.unique(groups):
        assert len(set(split[groups == group].tolist())) == 1


def test_singleton_groups_hit_ratios_exactly():
    split = assign_groups(np.arange(1000), [0.7, 0.2, 0.1])
    assert np.bincount(split).tolist() == [700, 200, 100]


def test_label_shares_are_balanced():
    rng = np.random.RandomState(0)
    labels = [
        (label,) for label in rng.choice(["a", "b", "c", "rare"], 2000, p=[0.5, 0.3, 0.18, 0.02])
    ]
    split = assign_groups(np.arange(2000), [0.8, 0.2], labels)
    overall = Counter(label for (label,) in labels)
    test = Counter(label for (label,), side in zip(labels, split) if side == 1)
    for label, count in overall.items():
        assert abs(test[label] / count - 0.2) < 0.01


def test_multilabel_rows_and_empty_label_sets():
    labels = [("x", "y"), ("x",), (), ("y",)] * 50
    split = assign_groups(np.arange(200), [0.5, 0.5], labels)
    assert np.bincount(split).tolist() == [100, 100]
    for name in ("x", "y"):
        sides = [side for row, side in zip(labels, split) if name in row]
        assert abs(np.mean(sides) - 0.5) < 0.05


def test_every_split_gets_a_group_when_possible():
    groups = [0] * 90 + [1] * 5 + [2] * 5
    split = assign_groups(groups, [0.8, 0.1, 0.1])
    assert set(split.tolist()) == {0, 1, 2}


def test_seed_changes_ties_not_quality():
    first = assign_groups(np.arange(100), [0.5, 0.5], seed=1)
    second = assign_groups(np.arange(100), [0.5, 0.5], seed=2)
    assert not np.array_equal(first, second)
    assert np.bincount(first).tolist() == np.bincount(second).tolist() == [50, 50]
    assert np.array_equal(first, assign_groups(np.arange(100), [0.5, 0.5], seed=1))


def test_input_validation():
    with pytest.raises(ValueError):
        assign_groups([1, 2], [0.5, 0.6])
    with pytest.raises(ValueError):
        assign_groups([1, 2], [1.0, 0.0])
    with pytest.raises(ValueError):
        assign_groups([], [1.0])
    with pytest.raises(ValueError):
        assign_groups([1, 2], [0.5, 0.5], [("a",)])
