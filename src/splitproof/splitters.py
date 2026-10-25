"""Splitters that keep near-duplicate texts on the same side of every split.

``NearDupKFold`` follows the scikit-learn splitter protocol (``split`` and
``get_n_splits``), so it can be passed as ``cv=`` to ``cross_val_score``,
``GridSearchCV`` and friends. scikit-learn itself is not required.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterator, Mapping, Sequence
from typing import Any

import numpy as np

from .assign import assign_groups
from .neardup import _UnionFind, near_duplicate_groups


def as_texts(values: Any) -> list[str]:
    """Turn a list, tuple, NumPy array or pandas Series of strings into a list."""
    if hasattr(values, "tocsr"):
        raise TypeError("X is a sparse feature matrix; pass the raw texts with texts=...")
    if hasattr(values, "tolist") and not isinstance(values, (list, tuple)):
        array = np.asarray(values, dtype=object)
        if array.ndim != 1:
            raise TypeError(
                "expected a 1-D sequence of texts; for feature matrices pass the raw texts "
                "with texts=..."
            )
        values = array.tolist()
    texts = list(values)
    bad = next((value for value in texts if not isinstance(value, str)), None)
    if bad is not None:
        raise TypeError(
            f"expected texts but found a {type(bad).__name__}; if X holds features, "
            "pass the raw texts with texts=..."
        )
    return texts


_LABEL_COLLECTIONS = (list, tuple, set, frozenset, np.ndarray)


def label_sets(y: Any) -> list[tuple[Hashable, ...]] | None:
    """Convert ``y`` to one tuple of labels per row.

    Accepts class labels (1-D), a multi-label indicator matrix (a 2-D NumPy
    array, pandas DataFrame or scipy sparse matrix) or one collection of labels
    per row (a list of lists, a pandas Series of lists).
    """
    if y is None:
        return None
    if hasattr(y, "tocsr"):  # scipy sparse indicator matrix
        matrix = y.tocsr()
        return [
            tuple(matrix.indices[matrix.indptr[i] : matrix.indptr[i + 1]].tolist())
            for i in range(matrix.shape[0])
        ]
    if getattr(y, "ndim", 1) == 2:
        dense = np.asarray(y.to_numpy() if hasattr(y, "to_numpy") else y)
        return [tuple(np.flatnonzero(row).tolist()) for row in dense]
    values = y.tolist() if hasattr(y, "tolist") else list(y)
    if values and all(isinstance(item, _LABEL_COLLECTIONS) for item in values):
        return [tuple(item.tolist() if isinstance(item, np.ndarray) else item) for item in values]
    if any(isinstance(item, _LABEL_COLLECTIONS) for item in values):
        raise ValueError("y mixes single labels and label lists")
    return [(value,) for value in values]


def _missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and value != value)


def merge_groups(*labelings: Sequence[Hashable] | np.ndarray | None) -> np.ndarray:
    """Combine several groupings: rows that share a label in any of them end up together.

    ``None`` and NaN entries mean "no group" and never link rows.
    """
    present = [
        item.tolist() if hasattr(item, "tolist") else list(item)
        for item in labelings
        if item is not None
    ]
    if not present:
        raise ValueError("at least one grouping is required")
    size = len(present[0])
    if any(len(item) != size for item in present):
        raise ValueError("all groupings must have the same length")
    links = _UnionFind(size)
    for labeling in present:
        first: dict[Hashable, int] = {}
        for row, value in enumerate(labeling):
            if _missing(value):
                continue
            anchor = first.setdefault(value, row)
            if anchor != row:
                links.union(anchor, row)
    roots = np.fromiter((links.find(row) for row in range(size)), dtype=np.int64, count=size)
    _, merged = np.unique(roots, return_inverse=True)
    return merged.ravel()


class NearDupKFold:
    """K-fold cross-validation where near-duplicate texts never straddle folds.

    Texts are grouped with :func:`~splitproof.near_duplicate_groups`; groups
    passed to :meth:`split` (for example conversation or author IDs) are merged
    in. Whole groups are then assigned to folds so that fold sizes and, when
    ``y`` is given and ``stratify`` is true, the share of every label match the
    full dataset as closely as possible. ``y`` may hold one class per row or be
    a multi-label indicator matrix.

    Parameters
    ----------
    n_splits:
        Number of folds.
    threshold:
        Jaccard similarity of character n-gram sets at which two texts count as
        near-duplicates. ``1.0`` only groups texts that are identical after
        normalization.
    texts:
        The raw texts. Leave as ``None`` when ``X`` itself is a list of strings
        (for example in a ``TfidfVectorizer`` pipeline).
    stratify:
        Balance label shares across folds when ``y`` is given.
    ngram, num_perm:
        Shingle length and number of MinHash permutations.
    random_state:
        Breaks ties between equally good assignments.
    """

    def __init__(
        self,
        n_splits: int = 5,
        *,
        threshold: float = 0.8,
        texts: Sequence[str] | None = None,
        stratify: bool = True,
        ngram: int = 5,
        num_perm: int = 128,
        random_state: int = 0,
    ) -> None:
        if isinstance(n_splits, bool) or not isinstance(n_splits, int) or n_splits < 2:
            raise ValueError("n_splits must be an integer of at least 2")
        self.n_splits = n_splits
        self.threshold = threshold
        self.texts = texts
        self.stratify = stratify
        self.ngram = ngram
        self.num_perm = num_perm
        self.random_state = random_state
        self._cache: tuple[tuple[Any, ...], np.ndarray] | None = None

    def __repr__(self) -> str:
        return (
            f"NearDupKFold(n_splits={self.n_splits}, threshold={self.threshold}, "
            f"stratify={self.stratify}, random_state={self.random_state})"
        )

    def get_n_splits(self, X: Any = None, y: Any = None, groups: Any = None) -> int:
        """Return the number of folds (scikit-learn protocol)."""
        return self.n_splits

    def near_duplicate_groups(self, X: Any) -> np.ndarray:
        """Return the near-duplicate group of every row (cached between calls)."""
        texts = as_texts(X if self.texts is None else self.texts)
        key = (len(texts), hash(tuple(texts)), self.threshold, self.ngram, self.num_perm)
        if self._cache is None or self._cache[0] != key:
            labels = near_duplicate_groups(
                texts, self.threshold, ngram=self.ngram, num_perm=self.num_perm
            )
            self._cache = (key, labels)
        return self._cache[1]

    def fold_ids(self, X: Any, y: Any = None, groups: Any = None) -> np.ndarray:
        """Return the test fold (``0 .. n_splits-1``) of every row."""
        neardup = self.near_duplicate_groups(X)
        n_rows = len(neardup)
        if self.texts is not None and X is not None and _length(X) != n_rows:
            raise ValueError(f"X has {_length(X)} rows but texts has {n_rows}")
        merged = merge_groups(neardup, groups)
        n_groups = int(np.max(merged)) + 1
        if self.n_splits > n_groups:
            raise ValueError(
                f"cannot make {self.n_splits} folds from {n_groups} groups of near-duplicates"
            )
        labels = label_sets(y) if self.stratify else None
        if labels is not None and len(labels) != n_rows:
            raise ValueError(f"y has {len(labels)} rows but there are {n_rows} texts")
        return assign_groups(
            merged, [1.0 / self.n_splits] * self.n_splits, labels, seed=self.random_state
        )

    def split(
        self, X: Any, y: Any = None, groups: Any = None
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield ``(train_indices, test_indices)`` for each fold."""
        folds = self.fold_ids(X, y, groups)
        indices = np.arange(len(folds))
        for fold in range(self.n_splits):
            mask = folds == fold
            yield indices[~mask], indices[mask]


def assign_splits(
    texts: Sequence[str],
    labels: Any = None,
    *,
    ratios: Mapping[str, float] | None = None,
    groups: Sequence[Hashable] | None = None,
    threshold: float = 0.8,
    ngram: int = 5,
    num_perm: int = 128,
    random_state: int = 0,
) -> np.ndarray:
    """Name a split (``"train"``, ``"test"``, ...) for every text.

    ``ratios`` defaults to ``{"train": 0.8, "test": 0.2}``. ``labels`` and
    ``groups`` behave as in :class:`NearDupKFold`. The result is an array of
    split names, convenient for ``df["split"] = assign_splits(df.text, df.label)``.
    """
    targets = dict(ratios or {"train": 0.8, "test": 0.2})
    text_list = as_texts(texts)
    neardup = near_duplicate_groups(text_list, threshold, ngram=ngram, num_perm=num_perm)
    merged = merge_groups(neardup, groups)
    label_list = label_sets(labels)
    if label_list is not None and len(label_list) != len(text_list):
        raise ValueError("labels and texts must have the same length")
    names = list(targets)
    index = assign_groups(merged, list(targets.values()), label_list, seed=random_state)
    result: np.ndarray = np.asarray(names, dtype=object)[index]
    return result


def _length(values: Any) -> int:
    shape = getattr(values, "shape", None)
    if shape is not None and len(shape) > 0:
        return int(shape[0])
    return len(values)
