"""Measure how much of a test set already appears (almost) verbatim in training."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .neardup import NearDuplicatePair, find_cross_duplicates


@dataclass(frozen=True)
class LeakageReport:
    """Test rows that have an exact or near-duplicate in the training data."""

    n_train: int
    n_test: int
    threshold: float
    pairs: tuple[NearDuplicatePair, ...]

    @property
    def leaked(self) -> int:
        """Number of test rows with at least one match in training."""
        return len(self.pairs)

    @property
    def exact(self) -> int:
        """Leaked test rows whose normalized text also occurs in training."""
        return sum(1 for pair in self.pairs if pair.similarity >= 1.0)

    @property
    def fraction(self) -> float:
        """Share of test rows that leaked."""
        return self.leaked / self.n_test if self.n_test else 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "n_train": self.n_train,
            "n_test": self.n_test,
            "threshold": self.threshold,
            "leaked": self.leaked,
            "exact": self.exact,
            "fraction": self.fraction,
            "pairs": [
                {"train": pair.left, "test": pair.right, "similarity": round(pair.similarity, 4)}
                for pair in self.pairs
            ],
        }


def leakage_report(
    train_texts: Sequence[str],
    test_texts: Sequence[str],
    threshold: float = 0.8,
    *,
    ngram: int = 5,
    num_perm: int = 128,
) -> LeakageReport:
    """Report every test text with a training text at or above ``threshold``.

    Similarity is the Jaccard index of character ``ngram`` shingles after
    normalization (see :func:`~splitproof.normalize_text`).
    """
    pairs = find_cross_duplicates(
        train_texts, test_texts, threshold, ngram=ngram, num_perm=num_perm
    )
    return LeakageReport(len(train_texts), len(test_texts), float(threshold), tuple(pairs))
