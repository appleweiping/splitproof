"""Find exact and near-duplicate texts.

Texts are normalized (Unicode NFKC, case-folded, punctuation collapsed to single
spaces) and cut into overlapping character n-grams ("shingles"). MinHash
signatures plus LSH banding propose candidate pairs cheaply; every candidate is
then checked with the exact Jaccard similarity of the two shingle sets, so a
pair reported here really is at or above the threshold. Pairs that never share
an LSH bucket can be missed; the band layout is chosen so that a pair exactly
at the threshold is proposed with probability of at least 98%, and more similar
pairs almost always are.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np

_NON_WORD = re.compile(r"[\W_]+", re.UNICODE)
_MAX_HASH = np.uint32(0xFFFFFFFF)
_BASE = np.uint64(0x100000001B3)  # FNV-64 prime, odd
_MIX = np.uint64(0xFF51AFD7ED558CCD)  # MurmurHash3 finalizer constant


def normalize_text(text: str) -> str:
    """Return the form used for comparison: NFKC, case-folded, punctuation as spaces."""
    if not isinstance(text, str):
        raise TypeError(f"expected str, got {type(text).__name__}")
    folded = unicodedata.normalize("NFKC", text).casefold()
    return _NON_WORD.sub(" ", folded).strip()


def shingle_sets(texts: Sequence[str], ngram: int = 5) -> list[np.ndarray]:
    """Return, for every text, the sorted unique 32-bit hashes of its character n-grams.

    ``texts`` should already be normalized. A text shorter than ``ngram`` is a
    single shingle; an empty text has none. All texts are hashed in one
    vectorized pass, which is what keeps tens of thousands of texts fast.
    """
    if ngram < 1:
        raise ValueError("ngram must be positive")
    lengths = np.fromiter((len(text) for text in texts), dtype=np.int64, count=len(texts))
    if lengths.size == 0:
        return []
    # Pad short texts with NUL (never present after normalization) to one full shingle.
    padded = np.where(lengths > 0, np.maximum(lengths, ngram), 0)
    joined = "".join(
        text + "\0" * (ngram - len(text)) if 0 < len(text) < ngram else text for text in texts
    )
    codes = np.frombuffer(joined.encode("utf-32-le"), dtype=np.uint32).astype(np.uint64)
    windows = np.where(padded > 0, padded - ngram + 1, 0)
    total = int(windows.sum())
    if total == 0:
        return [np.empty(0, dtype=np.uint32) for _ in texts]
    starts = np.cumsum(padded) - padded
    owner = np.repeat(np.arange(len(texts), dtype=np.int64), windows)
    first_window = np.cumsum(windows) - windows
    position = np.arange(total, dtype=np.int64) - first_window[owner] + starts[owner]
    with np.errstate(over="ignore"):
        rolling = np.zeros(total, dtype=np.uint64)
        for offset in range(ngram):
            rolling = rolling * _BASE + codes[position + offset]
        rolling ^= rolling >> np.uint64(33)
        rolling *= _MIX
        rolling ^= rolling >> np.uint64(33)
    hashes = rolling >> np.uint64(32)
    # Sort by (text, hash) and drop repeats so each text keeps a set of hashes.
    keyed = np.sort((owner.astype(np.uint64) << np.uint64(32)) | hashes)
    if keyed.size > 1:
        keyed = keyed[np.concatenate(([True], keyed[1:] != keyed[:-1]))]
    text_of_key = (keyed >> np.uint64(32)).astype(np.int64)
    values = (keyed & np.uint64(0xFFFFFFFF)).astype(np.uint32)
    counts = np.bincount(text_of_key, minlength=len(texts))
    return np.split(values, np.cumsum(counts)[:-1])


def shingle_hashes(text: str, ngram: int = 5) -> np.ndarray:
    """Return the sorted unique shingle hashes of one already-normalized text."""
    return shingle_sets([text], ngram)[0]


def jaccard(left: np.ndarray, right: np.ndarray) -> float:
    """Exact Jaccard similarity of two sorted unique hash arrays."""
    if left.size == 0 and right.size == 0:
        return 1.0
    shared = np.intersect1d(left, right, assume_unique=True).size
    return shared / (left.size + right.size - shared)


def lsh_bands(threshold: float, num_perm: int, min_recall: float = 0.98) -> tuple[int, int]:
    """Pick ``(bands, rows)`` for banded LSH.

    Returns the most selective layout (largest ``rows``) for which a pair whose
    true similarity equals ``threshold`` still becomes a candidate with
    probability ``min_recall``. Higher similarities are proposed more reliably.
    """
    if not 0.0 < threshold <= 1.0:
        raise ValueError("threshold must be in (0, 1]")
    best = (num_perm, 1)
    for rows in range(1, num_perm + 1):
        bands = num_perm // rows
        recall = 1.0 - (1.0 - threshold**rows) ** bands
        if recall >= min_recall:
            best = (bands, rows)
    return best


class MinHasher:
    """MinHash signatures with a fixed, seeded family of multiply-shift hashes."""

    def __init__(self, num_perm: int = 128, seed: int = 0) -> None:
        if num_perm < 1:
            raise ValueError("num_perm must be positive")
        rng = np.random.RandomState(seed)
        high = np.uint64(1 << 63)
        self.num_perm = num_perm
        self._a = rng.randint(1, high, size=num_perm, dtype=np.uint64) | np.uint64(1)
        self._b = rng.randint(0, high, size=num_perm, dtype=np.uint64)

    def signatures(self, sets: Sequence[np.ndarray]) -> np.ndarray:
        """Return a ``(len(sets), num_perm)`` uint32 signature matrix.

        Empty sets keep the all-ones signature; callers should not bucket them.
        """
        result = np.full((len(sets), self.num_perm), _MAX_HASH, dtype=np.uint32)
        sizes = np.fromiter((item.size for item in sets), dtype=np.int64, count=len(sets))
        nonempty = np.flatnonzero(sizes)
        if nonempty.size == 0:
            return result
        flat = np.concatenate([sets[i] for i in nonempty.tolist()]).astype(np.uint64)
        offsets = np.cumsum(sizes[nonempty]) - sizes[nonempty]
        hashed = np.empty_like(flat)
        shift = np.uint64(32)
        with np.errstate(over="ignore"):
            for column in range(self.num_perm):
                np.multiply(flat, self._a[column], out=hashed)
                hashed += self._b[column]
                hashed >>= shift
                result[nonempty, column] = np.minimum.reduceat(hashed, offsets)
        return result


@dataclass(frozen=True)
class NearDuplicatePair:
    """Two row indices whose normalized texts reach the similarity threshold."""

    left: int
    right: int
    similarity: float


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        parent = self.parent
        root = item
        while parent[root] != root:
            root = parent[root]
        while parent[item] != root:
            parent[item], item = root, parent[item]
        return root

    def union(self, left: int, right: int) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            if a < b:
                self.parent[b] = a
            else:
                self.parent[a] = b


class _Corpus:
    """Unique normalized texts with their shingles and MinHash signatures."""

    def __init__(self, texts: Sequence[str], *, ngram: int, num_perm: int, seed: int) -> None:
        if ngram < 1:
            raise ValueError("ngram must be positive")
        unique: dict[str, int] = {}
        row_to_unique = np.empty(len(texts), dtype=np.int64)
        for row, text in enumerate(texts):
            key = normalize_text(text)
            row_to_unique[row] = unique.setdefault(key, len(unique))
        self.keys = list(unique)
        self.row_to_unique = row_to_unique
        self.shingles = shingle_sets(self.keys, ngram)
        self.signatures = MinHasher(num_perm, seed).signatures(self.shingles)

    def buckets(self, bands: int, rows: int) -> Iterator[np.ndarray]:
        """Yield arrays of unique-text indices that share at least one LSH band."""
        searchable = np.flatnonzero([item.size > 0 for item in self.shingles])
        if searchable.size < 2:
            return
        signatures = self.signatures[searchable]
        for band in range(bands):
            block = np.ascontiguousarray(signatures[:, band * rows : (band + 1) * rows])
            keys = block.view(np.dtype((np.void, block.dtype.itemsize * rows))).ravel()
            _, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
            inverse = inverse.ravel()
            shared = np.flatnonzero(counts > 1)
            if shared.size == 0:
                continue
            in_shared = np.flatnonzero(np.isin(inverse, shared))
            order = in_shared[np.argsort(inverse[in_shared], kind="stable")]
            for members in np.split(order, np.cumsum(counts[shared])[:-1]):
                yield searchable[members]


def _check_threshold(threshold: float) -> float:
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise TypeError("threshold must be a number")
    value = float(threshold)
    if not 0.0 < value <= 1.0:
        raise ValueError("threshold must be in (0, 1]")
    return value


def near_duplicate_groups(
    texts: Sequence[str],
    threshold: float = 0.8,
    *,
    ngram: int = 5,
    num_perm: int = 128,
    seed: int = 0,
) -> np.ndarray:
    """Label every text with a group ID so that near-duplicates share a label.

    Two texts are linked when the Jaccard similarity of their character
    ``ngram`` shingles is at least ``threshold`` (texts that are identical after
    normalization always are). Groups are the connected components of these
    links, so a chain A~B~C ends up in one group even if A and C differ more.
    Labels are consecutive integers in order of first appearance, which makes
    the result usable as ``groups=`` for any scikit-learn group splitter.
    """
    threshold = _check_threshold(threshold)
    corpus = _Corpus(texts, ngram=ngram, num_perm=num_perm, seed=seed)
    links = _UnionFind(len(corpus.keys))
    if threshold < 1.0:
        bands, rows = lsh_bands(threshold, num_perm)
        rejected: set[tuple[int, int]] = set()
        for bucket in corpus.buckets(bands, rows):
            members = bucket.tolist()
            for position, left in enumerate(members):
                for right in members[position + 1 :]:
                    if links.find(left) == links.find(right):
                        continue
                    pair = (left, right) if left < right else (right, left)
                    if pair in rejected:
                        continue
                    if jaccard(corpus.shingles[left], corpus.shingles[right]) >= threshold:
                        links.union(left, right)
                    else:
                        rejected.add(pair)
    roots = np.fromiter(
        (links.find(item) for item in corpus.row_to_unique.tolist()),
        dtype=np.int64,
        count=len(texts),
    )
    if roots.size == 0:
        return roots
    _, first_seen, labels = np.unique(roots, return_index=True, return_inverse=True)
    # Renumber so that group IDs follow the order in which groups first appear.
    rank = np.empty(first_seen.size, dtype=np.int64)
    rank[np.argsort(first_seen, kind="stable")] = np.arange(first_seen.size)
    return rank[labels.ravel()]


def find_cross_duplicates(
    train_texts: Sequence[str],
    test_texts: Sequence[str],
    threshold: float = 0.8,
    *,
    ngram: int = 5,
    num_perm: int = 128,
    seed: int = 0,
) -> list[NearDuplicatePair]:
    """For each test text, find one training text at or above ``threshold``.

    Returns one pair per leaked test row, with ``left`` indexing ``train_texts``
    and ``right`` indexing ``test_texts``. Exact matches (after normalization)
    are reported with similarity 1.0 and preferred over approximate ones.
    """
    threshold = _check_threshold(threshold)
    combined = list(train_texts) + list(test_texts)
    corpus = _Corpus(combined, ngram=ngram, num_perm=num_perm, seed=seed)
    n_train = len(train_texts)
    train_rows: dict[int, int] = {}
    for row in range(n_train):
        train_rows.setdefault(int(corpus.row_to_unique[row]), row)
    best: dict[int, tuple[int, float]] = {key: (row, 1.0) for key, row in train_rows.items()}
    if threshold < 1.0:
        bands, rows = lsh_bands(threshold, num_perm)
        test_unique = {int(corpus.row_to_unique[row]) for row in range(n_train, len(combined))}
        pending = test_unique - best.keys()
        checked: set[tuple[int, int]] = set()
        for bucket in corpus.buckets(bands, rows):
            if not pending:
                break
            members = bucket.tolist()
            in_train = [item for item in members if item in train_rows]
            if not in_train:
                continue
            for query in members:
                if query not in pending:
                    continue
                for candidate in in_train:
                    if (query, candidate) in checked:
                        continue
                    checked.add((query, candidate))
                    similarity = jaccard(corpus.shingles[query], corpus.shingles[candidate])
                    if similarity >= threshold:
                        best[query] = (train_rows[candidate], similarity)
                        pending.discard(query)
                        break
    result: list[NearDuplicatePair] = []
    for row in range(n_train, len(combined)):
        match = best.get(int(corpus.row_to_unique[row]))
        if match is not None:
            result.append(NearDuplicatePair(match[0], row - n_train, match[1]))
    return result
