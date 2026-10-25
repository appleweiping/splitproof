import numpy as np
import pytest

from splitproof import find_cross_duplicates, near_duplicate_groups, normalize_text
from splitproof.neardup import MinHasher, jaccard, lsh_bands, shingle_hashes, shingle_sets


def test_normalize_text_folds_case_width_and_punctuation():
    assert normalize_text("  Hello,   WORLD!!  ") == "hello world"
    assert normalize_text("ＦＵＬＬ　ｗｉｄｔｈ") == "full width"
    assert normalize_text("snake_case-and.dots") == "snake case and dots"
    with pytest.raises(TypeError):
        normalize_text(None)  # type: ignore[arg-type]


def test_shingles_are_sets_and_short_texts_have_one():
    assert shingle_hashes("", 5).size == 0
    assert shingle_hashes("abc", 5).size == 1
    assert shingle_hashes("aaaaaaaa", 5).size == 1  # every 5-gram is "aaaaa"
    assert shingle_hashes("abcdef", 5).size == 2
    values = shingle_hashes("the quick brown fox", 3)
    assert np.all(np.diff(values.astype(np.int64)) > 0)


def test_batch_shingling_matches_single_text_shingling():
    texts = ["", "ab", "hello world", "hello world again", "abcde"]
    batch = shingle_sets(texts, 4)
    for text, values in zip(texts, batch):
        assert np.array_equal(values, shingle_hashes(text, 4))


def test_jaccard_is_exact():
    left = shingle_hashes("abcdefgh", 3)  # abc bcd cde def efg fgh
    right = shingle_hashes("abcdefxy", 3)  # abc bcd cde def efx fxy
    assert jaccard(left, right) == pytest.approx(4 / 8)
    assert jaccard(left, left) == 1.0
    empty = np.empty(0, dtype=np.uint32)
    assert jaccard(empty, empty) == 1.0


def test_lsh_band_layout_meets_recall_at_threshold():
    for threshold in (0.5, 0.7, 0.8, 0.9, 0.95):
        bands, rows = lsh_bands(threshold, 128)
        assert bands * rows <= 128
        assert 1 - (1 - threshold**rows) ** bands >= 0.98
    with pytest.raises(ValueError):
        lsh_bands(0.0, 128)


def test_minhash_estimates_similarity():
    left = shingle_hashes(normalize_text("the cat sat on the mat and looked at the dog " * 3), 5)
    right = shingle_hashes(normalize_text("the cat sat on the mat and stared at the dog " * 3), 5)
    signatures = MinHasher(256, seed=1).signatures([left, right])
    estimate = float(np.mean(signatures[0] == signatures[1]))
    assert abs(estimate - jaccard(left, right)) < 0.12


def test_near_duplicate_groups_link_variants_and_chains():
    texts = [
        "Free entry in a weekly competition to win FA Cup final tickets!",
        "free entry in a weekly competition to win fa cup final tickets",
        "Ok lar... Joking wif u oni...",
        "FREE entry in a weekly competition to win FA Cup final tkts!!",
        "Something completely different about the weather today.",
        "",
        "",
    ]
    groups = near_duplicate_groups(texts, 0.7)
    assert groups.tolist()[:5] == [0, 0, 1, 0, 2]
    assert groups[5] == groups[6] == 3  # empty texts are identical to each other


def test_threshold_one_only_groups_identical_normalized_texts():
    texts = ["Hello there!", "hello there", "hello there friend"]
    assert near_duplicate_groups(texts, 1.0).tolist() == [0, 0, 1]


def test_near_duplicate_groups_validates_threshold():
    with pytest.raises(ValueError):
        near_duplicate_groups(["a"], 1.5)
    with pytest.raises(TypeError):
        near_duplicate_groups(["a"], True)
    assert near_duplicate_groups([], 0.8).size == 0


def test_random_texts_are_not_grouped():
    rng = np.random.RandomState(0)
    words = [f"w{i}" for i in range(500)]
    texts = [" ".join(rng.choice(words, 12)) for _ in range(300)]
    groups = near_duplicate_groups(texts, 0.8)
    assert groups.max() + 1 == 300


def test_find_cross_duplicates_reports_each_leaked_test_row_once():
    train = ["the cat sat on the mat", "a completely unrelated sentence", "the cat sat on the mat"]
    test = ["The cat sat on the mat!", "the cat sat on a mat", "nothing like the others here"]
    pairs = find_cross_duplicates(train, test, 0.4)
    by_test = {pair.right: pair for pair in pairs}
    assert set(by_test) == {0, 1}
    assert by_test[0].similarity == 1.0 and by_test[0].left == 0
    assert by_test[1].similarity == pytest.approx(11 / 23)
