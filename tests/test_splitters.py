import numpy as np
import pytest

from splitproof import NearDupKFold, assign_splits, leakage_report, merge_groups
from splitproof.splitters import as_texts, label_sets

BASE = [
    "I loved this film, the acting was great",
    "Terrible plot and the pacing was awful",
    "An okay movie, nothing special about it",
    "Best soundtrack I have heard in years",
    "The cinematography alone is worth the ticket",
    "I fell asleep halfway through, so boring",
]


def corpus(copies: int = 4) -> tuple[list[str], list[int]]:
    """Each base review appears ``copies`` times with small edits, plus unique filler."""
    texts, labels = [], []
    for index, text in enumerate(BASE):
        for copy in range(copies):
            texts.append(text + "!" * copy if copy % 2 else text.upper())
            labels.append(index % 2)
    for index in range(60):
        texts.append(f"unique review number {index} about film {index * 7919 % 1000}")
        labels.append(index % 2)
    return texts, labels


def test_folds_keep_near_duplicates_together():
    texts, labels = corpus()
    cv = NearDupKFold(n_splits=4, threshold=0.8)
    assert cv.get_n_splits() == 4
    seen = []
    for train, test in cv.split(texts, labels):
        assert len(set(train) & set(test)) == 0
        report = leakage_report([texts[i] for i in train], [texts[i] for i in test], 0.8)
        assert report.leaked == 0
        seen.extend(test.tolist())
    assert sorted(seen) == list(range(len(texts)))


def test_user_groups_are_merged_with_near_duplicate_groups():
    texts = [f"distinct text {i} " + "x" * (i % 7) for i in range(40)]
    groups = [i // 10 for i in range(40)]  # four conversations of ten rows
    folds = NearDupKFold(n_splits=4).fold_ids(texts, groups=groups)
    for conversation in range(4):
        assert len(set(folds[conversation * 10 : conversation * 10 + 10].tolist())) == 1


def test_texts_argument_for_feature_matrices():
    texts, labels = corpus()
    features = np.zeros((len(texts), 3))
    cv = NearDupKFold(n_splits=3, texts=texts)
    assert sum(len(test) for _, test in cv.split(features, labels)) == len(texts)
    with pytest.raises(ValueError):
        list(cv.split(features[:-1], labels))
    with pytest.raises(TypeError):
        list(NearDupKFold(n_splits=3).split(features, labels))


def test_too_many_folds_for_groups():
    with pytest.raises(ValueError):
        list(NearDupKFold(n_splits=3).split(["same text"] * 10))
    with pytest.raises(ValueError):
        NearDupKFold(n_splits=1)


def test_assign_splits_names_rows():
    texts, labels = corpus()
    names = assign_splits(texts, labels, ratios={"train": 0.75, "test": 0.25})
    assert set(names.tolist()) == {"train", "test"}
    train = [t for t, n in zip(texts, names) if n == "train"]
    test = [t for t, n in zip(texts, names) if n == "test"]
    assert leakage_report(train, test, 0.8).leaked == 0
    assert abs(len(test) / len(texts) - 0.25) < 0.05


def test_merge_groups_ignores_missing_values():
    merged = merge_groups([0, 1, 2, 3], ["a", None, "a", None], [None, 5, None, float("nan")])
    assert merged[0] == merged[2]
    assert len({merged[0], merged[1], merged[3]}) == 3
    with pytest.raises(ValueError):
        merge_groups([0, 1], [0])
    with pytest.raises(ValueError):
        merge_groups(None)


def test_label_sets_accepts_common_label_formats():
    assert label_sets(None) is None
    assert label_sets(["a", "b"]) == [("a",), ("b",)]
    assert label_sets(np.array([1, 0])) == [(1,), (0,)]
    assert label_sets([["a", "b"], []]) == [("a", "b"), ()]
    assert label_sets(np.array([[1, 0, 1], [0, 0, 0]])) == [(0, 2), ()]
    with pytest.raises(ValueError):
        label_sets([["a"], "b"])


def test_as_texts_rejects_non_strings():
    assert as_texts(np.array(["a", "b"])) == ["a", "b"]
    with pytest.raises(TypeError):
        as_texts([1, 2])


def test_works_with_scikit_learn_cross_validation():
    sklearn = pytest.importorskip("sklearn")
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV, cross_val_score
    from sklearn.pipeline import make_pipeline

    assert sklearn
    texts, labels = corpus()
    model = make_pipeline(TfidfVectorizer(), LogisticRegression())
    scores = cross_val_score(model, texts, labels, cv=NearDupKFold(n_splits=3))
    assert scores.shape == (3,)
    search = GridSearchCV(model, {"logisticregression__C": [0.1, 1.0]}, cv=NearDupKFold(3))
    search.fit(texts, labels)
    assert search.best_params_
