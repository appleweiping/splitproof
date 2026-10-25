# splitproof

splitproof makes train/test splits and cross-validation folds for text datasets that keep
near-duplicate texts on the same side, so the test set does not quietly contain copies of the
training data.

```console
$ splitproof audit emotion-train.parquet emotion-test.parquet --examples 3
train: 16000 rows, test: 2000 rows
near-duplicate threshold: Jaccard >= 0.80 on character 5-grams
test rows with a near-duplicate in train: 16 (0.8%)
  identical after normalization: 11
  similar but not identical:     5
examples:
  0.93  test  "i get scared i feel ignored i feel happy i get silly i choke on my ow..."
        train "i get scared i feel ignored i feel happy i get silly i choke on my ow..."
  0.89  test  "i did not care much about the number of viewers and the viewer rating..."
        train "i did not care much about the number of viewers and the viewer rating..."
  0.87  test  "i am feeling very petty right now"
        train "im feeling very petty right now"
```

That is the official train/test split of [dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion).
All 11 identical pairs have a *different* label in train than in test.

## Why

Public text datasets are full of repeats: spam templates, retweets, boilerplate replies, the
same Reddit comment posted twice. A random split scatters the copies across train and test, the
model gets credit for remembering them, and the score you report is a little too good.

On the [SMS Spam Collection](https://huggingface.co/datasets/ucirvine/sms_spam) (5,574 messages,
only 5,159 distinct after lowercasing and collapsing whitespace) a random 80/20 split puts a near-duplicate of a training
message into **14.1%** of the test set (13.1–15.2% over five seeds); 11.8% are identical after
normalization. A splitproof split has none:

| 80/20 split, 5 seeds | test rows with a near-duplicate in train | identical after normalization |
|---|---:|---:|
| random | 14.1% (13.1–15.2%) | 11.8% |
| splitproof | 0.0% | 0.0% |

With TF-IDF + logistic regression and 5-fold cross-validation, spam F1 drops from
0.934 ± 0.013 (`KFold(shuffle=True)`) to 0.922 ± 0.013 (`NearDupKFold`). Not dramatic, but the
second number is the one I would believe.

## Install

```bash
pip install "splitproof[sklearn,parquet] @ git+https://github.com/appleweiping/splitproof"
```

The only required dependency is NumPy. The `sklearn` extra is for the examples below and
`parquet` adds pyarrow for reading `.parquet` files. Python 3.10 or newer.

## Use it with scikit-learn

`NearDupKFold` follows the scikit-learn splitter protocol, so it drops into `cross_val_score`,
`cross_validate`, `GridSearchCV` and friends:

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from splitproof import NearDupKFold

model = make_pipeline(TfidfVectorizer(), LogisticRegression(max_iter=1000))
scores = cross_val_score(model, texts, labels, cv=NearDupKFold(n_splits=5, threshold=0.8))
```

- `X` can be the list of texts itself (as in a TF-IDF pipeline). If `X` is a feature matrix,
  pass the raw texts with `NearDupKFold(texts=texts)`.
- `y` can be class labels, a multi-label indicator matrix, or a list of label lists. Folds are
  balanced so every label's share in each fold stays close to the full dataset.
- `groups` (for example conversation or author IDs) are merged with the near-duplicate groups:
  rows linked by either never end up on different sides.

For one fixed split instead of folds:

```python
from splitproof import assign_splits, leakage_report

ratios = {"train": 0.8, "validation": 0.1, "test": 0.1}
df["split"] = assign_splits(df.text, df.label, ratios=ratios)

# check any existing split
report = leakage_report(train_texts, test_texts, threshold=0.8)
print(report.leaked, report.fraction)
```

`near_duplicate_groups(texts)` returns the group ID of every text if you want to feed it to
`GroupKFold` or `StratifiedGroupKFold` yourself.

## Command line

`splitproof split` adds a `split` (or `fold`) column; `splitproof audit` checks a split. Both read
and write `.jsonl`, `.json`, `.csv`, `.tsv` and `.parquet`.

```console
$ splitproof split sms.parquet --text-field sms --label-field label -o sms.split.jsonl
5574 rows -> 5031 groups (near-duplicate threshold 0.80)
rows sharing a group with another row: 890, largest group: 30 rows
split           rows   share           0           1
train           4459   80.0%       86.6%       13.4%
test            1115   20.0%       86.6%       13.4%
(all)           5574  100.0%       86.6%       13.4%
wrote sms.split.jsonl
$ splitproof audit sms.split.jsonl --text-field sms
train: 4459 rows, test: 1115 rows
near-duplicate threshold: Jaccard >= 0.80 on character 5-grams
test rows with a near-duplicate in train: 0 (0.0%)
  identical after normalization: 0
  similar but not identical:     0
```

Useful options: `--ratios train=0.8,validation=0.1,test=0.1`, `--folds 5`,
`--group-field conversation_id`, `--threshold 0.9`. `audit` takes one file with a split column or
two files (train, test), exits with status 1 when the leaked share is above `--max-leak`
(default 0), and prints JSON with `--json`, so it can run in CI.

## How it works

1. Each text is normalized (NFKC, case-folded, punctuation turned into spaces) and cut into
   character 5-grams.
2. MinHash signatures (128 hash functions) and LSH banding propose candidate pairs. The band
   layout is picked so a pair exactly at the threshold is found with at least 98% probability.
3. Every candidate is checked with the exact Jaccard similarity of the two 5-gram sets. Pairs at
   or above the threshold are linked, and groups are the connected components.
4. Whole groups are assigned to splits: largest and rarest-label groups first, each to the split
   where it reduces the squared gap between actual and target shares (rows and every label) the
   most, then single-group moves while they keep lowering that gap.

Character 5-grams catch reworded copies, typos and changed numbers, not paraphrases. If two texts
mean the same thing with different words, splitproof will not see it.

## Benchmark: 5 folds on GoEmotions

The train split of [GoEmotions](https://huggingface.co/datasets/google-research-datasets/go_emotions)
(43,410 Reddit comments, 28 labels, 1–3 labels each). "Label-share deviation" is
|share of a label in a fold − share in the whole dataset| / share in the whole dataset, over all
folds and labels.

| splitter | mean label-share deviation | worst | test rows with a near-duplicate in train (identical) | split time |
|---|---:|---:|---:|---:|
| scikit-learn `KFold(shuffle=True)` | 5.90% | 35.1% | 821 (741) | 0.2 s |
| iterative-stratification `MultilabelStratifiedKFold` | 0.55% | 4.2% | 834 (760) | 7.1 s |
| scikit-learn `StratifiedGroupKFold`, near-dup groups, first label only | 2.53% | 21.6% | 0 (0) | 115 s + grouping |
| splitproof `NearDupKFold` | 0.38% | 3.9% | 0 (0) | 2.5 s + grouping |

Finding the near-duplicate groups took 43 s on a 16-thread desktop that was busy with other jobs
(about 12 s when it was quieter). Leakage is measured with splitproof's own detector at Jaccard
0.8, so the zero for the two group-aware rows is by construction; the "identical" counts do not
depend on the detector.

Reproduce both benchmarks (downloads about 4 MB from the Hugging Face Hub, cached in
`~/.cache/splitproof`):

```bash
git clone https://github.com/appleweiping/splitproof && cd splitproof
pip install -e ".[bench]"
python benchmarks/leakage.py      # SMS Spam, emotion
python benchmarks/multilabel.py   # GoEmotions
```

Raw numbers are in [`benchmarks/results/`](benchmarks/results/).

## Why not ...

- **`GroupKFold` / `StratifiedGroupKFold`**: they are the right tools once you have groups.
  splitproof's job is to make the groups from the text. `StratifiedGroupKFold` also takes only
  one label per row, and its assignment step was about 45× slower than `NearDupKFold`'s on
  GoEmotions.
- **iterative-stratification**: good multi-label balance, but no notion of groups, so duplicates
  land on both sides.
- **Deduplicating first** (datasketch, text-dedup, `drop_duplicates`): fine if you want the copies
  gone. Often you don't: the duplicates are part of the real distribution, or they carry
  conflicting labels you want to look at. Grouping keeps every row and still makes the test
  set honest.

## Limitations

- Everything is in memory. Tens of thousands of short texts take tens of seconds; I have not
  tried millions.
- LSH can miss a pair that is barely above the threshold (at most 2% at exactly the threshold).
- Long texts that share a large boilerplate block (headers, signatures, licenses) will be grouped
  together. Strip the boilerplate first or raise the threshold.
- Very large duplicate clusters make exact split ratios impossible; the summary table shows what
  you actually got.

Related tools: [corpusledger](https://github.com/appleweiping/corpusledger) diffs two versions of
a dataset, [turnscope](https://github.com/appleweiping/turnscope) lints chat fine-tuning data,
[metricguard](https://github.com/appleweiping/metricguard) and
[promptwitness](https://github.com/appleweiping/promptwitness) cover metric edge cases and prompt
changes.

MIT License. Version 0.2 (weighted splits with checksummed manifests) is under the `v0.2.0` tag.
