"""Command line: ``splitproof audit`` and ``splitproof split``."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Hashable, Sequence
from typing import Any

import numpy as np

from . import __version__
from .assign import assign_groups
from .audit import leakage_report
from .io import read_rows, write_rows
from .neardup import near_duplicate_groups
from .splitters import merge_groups


def _ratios(value: str) -> dict[str, float]:
    result: dict[str, float] = {}
    try:
        for part in value.split(","):
            name, ratio = part.split("=", 1)
            name = name.strip()
            if not name or name in result:
                raise ValueError
            result[name] = float(ratio)
    except ValueError:
        raise argparse.ArgumentTypeError(
            "expected NAME=RATIO pairs such as train=0.8,validation=0.1,test=0.1"
        ) from None
    if abs(sum(result.values()) - 1.0) > 1e-6 or min(result.values()) <= 0:
        raise argparse.ArgumentTypeError("ratios must be positive and sum to 1")
    return result


def _threshold(value: str) -> float:
    number = float(value)
    if not 0.0 < number <= 1.0:
        raise argparse.ArgumentTypeError("threshold must be in (0, 1]")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="splitproof",
        description="Find near-duplicate leakage between splits, and make splits without it.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    audit = commands.add_parser(
        "audit",
        help="count test rows that have a near-duplicate in train",
        description="Count test rows that have an exact or near-duplicate in the training data. "
        "Give either one file with a split column or two files (train, test).",
    )
    audit.add_argument("data", nargs="+", help="DATA with a split column, or TRAIN TEST")
    audit.add_argument("--text-field", default="text")
    audit.add_argument("--split-field", default="split")
    audit.add_argument(
        "--train", default="train", help="split value(s) treated as training, comma separated"
    )
    audit.add_argument(
        "--test", default="test", help="split value(s) treated as test, comma separated"
    )
    audit.add_argument("--threshold", type=_threshold, default=0.8)
    audit.add_argument("--examples", type=int, default=5, help="number of examples to print")
    audit.add_argument(
        "--max-leak",
        type=float,
        default=0.0,
        help="exit 1 if the leaked share is above this (0..1)",
    )
    audit.add_argument("--json", action="store_true", help="print a JSON report instead of text")

    split = commands.add_parser(
        "split",
        help="assign rows to splits or folds, keeping near-duplicates together",
        description="Group near-duplicate texts, then assign whole groups to splits so that "
        "split sizes and label shares match the targets.",
    )
    split.add_argument("data")
    split.add_argument(
        "-o", "--output", required=True, help="output file (.jsonl, .json, .csv, .tsv)"
    )
    split.add_argument("--text-field", default="text")
    split.add_argument(
        "--label-field", help="class label, or a list of labels for multi-label data"
    )
    split.add_argument(
        "--group-field", help="rows sharing this value stay together (e.g. a conversation ID)"
    )
    target = split.add_mutually_exclusive_group()
    target.add_argument(
        "--ratios", type=_ratios, help="for example train=0.8,validation=0.1,test=0.1"
    )
    target.add_argument(
        "--folds", type=int, help="write a fold number 0..K-1 instead of split names"
    )
    split.add_argument(
        "--column", help="name of the added column (default: split, or fold with --folds)"
    )
    split.add_argument("--threshold", type=_threshold, default=0.8)
    split.add_argument("--seed", type=int, default=0)
    return parser


def _texts(rows: Sequence[dict[str, Any]], field: str, source: str) -> list[str]:
    texts = []
    for number, row in enumerate(rows, 1):
        value = row.get(field)
        if not isinstance(value, str):
            raise ValueError(f"{source}: row {number} has no string field {field!r}")
        texts.append(value)
    return texts


def _labels(rows: Sequence[dict[str, Any]], field: str) -> list[tuple[Hashable, ...]]:
    result: list[tuple[Hashable, ...]] = []
    for row in rows:
        value = row.get(field)
        if value is None or value == "":
            result.append(())
        elif isinstance(value, list):
            result.append(
                tuple(
                    json.dumps(item) if isinstance(item, (dict, list)) else item for item in value
                )
            )
        else:
            result.append((value,))
    return result


def _group_key(value: Any) -> Hashable:
    if value is None or value == "":
        return None
    return json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else str(value)


def _short(text: str, width: int = 72) -> str:
    flat = " ".join(text.split())
    return json.dumps(flat if len(flat) <= width else flat[: width - 3] + "...", ensure_ascii=False)


def _run_audit(args: argparse.Namespace) -> int:
    if len(args.data) == 2:
        train_rows, test_rows = read_rows(args.data[0]), read_rows(args.data[1])
        train = _texts(train_rows, args.text_field, args.data[0])
        test = _texts(test_rows, args.text_field, args.data[1])
    elif len(args.data) == 1:
        rows = read_rows(args.data[0])
        train_names = {name.strip() for name in args.train.split(",")}
        test_names = {name.strip() for name in args.test.split(",")}
        values = Counter(str(row.get(args.split_field)) for row in rows)
        if not any(name in values for name in test_names):
            seen = ", ".join(sorted(values)) or "none"
            raise ValueError(
                f"no rows with {args.split_field} in {sorted(test_names)}; values found: {seen}"
            )
        train = _texts(
            [r for r in rows if str(r.get(args.split_field)) in train_names],
            args.text_field,
            args.data[0],
        )
        test = _texts(
            [r for r in rows if str(r.get(args.split_field)) in test_names],
            args.text_field,
            args.data[0],
        )
    else:
        raise ValueError("give one file with a split column, or two files: TRAIN TEST")
    report = leakage_report(train, test, args.threshold)
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(f"train: {report.n_train} rows, test: {report.n_test} rows")
        print(f"near-duplicate threshold: Jaccard >= {report.threshold:.2f} on character 5-grams")
        print(f"test rows with a near-duplicate in train: {report.leaked} ({report.fraction:.1%})")
        print(f"  identical after normalization: {report.exact}")
        print(f"  similar but not identical:     {report.leaked - report.exact}")
        shown = sorted(report.pairs, key=lambda pair: (pair.similarity >= 1.0, -pair.similarity))
        if shown and args.examples > 0:
            print("examples:")
            for pair in shown[: args.examples]:
                print(f"  {pair.similarity:.2f}  test  {_short(test[pair.right])}")
                print(f"        train {_short(train[pair.left])}")
    return 1 if report.fraction > args.max_leak else 0


def _summary(
    names: Sequence[str], assigned: np.ndarray, labels: list[tuple[Hashable, ...]] | None
) -> list[str]:
    total = len(assigned)
    overall: Counter[Hashable] = Counter()
    if labels is not None:
        for row_labels in labels:
            overall.update(set(row_labels))
    few = labels is not None and 0 < len(overall) <= 4
    columns = sorted(overall, key=str) if few else []
    header = f"{'split':<12}{'rows':>8}{'share':>8}"
    header += "".join(f"{str(label)[:10]:>12}" for label in columns)
    if labels is not None and not few:
        header += f"{'worst label gap':>17}"
    lines = [header]
    for index, name in enumerate(names):
        mask = assigned == index
        count = int(mask.sum())
        line = f"{name:<12}{count:>8}{count / total:>8.1%}"
        if labels is not None:
            inside: Counter[Hashable] = Counter()
            for row in np.flatnonzero(mask).tolist():
                inside.update(set(labels[row]))
            if few:
                line += "".join(f"{inside[label] / max(count, 1):>12.1%}" for label in columns)
            else:
                gap = max(
                    (
                        abs(inside[label] / max(count, 1) - overall[label] / total)
                        for label in overall
                    ),
                    default=0.0,
                )
                line += f"{gap * 100:>14.2f} pp"
        lines.append(line)
    if few:
        lines.append(
            f"{'(all)':<12}{total:>8}{1:>8.1%}"
            + "".join(f"{overall[label] / total:>12.1%}" for label in columns)
        )
    return lines


def _run_split(args: argparse.Namespace) -> int:
    rows = read_rows(args.data)
    texts = _texts(rows, args.text_field, args.data)
    neardup = near_duplicate_groups(texts, args.threshold)
    user_groups = (
        [_group_key(row.get(args.group_field)) for row in rows] if args.group_field else None
    )
    merged = merge_groups(neardup, user_groups)
    labels = _labels(rows, args.label_field) if args.label_field else None
    if args.folds is not None:
        if args.folds < 2:
            raise ValueError("--folds must be at least 2")
        names = [str(fold) for fold in range(args.folds)]
        ratios = [1.0 / args.folds] * args.folds
        column = args.column or "fold"
    else:
        targets = args.ratios or {"train": 0.8, "test": 0.2}
        names, ratios = list(targets), list(targets.values())
        column = args.column or "split"
    n_groups = int(np.max(merged)) + 1
    if n_groups < len(names):
        raise ValueError(f"only {n_groups} groups, cannot fill {len(names)} splits")
    assigned = assign_groups(merged, ratios, labels, seed=args.seed)
    sizes = np.bincount(merged)
    print(f"{len(rows)} rows -> {n_groups} groups (near-duplicate threshold {args.threshold:.2f})")
    print(
        f"rows sharing a group with another row: {int(sizes[sizes > 1].sum())}, "
        f"largest group: {int(np.max(sizes))} rows"
    )
    for line in _summary(names, assigned, labels):
        print(line)
    for row, index in zip(rows, assigned.tolist()):
        row[column] = int(names[index]) if args.folds is not None else names[index]
    write_rows(rows, args.output)
    print(f"wrote {args.output}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "audit":
            return _run_audit(args)
        return _run_split(args)
    except (ValueError, OSError, TypeError) as error:
        print(f"splitproof: error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
