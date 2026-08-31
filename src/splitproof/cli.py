"""Command-line interface for reproducible splitting workflows."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from itertools import combinations
from pathlib import Path

from .assigners import (
    ALGORITHM_VERSION,
    balanced_group_split,
    hash_split,
    stratified_group_split,
)
from .diagnostics import diagnose
from .io import load_assignments, load_records, save_assignments
from .kfold import assign_kfold
from .manifest import create_manifest, load_manifest, save_manifest, verify_manifest
from .reporting import report_json, report_markdown


def _ratios(value: str) -> dict[str, float]:
    try:
        result: dict[str, float] = {}
        for part in value.split(","):
            name, ratio = part.split("=", 1)
            normalized_name = name.strip()
            if normalized_name in result:
                raise argparse.ArgumentTypeError(f"duplicate split ratio name {normalized_name!r}")
            result[normalized_name] = float(ratio)
    except argparse.ArgumentTypeError:
        raise
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("expected NAME=RATIO pairs separated by commas") from error
    if not result:
        raise argparse.ArgumentTypeError("at least one ratio is required")
    return result


def _fields(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("input", type=Path, help="JSON array or JSONL dataset")
    parser.add_argument("--id-field", default="id")
    parser.add_argument("--group-field", default="group")
    parser.add_argument("--label-field", default="label")


def build_parser() -> argparse.ArgumentParser:
    """Build the public argument parser."""
    parser = argparse.ArgumentParser(
        prog="splitproof",
        description="Reproducible NLP dataset splits with verifiable manifests.",
    )
    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")
    commands = parser.add_subparsers(dest="command", required=True)

    split = commands.add_parser("split", help="create train/validation/test assignments")
    _fields(split)
    split.add_argument(
        "--ratios",
        type=_ratios,
        default={"train": 0.8, "validation": 0.1, "test": 0.1},
    )
    split.add_argument(
        "--algorithm",
        choices=("hash", "group", "stratified-group"),
        default="stratified-group",
    )
    split.add_argument("--seed", default="0")
    split.add_argument("--assignments", type=Path, required=True)
    split.add_argument("--manifest", type=Path, required=True)

    kfold = commands.add_parser("kfold", help="create group-aware k-fold assignments")
    _fields(kfold)
    kfold.add_argument("--folds", type=int, default=5)
    kfold.add_argument("--seed", default="0")
    kfold.add_argument("--stratified", action="store_true")
    kfold.add_argument("--assignments", type=Path, required=True)
    kfold.add_argument("--manifest", type=Path, required=True)

    verify = commands.add_parser("verify", help="verify a manifest against current data")
    _fields(verify)
    verify.add_argument("--manifest", type=Path, required=True)
    verify.add_argument(
        "--assignments",
        type=Path,
        help="also compare an external assignments JSONL file with the manifest",
    )

    inspect = commands.add_parser("inspect", help="inspect manifest diagnostics")
    _fields(inspect)
    inspect.add_argument("--manifest", type=Path, required=True)
    inspect.add_argument("--format", choices=("json", "markdown"), default="markdown")
    inspect.add_argument("--output", type=Path)
    return parser


def _load(args: argparse.Namespace):  # type: ignore[no-untyped-def]
    return load_records(
        args.input,
        id_field=args.id_field,
        group_field=args.group_field,
        label_field=args.label_field,
    )


def _paths_collide(left: Path, right: Path) -> bool:
    """Return whether two CLI paths resolve to the same filesystem entry."""
    if left.resolve(strict=False) == right.resolve(strict=False):
        return True
    if left.exists() and right.exists():
        return left.samefile(right)
    return False


def _require_distinct_paths(**paths: Path | None) -> None:
    """Reject ambiguous CLI roles before any output file can be written."""
    present = [(name, path) for name, path in paths.items() if path is not None]
    for (left_name, left), (right_name, right) in combinations(present, 2):
        assert left is not None and right is not None
        if _paths_collide(left, right):
            raise ValueError(f"{left_name} and {right_name} paths must be different")


def _run_split(args: argparse.Namespace) -> int:
    _require_distinct_paths(
        input=args.input,
        assignments=args.assignments,
        manifest=args.manifest,
    )
    records = _load(args)
    algorithms = {
        "hash": hash_split,
        "group": balanced_group_split,
        "stratified-group": stratified_group_split,
    }
    assignments = algorithms[args.algorithm](records, args.ratios, seed=args.seed)
    manifest = create_manifest(
        records,
        assignments,
        algorithm=args.algorithm,
        algorithm_version=ALGORITHM_VERSION,
        seed=args.seed,
        ratios=args.ratios,
    )
    save_assignments(assignments, args.assignments)
    save_manifest(manifest, args.manifest)
    print(report_markdown(diagnose(records, assignments, args.ratios)))
    return 0


def _run_kfold(args: argparse.Namespace) -> int:
    _require_distinct_paths(
        input=args.input,
        assignments=args.assignments,
        manifest=args.manifest,
    )
    records = _load(args)
    assignments = assign_kfold(records, args.folds, seed=args.seed, stratified=args.stratified)
    ratios = {f"fold-{index}": 1 / args.folds for index in range(args.folds)}
    manifest = create_manifest(
        records,
        assignments,
        algorithm="stratified-group-kfold" if args.stratified else "group-kfold",
        algorithm_version=ALGORITHM_VERSION,
        seed=args.seed,
        ratios=ratios,
        metadata={"folds": args.folds},
    )
    save_assignments(assignments, args.assignments)
    save_manifest(manifest, args.manifest)
    print(report_markdown(diagnose(records, assignments, ratios)))
    return 0


def _run_verify(args: argparse.Namespace) -> int:
    _require_distinct_paths(
        input=args.input,
        manifest=args.manifest,
        assignments=args.assignments,
    )
    external = load_assignments(args.assignments) if args.assignments else None
    errors = verify_manifest(load_manifest(args.manifest), _load(args), external)
    return _print_verification(errors)


def _print_verification(errors: tuple[str, ...]) -> int:
    if errors:
        print("Verification failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print(
        "Verification passed: checksum, dataset, coverage, assignments, "
        "and algorithm constraints match."
    )
    return 0


def _run_inspect(args: argparse.Namespace) -> int:
    _require_distinct_paths(
        input=args.input,
        manifest=args.manifest,
        output=args.output,
    )
    records = _load(args)
    manifest = load_manifest(args.manifest)
    errors = verify_manifest(manifest, records)
    if errors:
        return _print_verification(errors)
    report = diagnose(records, manifest.assignments, manifest.ratios)
    rendered = report_json(report) + "\n" if args.format == "json" else report_markdown(report)
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and convert validation errors into concise exit status 2."""
    args = build_parser().parse_args(argv)
    runners = {
        "split": _run_split,
        "kfold": _run_kfold,
        "verify": _run_verify,
        "inspect": _run_inspect,
    }
    try:
        return runners[args.command](args)
    except (ValueError, OSError, TypeError, KeyError) as error:
        print(f"splitproof: error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
