"""Command-line interface for reproducible splitting workflows."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict
from datetime import datetime, timedelta
from itertools import combinations
from pathlib import Path

from . import __version__
from .assigners import (
    ALGORITHM_VERSION,
    balanced_group_split,
    exact_group_split,
    hash_split,
    stratified_group_split,
)
from .comparison import compare_manifests
from .diagnostics import diagnose
from .io import iter_records, load_assignments, load_records, save_assignments
from .kfold import assign_kfold
from .leakage import audit_leakage, audit_near_duplicates
from .manifest import (
    create_manifest,
    load_manifest,
    migrate_manifest,
    save_manifest,
    verify_manifest,
)
from .materialize import write_materialized
from .repeat import (
    holdout_stability_report,
    repeated_group_holdout,
    repeated_kfold,
    stability_report,
)
from .reporting import report_json, report_markdown
from .streaming import verify_hash_split_stream, write_hash_split_stream
from .temporal import TimeInterval, purged_kfold


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


def _minimum_counts(values: list[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        try:
            name, count = value.split("=", 1)
            normalized_name = name.strip()
            parsed = int(count)
        except (TypeError, ValueError) as error:
            raise argparse.ArgumentTypeError("expected NAME=COUNT pairs") from error
        if not normalized_name or normalized_name in result or parsed < 1:
            raise argparse.ArgumentTypeError(
                "minimum counts require unique names and positive counts"
            )
        result[normalized_name] = parsed
    return result


def _fields(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("input", type=Path, help="JSON array or JSONL dataset")
    parser.add_argument("--id-field", default="id")
    parser.add_argument("--group-field", default="group")
    parser.add_argument("--label-field", default="label")
    parser.add_argument("--weight-field", default="weight")
    parser.add_argument("--group-weight-field", default="group_weight")
    parser.add_argument(
        "--fingerprint-field",
        action="append",
        default=[],
        help="repeat to include a top-level payload field in the dataset fingerprint",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the public argument parser."""
    parser = argparse.ArgumentParser(
        prog="splitproof",
        description="Reproducible NLP dataset splits with verifiable manifests.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
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
        choices=("hash", "group", "stratified-group", "exact-group"),
        default="stratified-group",
    )
    split.add_argument("--seed", default="0")
    split.add_argument("--assignments", type=Path, required=True)
    split.add_argument("--manifest", type=Path, required=True)
    split.add_argument(
        "--max-local-iterations",
        type=int,
        help="local-improvement limit for group algorithms; default is deterministic auto",
    )
    split.add_argument(
        "--max-groups",
        type=int,
        default=12,
        help="exact-group search limit (default: 12)",
    )
    split.add_argument(
        "--stratified",
        action="store_true",
        help="balance labels during exact-group optimization",
    )

    kfold = commands.add_parser("kfold", help="create group-aware k-fold assignments")
    _fields(kfold)
    kfold.add_argument("--folds", type=int, default=5)
    kfold.add_argument("--seed", default="0")
    kfold.add_argument("--stratified", action="store_true")
    kfold.add_argument("--assignments", type=Path, required=True)
    kfold.add_argument("--manifest", type=Path, required=True)
    kfold.add_argument(
        "--max-local-iterations",
        type=int,
        help="local-improvement limit; use 0 for greedy-only placement",
    )

    verify = commands.add_parser("verify", help="verify a manifest against current data")
    _fields(verify)
    verify.add_argument("--manifest", type=Path, required=True)
    verify.add_argument(
        "--assignments",
        type=Path,
        help="also compare an external assignments JSONL file with the manifest",
    )
    migrate = commands.add_parser(
        "migrate-manifest", help="migrate a verified schema-v1 manifest to schema v2"
    )
    _fields(migrate)
    migrate.add_argument("--manifest", type=Path, required=True)
    migrate.add_argument("--output", type=Path, required=True)

    inspect = commands.add_parser("inspect", help="inspect manifest diagnostics")
    _fields(inspect)
    inspect.add_argument("--manifest", type=Path, required=True)
    inspect.add_argument("--format", choices=("json", "markdown"), default="markdown")
    inspect.add_argument("--output", type=Path)
    compare = commands.add_parser(
        "compare", help="compare two split manifests for assignment drift"
    )
    compare.add_argument("before", type=Path)
    compare.add_argument("after", type=Path)
    compare.add_argument("--output", type=Path)
    temporal = commands.add_parser("temporal-kfold", help="create purged time-interval folds")
    _fields(temporal)
    temporal.add_argument("--start-field", default="start")
    temporal.add_argument("--end-field", default="end")
    temporal.add_argument("--folds", type=int, default=5)
    temporal.add_argument("--gap-seconds", type=int, default=0)
    temporal.add_argument("--embargo-seconds", type=int, default=0)
    temporal.add_argument("--allow-group-overlap", action="store_true")
    temporal.add_argument("--output", type=Path)
    repeat = commands.add_parser(
        "repeat", help="create repeated k-fold assignments and stability evidence"
    )
    _fields(repeat)
    repeat.add_argument("--folds", type=int, default=5)
    repeat.add_argument("--repeats", type=int, default=3)
    repeat.add_argument("--seed", default="0")
    repeat.add_argument("--stratified", action="store_true")
    repeat.add_argument("--output", type=Path)
    holdout = commands.add_parser(
        "repeat-holdout", help="create repeated group-preserving named holdout assignments"
    )
    _fields(holdout)
    holdout.add_argument("--ratios", type=_ratios, default={"train": 0.8, "test": 0.2})
    holdout.add_argument("--repeats", type=int, default=3)
    holdout.add_argument("--seed", default="0")
    holdout.add_argument("--stratified", action="store_true")
    holdout.add_argument(
        "--minimum-count",
        action="append",
        default=[],
        metavar="SPLIT=COUNT",
        help="require at least COUNT records in SPLIT (repeatable)",
    )
    holdout.add_argument("--output", type=Path)

    materialize = commands.add_parser(
        "materialize", help="write verified split payloads as deterministic JSONL files"
    )
    _fields(materialize)
    materialize.add_argument("--manifest", type=Path, required=True)
    materialize.add_argument("--output-dir", type=Path, required=True)
    hash_stream = commands.add_parser(
        "hash-stream", help="stream a JSONL record-hash split without retaining payloads"
    )
    _fields(hash_stream)
    hash_stream.add_argument("--ratios", type=_ratios, required=True)
    hash_stream.add_argument("--seed", default="0")
    hash_stream.add_argument("--assignments", type=Path, required=True)
    hash_stream.add_argument("--report", type=Path)
    hash_verify = commands.add_parser(
        "hash-stream-verify", help="verify a streamed assignment file against its report"
    )
    hash_verify.add_argument("assignments", type=Path)
    hash_verify.add_argument("report", type=Path)
    leakage = commands.add_parser(
        "leakage-audit", help="find normalized exact duplicates across split boundaries"
    )
    _fields(leakage)
    leakage.add_argument("--assignments", type=Path, required=True)
    leakage.add_argument("--fields", default="text,content,question,answer")
    leakage.add_argument("--min-length", type=int, default=1)
    leakage.add_argument("--max-findings", type=int, default=10_000)
    leakage.add_argument("--output", type=Path)
    near = commands.add_parser(
        "near-duplicate-audit", help="find token-shingle near duplicates across splits"
    )
    _fields(near)
    near.add_argument("--assignments", type=Path, required=True)
    near.add_argument("--field", default="text")
    near.add_argument("--threshold", type=float, default=0.8)
    near.add_argument("--min-tokens", type=int, default=3)
    near.add_argument("--max-pairs", type=int, default=10_000)
    near.add_argument("--output", type=Path)
    return parser


def _load(args: argparse.Namespace):  # type: ignore[no-untyped-def]
    return load_records(
        args.input,
        id_field=args.id_field,
        group_field=args.group_field,
        label_field=args.label_field,
        weight_field=args.weight_field,
        group_weight_field=args.group_weight_field,
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
        if _paths_collide(left, right):
            raise ValueError(f"{left_name} and {right_name} paths must be different")


def _run_split(args: argparse.Namespace) -> int:
    _require_distinct_paths(
        input=args.input,
        assignments=args.assignments,
        manifest=args.manifest,
    )
    records = _load(args)
    if args.stratified and args.algorithm not in {"exact-group", "stratified-group"}:
        raise ValueError("--stratified is only valid with exact-group or stratified-group")
    if args.algorithm == "hash":
        if args.max_local_iterations is not None:
            raise ValueError("--max-local-iterations is only valid for group algorithms")
        assignments = hash_split(records, args.ratios, seed=args.seed)
        optimizer = "record-hash-v1"
    else:
        if args.algorithm == "exact-group":
            if args.max_local_iterations is not None:
                raise ValueError("--max-local-iterations is not valid for exact-group")
            assignments = exact_group_split(
                records,
                args.ratios,
                seed=args.seed,
                stratified=args.stratified,
                max_groups=args.max_groups,
            )
            optimizer = "exhaustive-stratified-v1" if args.stratified else "exhaustive-v1"
        else:
            algorithm = (
                balanced_group_split if args.algorithm == "group" else stratified_group_split
            )
            assignments = algorithm(
                records,
                args.ratios,
                seed=args.seed,
                max_local_iterations=args.max_local_iterations,
            )
            optimizer = "greedy-local-v3"
    manifest = create_manifest(
        records,
        assignments,
        algorithm=args.algorithm,
        algorithm_version=ALGORITHM_VERSION,
        seed=args.seed,
        ratios=args.ratios,
        metadata={
            "optimizer": optimizer,
            "max_local_iterations": args.max_local_iterations,
        },
        fingerprint_fields=args.fingerprint_field or None,
    )
    save_assignments(assignments, args.assignments)
    save_manifest(manifest, args.manifest)
    print(
        report_markdown(
            diagnose(
                records,
                assignments,
                args.ratios,
                include_label_balance=args.algorithm == "stratified-group",
            )
        )
    )
    return 0


def _run_kfold(args: argparse.Namespace) -> int:
    _require_distinct_paths(
        input=args.input,
        assignments=args.assignments,
        manifest=args.manifest,
    )
    records = _load(args)
    assignments = assign_kfold(
        records,
        args.folds,
        seed=args.seed,
        stratified=args.stratified,
        max_local_iterations=args.max_local_iterations,
    )
    ratios = {f"fold-{index}": 1 / args.folds for index in range(args.folds)}
    manifest = create_manifest(
        records,
        assignments,
        algorithm="stratified-group-kfold" if args.stratified else "group-kfold",
        algorithm_version=ALGORITHM_VERSION,
        seed=args.seed,
        ratios=ratios,
        metadata={
            "folds": args.folds,
            "optimizer": "greedy-local-v3",
            "max_local_iterations": args.max_local_iterations,
        },
        fingerprint_fields=args.fingerprint_field or None,
    )
    save_assignments(assignments, args.assignments)
    save_manifest(manifest, args.manifest)
    print(
        report_markdown(
            diagnose(
                records,
                assignments,
                ratios,
                include_label_balance=args.stratified,
            )
        )
    )
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


def _run_migrate(args: argparse.Namespace) -> int:
    _require_distinct_paths(input=args.input, manifest=args.manifest, output=args.output)
    migrated = migrate_manifest(load_manifest(args.manifest), _load(args))
    save_manifest(migrated, args.output)
    print(
        json.dumps(
            {
                "schema_version": migrated.schema_version,
                "data_fingerprint": migrated.data_fingerprint,
                "checksum": migrated.checksum,
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


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
    report = diagnose(
        records,
        manifest.assignments,
        manifest.ratios,
        include_label_balance="stratified" in manifest.algorithm,
    )
    rendered = report_json(report) + "\n" if args.format == "json" else report_markdown(report)
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


def _run_temporal(args: argparse.Namespace) -> int:
    _require_distinct_paths(input=args.input, output=args.output)
    records = _load(args)
    intervals: dict[str, TimeInterval] = {}
    for record in records:
        values = [record.payload.get(field) for field in (args.start_field, args.end_field)]
        if not all(isinstance(value, str) for value in values):
            raise ValueError(
                f"record {record.id!r} requires ISO timestamp strings for start and end"
            )
        start, end = (str(value) for value in values)
        intervals[record.id] = TimeInterval(
            datetime.fromisoformat(start.replace("Z", "+00:00")),
            datetime.fromisoformat(end.replace("Z", "+00:00")),
        )
    folds = purged_kfold(
        records,
        intervals,
        args.folds,
        gap=timedelta(seconds=args.gap_seconds),
        embargo=timedelta(seconds=args.embargo_seconds),
        protect_groups=not args.allow_group_overlap,
    )
    rendered = (
        json.dumps(
            {
                "schema_version": 1,
                "algorithm": "purged-time-kfold-v1",
                "gap_seconds": args.gap_seconds,
                "embargo_seconds": args.embargo_seconds,
                "protect_groups": not args.allow_group_overlap,
                "folds": [asdict(fold) for fold in folds],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.write_text(rendered, encoding="utf-8")
    return 0


def _run_repeat(args: argparse.Namespace) -> int:
    _require_distinct_paths(input=args.input, output=args.output)
    records = _load(args)
    repetitions = repeated_kfold(
        records,
        args.folds,
        args.repeats,
        seed=args.seed,
        stratified=args.stratified,
    )
    report = stability_report(repetitions, folds=args.folds)
    rendered = (
        json.dumps(
            {
                "schema_version": 1,
                "folds": report.folds,
                "repeats": report.repeats,
                "pairwise_agreement": report.pairwise_agreement,
                "mean_entropy": report.mean_entropy,
                "per_record_entropy": dict(report.per_record_entropy),
                "fold_counts": {key: dict(value) for key, value in report.fold_counts.items()},
                "assignments": [
                    [asdict(item) for item in repetition] for repetition in repetitions
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.write_text(rendered, encoding="utf-8")
    return 0


def _run_repeat_holdout(args: argparse.Namespace) -> int:
    _require_distinct_paths(input=args.input, output=args.output)
    records = _load(args)
    repetitions = repeated_group_holdout(
        records,
        args.ratios,
        args.repeats,
        seed=args.seed,
        stratified=args.stratified,
        minimum_counts=_minimum_counts(args.minimum_count),
    )
    report = holdout_stability_report(repetitions)
    rendered = (
        json.dumps(
            {
                "schema_version": 1,
                "ratios": args.ratios,
                "repeats": report.repeats,
                "splits": list(report.splits),
                "allocation_rates": {
                    record_id: dict(rates) for record_id, rates in report.allocation_rates.items()
                },
                "assignments": [
                    [asdict(item) for item in repetition] for repetition in repetitions
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.write_text(rendered, encoding="utf-8")
    return 0


def _run_materialize(args: argparse.Namespace) -> int:
    _require_distinct_paths(input=args.input, manifest=args.manifest)
    records = _load(args)
    manifest = load_manifest(args.manifest)
    paths = write_materialized(records, manifest, args.output_dir)
    print(
        json.dumps(
            {"output_dir": str(args.output_dir), "files": [str(path) for path in paths]},
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _run_hash_stream(args: argparse.Namespace) -> int:
    if args.input.suffix.lower() != ".jsonl":
        raise ValueError("hash-stream requires a JSONL input so records can be bounded-memory")
    _require_distinct_paths(input=args.input, assignments=args.assignments, report=args.report)
    records = iter_records(
        args.input,
        id_field=args.id_field,
        group_field=args.group_field,
        label_field=args.label_field,
        weight_field=args.weight_field,
        group_weight_field=args.group_weight_field,
    )
    result = write_hash_split_stream(records, args.ratios, args.assignments, seed=args.seed)
    rendered = json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n"
    if args.report is None:
        print(rendered, end="")
    else:
        args.report.write_text(rendered, encoding="utf-8")
    return 0


def _run_hash_stream_verify(args: argparse.Namespace) -> int:
    result = verify_hash_split_stream(args.assignments, args.report)
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0


def _run_compare(args: argparse.Namespace) -> int:
    if args.output is not None:
        _require_distinct_paths(output=args.output, before=args.before, after=args.after)
    report = compare_manifests(load_manifest(args.before), load_manifest(args.after))
    rendered = json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.write_text(rendered, encoding="utf-8")
    return 1 if report.changed else 0


def _run_leakage(args: argparse.Namespace) -> int:
    _require_distinct_paths(input=args.input, assignments=args.assignments, output=args.output)
    report = audit_leakage(
        _load(args),
        load_assignments(args.assignments),
        fields=tuple(part.strip() for part in args.fields.split(",") if part.strip()),
        min_length=args.min_length,
        max_findings=args.max_findings,
    )
    rendered = json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.write_text(rendered, encoding="utf-8")
    return 1 if not report.valid else 0


def _run_near_duplicate(args: argparse.Namespace) -> int:
    _require_distinct_paths(input=args.input, assignments=args.assignments, output=args.output)
    report = audit_near_duplicates(
        _load(args),
        load_assignments(args.assignments),
        field=args.field,
        threshold=args.threshold,
        min_tokens=args.min_tokens,
        max_pairs=args.max_pairs,
    )
    rendered = json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.write_text(rendered, encoding="utf-8")
    return 1 if not report.valid else 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and convert validation errors into concise exit status 2."""
    args = build_parser().parse_args(argv)
    runners = {
        "split": _run_split,
        "kfold": _run_kfold,
        "verify": _run_verify,
        "migrate-manifest": _run_migrate,
        "inspect": _run_inspect,
        "temporal-kfold": _run_temporal,
        "repeat": _run_repeat,
        "repeat-holdout": _run_repeat_holdout,
        "materialize": _run_materialize,
        "hash-stream": _run_hash_stream,
        "hash-stream-verify": _run_hash_stream_verify,
        "compare": _run_compare,
        "leakage-audit": _run_leakage,
        "near-duplicate-audit": _run_near_duplicate,
    }
    try:
        return runners[args.command](args)
    except (ValueError, OSError, TypeError, KeyError, OverflowError) as error:
        print(f"splitproof: error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
