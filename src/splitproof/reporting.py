"""Machine-readable and human-readable diagnostic reports."""

from __future__ import annotations

from dataclasses import asdict

from .jsonutil import strict_dumps
from .models import SplitDiagnostics


def report_json(report: SplitDiagnostics) -> str:
    """Render diagnostics as formatted JSON."""
    return strict_dumps(asdict(report), ensure_ascii=False, indent=2, sort_keys=True)


def _cell(value: object) -> str:
    return (
        str(value).replace("\\", "\\\\").replace("|", "\\|").replace("\r", " ").replace("\n", " ")
    )


def report_markdown(report: SplitDiagnostics) -> str:
    """Render a compact Markdown report suitable for CI summaries."""
    lines = [
        "# SplitProof diagnostics",
        "",
        f"Status: **{'PASS' if report.valid else 'FAIL'}**",
        "",
        "| Split | Records | Record ratio | Record weight | Weight ratio | Group-weight ratio |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    names = sorted(
        set(report.counts)
        | set(report.ratios)
        | set(report.record_weights)
        | set(report.group_weights)
    )
    lines.extend(
        f"| `{_cell(name)}` | {report.counts.get(name, 0)} "
        f"| {report.ratios.get(name, 0.0):.4f} "
        f"| {report.record_weights.get(name, 0.0):.4f} "
        f"| {report.record_weight_ratios.get(name, 0.0):.4f} "
        f"| {report.group_weight_ratios.get(name, 0.0):.4f} |"
        for name in names
    )
    lines.extend(
        [
            "",
            "Balance deviations:",
            "",
            f"- record count: `{report.max_ratio_deviation:.6f}`",
            f"- record weight: `{report.max_record_weight_deviation:.6f}`",
            f"- effective group weight: `{report.max_group_weight_deviation:.6f}`",
            f"- per-label weight: `{report.max_label_deviation:.6f}`",
            f"- optimizer objective: `{report.objective_score:.8f}`",
            "",
            "Objective contributions:",
            "",
            *[
                f"- {_cell(name)}: `{value:.8f}`"
                for name, value in report.objective_components.items()
                if name != "total"
            ],
            "",
            "Per-label maximum deviations:",
            "",
            *(
                [
                    f"- {_cell(label)}: `{value:.6f}`"
                    for label, value in report.label_deviations.items()
                ]
                or ["- none"]
            ),
            "",
            f"Leaking groups: `{len(report.group_leakage)}`",
            f"Missing record IDs: `{len(report.missing_ids)}`",
            f"Unexpected record IDs: `{len(report.unexpected_ids)}`",
        ]
    )
    return "\n".join(lines) + "\n"
