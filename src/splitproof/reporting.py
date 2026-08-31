"""Machine-readable and human-readable diagnostic reports."""

from __future__ import annotations

from dataclasses import asdict

from .jsonutil import strict_dumps
from .models import SplitDiagnostics


def report_json(report: SplitDiagnostics) -> str:
    """Render diagnostics as formatted JSON."""
    return strict_dumps(asdict(report), ensure_ascii=False, indent=2, sort_keys=True)


def report_markdown(report: SplitDiagnostics) -> str:
    """Render a compact Markdown report suitable for CI summaries."""
    lines = [
        "# SplitProof diagnostics",
        "",
        f"Status: **{'PASS' if report.valid else 'FAIL'}**",
        "",
        "| Split | Records | Observed ratio |",
        "|---|---:|---:|",
    ]
    names = sorted(set(report.counts) | set(report.ratios))
    lines.extend(
        f"| `{name}` | {report.counts.get(name, 0)} | {report.ratios.get(name, 0.0):.4f} |"
        for name in names
    )
    lines.extend(
        [
            "",
            f"Maximum requested-ratio deviation: `{report.max_ratio_deviation:.6f}`",
            "",
            f"Leaking groups: `{len(report.group_leakage)}`",
            f"Missing record IDs: `{len(report.missing_ids)}`",
            f"Unexpected record IDs: `{len(report.unexpected_ids)}`",
        ]
    )
    return "\n".join(lines) + "\n"
