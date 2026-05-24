"""JSON and Markdown report writers."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agentorskill_cli.adapter_schema import AdapterSpec, adapter_spec_to_dict
from agentorskill_cli.hardware_probe import HardwareReport
from agentorskill_cli.test_suites.harness import EvalCaseResult
from agentorskill_cli.test_suites.types_summary import RunSummary, SuiteRun


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def case_to_dict(c: EvalCaseResult) -> dict[str, Any]:
    return {
        "case_id": c.case_id,
        "name": c.name,
        "ok": c.ok,
        "skipped": c.skipped,
        "category": c.category,
        "weight": c.weight,
        "counts_toward_adaptation": c.counts_toward_adaptation,
        "duration_s": c.duration_s,
        "error": c.error,
        "details": c.details,
    }


def suite_to_dict(s: SuiteRun) -> dict[str, Any]:
    return {
        "name": s.name,
        "passed": s.passed,
        "failed": s.failed,
        "skipped_n": s.skipped_n,
        "skipped": s.skipped,
        "cases": [case_to_dict(c) for c in s.cases],
    }


def build_report_payload(
    *,
    adapter: AdapterSpec,
    hardware: HardwareReport,
    summary: RunSummary,
    doc_sources: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "adapter": adapter_spec_to_dict(adapter),
        "hardware": hardware.to_dict(),
        "doc_sources": doc_sources or [],
        "evaluation_summary": summary.evaluation_summary,
        "numeric_delta": summary.numeric_delta,
        "numeric_fidelity": summary.numeric_fidelity,
        "agent_report": summary.agent_report,
        "smoke_failed": summary.smoke_failed,
        "suites": {k: suite_to_dict(v) for k, v in summary.suites.items()},
    }


def write_reports(
    payload: dict[str, Any],
    out_dir: Path,
    library_name: str,
    *,
    stamp: str | None = None,
) -> tuple[Path, Path]:
    """Write JSON and Markdown with the same report id."""
    out_dir.mkdir(parents=True, exist_ok=True)
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in library_name)[:64]
    sub = out_dir / safe
    sub.mkdir(parents=True, exist_ok=True)
    sid = stamp or _ts()
    jpath = sub / f"report-{sid}.json"
    mpath = sub / f"report-{sid}.md"
    jpath.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    es = payload.get("evaluation_summary") or {}
    nf = payload.get("numeric_fidelity") or {}
    agent = payload.get("agent_report") or {}

    lines: list[str] = [
        f"# Migration library evaluation: {library_name}",
        "",
        f"- Generated (UTC): `{payload.get('generated_at_utc', '')}`",
        "",
        "## Evaluation summary (adaptation rate)",
        "",
        f"- Overall adaptation rate: `{es.get('adaptation_rate_overall')}`",
        f"- Counts: `{json.dumps(es.get('counts', {}), ensure_ascii=False)}`",
        "",
        "### By category",
        "",
        "```",
        json.dumps(es.get("by_category", {}), indent=2, ensure_ascii=False),
        "```",
        "",
        "### By suite",
        "",
        "```",
        json.dumps(es.get("by_suite", {}), indent=2, ensure_ascii=False),
        "```",
        "",
        f"- Failed case IDs ({len(es.get('failed_case_ids') or [])}): `{', '.join((es.get('failed_case_ids') or [])[:40])}`",
        "",
        "## Numeric fidelity (vs CPU baseline)",
        "",
        "```",
        json.dumps(nf, indent=2, ensure_ascii=False),
        "```",
        "",
        "## Numeric delta details",
        "",
        "```",
        json.dumps(payload.get("numeric_delta", {}), indent=2, ensure_ascii=False),
        "```",
        "",
        "## Agent report",
        "",
        "```json",
        json.dumps(agent, indent=2, ensure_ascii=False)[:20000],
        "```",
        "",
        "## Hardware",
        "",
        "```",
        json.dumps(payload.get("hardware", {}), indent=2, ensure_ascii=False),
        "```",
        "",
        "## Adapter summary",
        "",
        "```json",
        json.dumps(payload.get("adapter", {}), indent=2, ensure_ascii=False)[:12000],
        "```",
        "",
        "## Suites",
        "",
    ]
    suites = payload.get("suites") or {}
    for name, s in suites.items():
        lines.append(f"### {name}")
        lines.append("")
        if s.get("skipped"):
            lines.append("_Skipped._")
            lines.append("")
            continue
        lines.append(
            f"- Passed: {s.get('passed', 0)}, Failed: {s.get('failed', 0)}, Skipped cases: {s.get('skipped_n', 0)}"
        )
        lines.append("")
        for c in s.get("cases") or []:
            if c.get("skipped"):
                icon = "skip"
            elif c.get("ok"):
                icon = "ok"
            else:
                icon = "x"
            cat = c.get("category", "")
            lines.append(
                f"- [{icon}] `{c.get('case_id')}` [{cat}] {c.get('name')} ({c.get('duration_s', 0):.3f}s)"
            )
            if not c.get("ok") and not c.get("skipped") and c.get("error"):
                lines.append(f"  - Error: `{str(c.get('error'))[:500]}`")
        lines.append("")
    mpath.write_text("\n".join(lines), encoding="utf-8")
    return jpath, mpath
