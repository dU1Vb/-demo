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


def _fmt_ratio(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2%}"


def _fmt_float(value: Any, digits: int = 4) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}f}"
    return "n/a"


def _bool_text(value: Any) -> str:
    if value is True:
        return "yes"
    if value is False:
        return "no"
    return "n/a"


def _top_failures(payload: dict[str, Any], *, limit: int = 8) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    suites = payload.get("suites") or {}
    for suite_name, suite in suites.items():
        for case in suite.get("cases") or []:
            if case.get("ok") or case.get("skipped"):
                continue
            items.append(
                {
                    "suite": suite_name,
                    "case_id": case.get("case_id"),
                    "category": case.get("category"),
                    "error": case.get("error"),
                    "duration_s": case.get("duration_s"),
                    "counts_toward_adaptation": case.get("counts_toward_adaptation"),
                }
            )
    items.sort(key=lambda item: (item["suite"], item["case_id"]))
    return items[:limit]


def _markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    if not rows:
        return ["_none_"]
    header = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    lines = [header, sep]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return lines


def _first_pass_rate(payload: dict[str, Any]) -> float | None:
    total = 0
    first_pass = 0
    for suite in (payload.get("suites") or {}).values():
        if suite.get("skipped"):
            continue
        for case in suite.get("cases") or []:
            if not case.get("counts_toward_adaptation") or case.get("skipped"):
                continue
            total += 1
            if case.get("ok") and not ((case.get("details") or {}).get("agent_successful_repair")):
                first_pass += 1
    return (first_pass / total) if total else None


def _numeric_summary(payload: dict[str, Any]) -> dict[str, Any]:
    nf = payload.get("numeric_fidelity") or {}
    delta = payload.get("numeric_delta") or {}
    y = (delta.get("y_metrics") or {}) if isinstance(delta, dict) else {}
    g = (delta.get("g_metrics") or {}) if isinstance(delta, dict) else {}
    max_errs = [v for v in [nf.get("max_abs_err_y"), nf.get("max_abs_err_g"), y.get("max_abs_err"), g.get("max_abs_err")] if isinstance(v, (int, float))]
    rmses = [v for v in [nf.get("rmse_y"), nf.get("rmse_g"), y.get("rmse"), g.get("rmse")] if isinstance(v, (int, float))]
    return {
        "within_tolerance": nf.get("within_tolerance_all"),
        "max_error": max(max_errs) if max_errs else None,
        "mean_rmse": (sum(rmses) / len(rmses)) if rmses else None,
    }


def _performance_summary(payload: dict[str, Any]) -> dict[str, Any]:
    bench = payload.get("benchmark_summary") or {}
    agg = bench.get("aggregate") or {}
    durations: list[float] = []
    for suite in (payload.get("suites") or {}).values():
        if suite.get("skipped"):
            continue
        for case in suite.get("cases") or []:
            value = case.get("duration_s")
            if isinstance(value, (int, float)):
                durations.append(float(value))
    return {
        "mean_case_seconds": (sum(durations) / len(durations)) if durations else None,
        "throughput_iter_s": agg.get("throughput_iter_s"),
        "mean_latency_s": agg.get("mean_latency_s"),
    }


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
        "benchmark_summary": summary.benchmark_summary,
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
    bench = payload.get("benchmark_summary") or {}
    agent = payload.get("agent_report") or {}
    adapter = payload.get("adapter") or {}
    repair_metrics = agent.get("repair_metrics") or {}
    effort_breakdown = agent.get("effort_breakdown") or {}
    numeric_summary = _numeric_summary(payload)
    performance_summary = _performance_summary(payload)
    first_pass_rate = _first_pass_rate(payload)
    migrate = repair_metrics.get("migrate_at_k") or {}
    env_issues = [item for item in (agent.get("environment_issues") or []) if not item.get("resolved")]
    top_failures = _top_failures(payload)
    suite_rows = []
    for suite_name, stats in sorted((es.get("by_suite") or {}).items()):
        suite_rows.append(
            [
                suite_name,
                _fmt_ratio(stats.get("adaptation_rate")),
                str(stats.get("passed", 0)),
                str(stats.get("total", 0)),
            ]
        )
    category_rows = []
    for cat, stats in sorted((es.get("by_category") or {}).items()):
        category_rows.append(
            [
                cat,
                _fmt_ratio(stats.get("adaptation_rate")),
                str(stats.get("passed", 0)),
                str(stats.get("total", 0)),
            ]
        )
    benchmark_rows = []
    for case_id, stats in sorted((bench.get("per_case") or {}).items()):
        benchmark_rows.append(
            [
                case_id,
                str(stats.get("iters", "n/a")),
                _fmt_float(stats.get("total_s")),
                _fmt_float(stats.get("latency_s"), 6),
                _fmt_float(stats.get("throughput_iter_s")),
            ]
        )

    lines: list[str] = [
        f"# Migration library evaluation: {library_name}",
        "",
        f"- Generated (UTC): `{payload.get('generated_at_utc', '')}`",
        f"- Library: `{adapter.get('library_name', library_name)}`",
        f"- Target device: `{((adapter.get('device') or {}).get('target_device')) or 'n/a'}`",
        "",
        "## Key Metrics",
        "",
        "| Metric | Value | Meaning |",
        "| --- | ---: | --- |",
        f"| Compatibility rate | `{_fmt_ratio(es.get('adaptation_rate_overall'))}` | Bridge-relevant pass rate after excluding environment/config noise. |",
        f"| First-pass rate | `{_fmt_ratio(first_pass_rate)}` | Passed without counted repair. |",
        f"| ME | `{_fmt_float(repair_metrics.get('effort_total'))}` | Total counted migration effort. |",
        f"| AR | `{_fmt_ratio(repair_metrics.get('AR'))}` | Work avoided versus calibrated no-bridge baseline. |",
        f"| Effort_adapt / Effort_repair | `{_fmt_float(effort_breakdown.get('effort_adapt'))}` / `{_fmt_float(effort_breakdown.get('effort_repair'))}` | Adapter authoring vs repair effort. |",
        f"| Numeric max error | `{_fmt_float(numeric_summary.get('max_error'))}` | Worst numeric fidelity error where measured. |",
        f"| Numeric mean RMSE | `{_fmt_float(numeric_summary.get('mean_rmse'))}` | Average RMSE across output/gradient vectors. |",
        f"| Mean case seconds | `{_fmt_float(performance_summary.get('mean_case_seconds'))}` | Average case runtime. |",
        f"| Benchmark throughput | `{_fmt_float(performance_summary.get('throughput_iter_s'))}` | Aggregate benchmark iterations per second. |",
        f"| Benchmark latency | `{_fmt_float(performance_summary.get('mean_latency_s'), 6)}` | Mean benchmark seconds per iteration. |",
        f"| migrate@1 | `{_fmt_ratio(migrate.get('migrate@1'))}` | One repair reroll succeeds, if repair was attempted. |",
        f"| Environment issues | `{len(env_issues)}` | Unresolved environment/config issues surfaced to user. |",
        "",
        "## Executive Summary",
        "",
        f"- Overall adaptation rate: `{_fmt_ratio(es.get('adaptation_rate_overall'))}`",
        f"- First-pass rate: `{_fmt_ratio(first_pass_rate)}`",
        f"- ME / AR: `{_fmt_float(repair_metrics.get('effort_total'))}` / `{_fmt_ratio(repair_metrics.get('AR'))}`",
        f"- Counted cases: `{(es.get('counts') or {}).get('total', 0)}`; passed: `{(es.get('counts') or {}).get('passed', 0)}`; failed: `{(es.get('counts') or {}).get('failed_n', 0)}`; skipped: `{(es.get('counts') or {}).get('skipped_n', 0)}`",
        f"- Smoke gate failed: `{_bool_text(payload.get('smoke_failed'))}`",
        f"- Numeric fidelity within tolerance: `{_bool_text(nf.get('within_tolerance_all'))}`",
        f"- Unresolved environment issues: `{len(env_issues)}`",
        "",
        "## Key Findings",
        "",
    ]
    if top_failures:
        for item in top_failures:
            lines.append(
                f"- [{item['suite']}] `{item['case_id']}` [{item['category']}] "
                f"error=`{str(item['error'])[:160]}` counted=`{item['counts_toward_adaptation']}`"
            )
    else:
        lines.append("- No unresolved failed cases in the recorded suites.")
    lines.extend(
        [
            "",
            "## Adaptation Summary",
            "",
            "### By Suite",
            "",
        ]
    )
    lines.extend(_markdown_table(["Suite", "Rate", "Passed", "Total"], suite_rows))
    lines.extend(
        [
            "",
            "### By Category",
            "",
        ]
    )
    lines.extend(_markdown_table(["Category", "Rate", "Passed", "Total"], category_rows))
    lines.extend(
        [
            "",
            "## Numeric Fidelity",
            "",
            f"- Within tolerance (outputs): `{_bool_text(nf.get('within_tolerance_y'))}`",
            f"- Within tolerance (gradients): `{_bool_text(nf.get('within_tolerance_g'))}`",
            f"- Max abs err (y / g): `{_fmt_float(nf.get('max_abs_err_y'))}` / `{_fmt_float(nf.get('max_abs_err_g'))}`",
            f"- RMSE (y / g): `{_fmt_float(nf.get('rmse_y'))}` / `{_fmt_float(nf.get('rmse_g'))}`",
            "",
            "## Benchmark Summary",
            "",
            f"- Successful benchmark cases: `{((bench.get('aggregate') or {}).get('successful_cases')) or 0}`",
            f"- Aggregate throughput: `{_fmt_float((bench.get('aggregate') or {}).get('throughput_iter_s'))}` iter/s",
            f"- Mean latency: `{_fmt_float((bench.get('aggregate') or {}).get('mean_latency_s'), 6)}` s/iter",
            "",
        ]
    )
    lines.extend(_markdown_table(["Case", "Iters", "Total(s)", "Latency(s)", "Throughput(iter/s)"], benchmark_rows))
    lines.extend(
        [
            "",
            "## Agent Summary",
            "",
            f"- Mode: `{agent.get('mode', 'off')}`",
            f"- LLM calls: `{((agent.get('migration_effort') or {}).get('llm_calls')) or 0}`",
            f"- Revalidation runs counted toward effort: `{((agent.get('migration_effort') or {}).get('revalidation_runs')) or 0}`",
            f"- Repair attempts: `{((agent.get('migration_effort') or {}).get('repair_attempts')) or 0}`",
            f"- Repair success rate: `{_fmt_ratio(((agent.get('repair_metrics') or {}).get('repair_success_rate')) )}`",
            f"- AR: `{_fmt_ratio(((agent.get('repair_metrics') or {}).get('AR')) )}`",
            f"- Baseline effort: `{_fmt_float(((agent.get('repair_metrics') or {}).get('baseline_effort')) )}`",
            f"- Effort total: `{_fmt_float(((agent.get('repair_metrics') or {}).get('effort_total')) )}`",
            f"- AR constant model: `{((agent.get('repair_metrics') or {}).get('ar_constant_model')) or 'n/a'}`",
            f"- AR agent system: `{((agent.get('repair_metrics') or {}).get('ar_agent_system_version')) or 'n/a'}`",
            f"- AR constant compatible: `{_bool_text(((agent.get('repair_metrics') or {}).get('ar_constant_compatible')) )}`",
            "",
        ]
    )
    if env_issues:
        lines.extend(["### Environment Issues", ""])
        for issue in env_issues:
            lines.append(
                f"- [{issue.get('suite')}] `{issue.get('case_id')}` `{issue.get('failure_class')}`: `{str(issue.get('final_error') or '')[:180]}`"
            )
            for action in issue.get("suggested_user_actions") or []:
                lines.append(f"  action: {action}")
        lines.append("")
    lines.extend(
        [
            "## Raw Metrics",
            "",
            "### Numeric delta details",
            "",
            "```json",
            json.dumps(payload.get("numeric_delta", {}), indent=2, ensure_ascii=False),
            "```",
            "",
            "### Agent report",
            "",
            "```json",
            json.dumps(agent, indent=2, ensure_ascii=False)[:20000],
            "```",
            "",
            "### Hardware",
            "",
            "```json",
            json.dumps(payload.get("hardware", {}), indent=2, ensure_ascii=False),
            "```",
            "",
            "### Adapter summary",
            "",
            "```json",
            json.dumps(adapter, indent=2, ensure_ascii=False)[:12000],
            "```",
            "",
            "## Suite Details",
            "",
        ]
    )
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
            status = "skip" if c.get("skipped") else ("pass" if c.get("ok") else "fail")
            cat = c.get("category", "")
            lines.append(
                f"- [{status}] `{c.get('case_id')}` [{cat}] {c.get('name')} "
                f"({float(c.get('duration_s', 0) or 0):.3f}s, counted=`{c.get('counts_toward_adaptation')}`)"
            )
            if not c.get("ok") and not c.get("skipped"):
                if c.get("error"):
                    lines.append(f"  error: `{str(c.get('error'))[:500]}`")
                details = c.get("details") or {}
                agent_diag = details.get("agent_diagnostic")
                if agent_diag:
                    lines.append(
                        f"  diagnostic: class=`{agent_diag.get('failure_class')}` env=`{agent_diag.get('is_environment_or_config')}`"
                    )
        lines.append("")
    mpath.write_text("\n".join(lines), encoding="utf-8")
    return jpath, mpath
