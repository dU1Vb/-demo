"""Typer CLI: probe, extract, eval, run."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.table import Table

from agentorskill_cli.adapter_schema import AdapterSpec, adapter_spec_to_dict, validate_adapter_dict
from agentorskill_cli.agent_schema import AgentRunReport, MigrationEffortStats, RepairMetrics
from agentorskill_cli.ar_identity import AR_AGENT_SYSTEM_VERSION, AR_EFFORT_FORMULA_VERSION
from agentorskill_cli.ar_calibration import load_task_cases, measure_baseline_effort, save_baseline_effort_constant
from agentorskill_cli.agent_loop import run_agent_loop
from agentorskill_cli.doc_reader import collect_docs
from agentorskill_cli.effort import calc_ar, calc_effort_from_stats
from agentorskill_cli.hardware_probe import format_hardware_summary, probe_hardware, user_confirms
from agentorskill_cli.llm_adapter import extract_adapter_with_llm
from agentorskill_cli.llm_provider import OpenAIChatProvider
from agentorskill_cli.metrics import compute_evaluation_summary
from agentorskill_cli.report import build_report_payload, write_reports
from agentorskill_cli.test_suites.runner import (
    execute_plan,
    execute_plan_inprocess,
    parse_adaptation_suites_arg,
)

app = typer.Typer(
    name="eval-migration",
    help="Evaluate PyTorch migration libraries with reproducible harness and reports.",
    no_args_is_help=True,
)
console = Console()


def _stats_to_effort_ledger(
    *,
    bridge_id: str,
    phase: str,
    stats: MigrationEffortStats,
    measurement: str,
) -> dict:
    return {
        "schema_version": "tbbcc.effort_ledger.v0.2",
        "entries": [
            {
                "bridge_id": bridge_id,
                "phase": phase,
                "rounds": stats.llm_calls + stats.revalidation_runs + stats.repair_attempts,
                "prompt_chars": stats.prompt_chars,
                "completion_chars": stats.completion_chars,
                "edit_units": stats.patch_lines_added + stats.patch_lines_deleted,
                "measurement": measurement,
                "classification_confirmation_counted": False,
                "environment_remediation_counted": False,
            }
        ],
        "migration_samples": [],
    }


def _add_effort_stats(a: MigrationEffortStats, b: MigrationEffortStats) -> MigrationEffortStats:
    return MigrationEffortStats(
        agent_rounds=a.agent_rounds + b.agent_rounds,
        llm_calls=a.llm_calls + b.llm_calls,
        prompt_chars=a.prompt_chars + b.prompt_chars,
        completion_chars=a.completion_chars + b.completion_chars,
        revalidation_runs=a.revalidation_runs + b.revalidation_runs,
        repair_attempts=a.repair_attempts + b.repair_attempts,
        patches_generated=a.patches_generated + b.patches_generated,
        patch_lines_added=a.patch_lines_added + b.patch_lines_added,
        patch_lines_deleted=a.patch_lines_deleted + b.patch_lines_deleted,
        elapsed_s=a.elapsed_s + b.elapsed_s,
    )


def _apply_ar_constant(
    *,
    summary,
    ar_constant_file: Path | None,
    model: str | None,
) -> None:
    if ar_constant_file is None or not summary.agent_report:
        return
    ar_payload = json.loads(ar_constant_file.read_text(encoding="utf-8"))
    baseline_effort = ar_payload.get("baseline_effort")
    current_model = (summary.agent_report or {}).get("model") or _resolve_effective_model(model)
    compatibility_issues = _validate_ar_constant_compatibility(
        ar_payload=ar_payload,
        current_model=current_model,
    )
    me = ((summary.agent_report or {}).get("migration_effort") or {})
    me_total = calc_effort_from_stats(MigrationEffortStats.model_validate(me)) if isinstance(me, dict) else None
    ar_value = None
    if not compatibility_issues and isinstance(baseline_effort, (int, float)):
        ar_value = calc_ar(me_total, float(baseline_effort))
    repair_metrics = summary.agent_report.get("repair_metrics") if summary.agent_report else {}
    if not isinstance(repair_metrics, dict):
        repair_metrics = {}
    repair_metrics["baseline_effort"] = baseline_effort
    repair_metrics["effort_total"] = me_total
    repair_metrics["AR"] = ar_value
    repair_metrics["ar_constant_model"] = ar_payload.get("model")
    repair_metrics["ar_agent_system_version"] = ar_payload.get("agent_system_version")
    repair_metrics["ar_constant_compatible"] = not compatibility_issues
    repair_metrics["ar_constant_issues"] = compatibility_issues
    repair_rate = repair_metrics.get("repair_success_rate")
    if repair_rate is None:
        attempted = repair_metrics.get("attempted_cases") or 0
        successful = repair_metrics.get("successful_repairs") or 0
        repair_rate = (successful / attempted) if attempted else None
    repair_metrics["repair_success_rate"] = repair_rate
    summary.agent_report["repair_metrics"] = repair_metrics
    if compatibility_issues:
        console.print("[yellow]AR constant compatibility check failed; AR will be omitted.[/yellow]")
        for issue in compatibility_issues:
            console.print(f"- {issue}")


def _has_unresolved_failures(summary) -> bool:
    for suite in summary.suites.values():
        if suite.skipped:
            continue
        for case in suite.cases:
            if case.ok or case.skipped:
                continue
            if not case.counts_toward_adaptation:
                continue
            if case.details.get("agent_successful_repair"):
                continue
            return True
    return False


def _emit_environment_issues(agent_report: dict) -> None:
    issues = agent_report.get("environment_issues") or []
    if not issues:
        return
    console.print("[yellow]Environment issues requiring user action:[/yellow]")
    for issue in issues:
        if issue.get("resolved"):
            continue
        suite = issue.get("suite", "?")
        case_id = issue.get("case_id", "?")
        failure_class = issue.get("failure_class", "EnvironmentFailure")
        console.print(f"- [{suite}] {case_id}: {failure_class}")
        final_error = issue.get("final_error")
        if final_error:
            console.print(f"  error: {final_error}")
        for action in issue.get("suggested_user_actions") or []:
            console.print(f"  action: {action}")


def _resolve_effective_model(cli_model: str | None) -> str | None:
    if cli_model:
        return cli_model
    return None


def _validate_ar_constant_compatibility(
    *,
    ar_payload: dict,
    current_model: str | None,
) -> list[str]:
    issues: list[str] = []
    constant_model = ar_payload.get("model")
    constant_system = ar_payload.get("agent_system_version")
    constant_formula = ar_payload.get("effort_formula_version")
    if current_model and constant_model and current_model != constant_model:
        issues.append(f"model mismatch: current={current_model}, constant={constant_model}")
    if constant_system and constant_system != AR_AGENT_SYSTEM_VERSION:
        issues.append(
            f"agent system mismatch: current={AR_AGENT_SYSTEM_VERSION}, constant={constant_system}"
        )
    if constant_formula and constant_formula != AR_EFFORT_FORMULA_VERSION:
        issues.append(
            f"effort formula mismatch: current={AR_EFFORT_FORMULA_VERSION}, constant={constant_formula}"
        )
    return issues


@app.command("probe")
def cmd_probe(
    yes: Annotated[
        bool,
        typer.Option("--yes", "-y", help="Skip interactive confirmation"),
    ] = False,
) -> None:
    """Print CPU/GPU/NPU and framework probe; optional confirmation."""
    r = probe_hardware()
    console.print(format_hardware_summary(r))
    if not yes:
        if not user_confirms("Record this environment for the next report?"):
            raise typer.Exit(0)


@app.command("calibrate-ar")
def cmd_calibrate_ar(
    task_file: Annotated[Path, typer.Option("--task-file", exists=True, help="JSON list of translation tasks or benchmark case directory")] ,
    rerolls: Annotated[int, typer.Option("--rerolls", min=2, help="Number of full rerolls to average")] = 5,
    max_workers: Annotated[int, typer.Option("--max-workers", min=1, help="Concurrent translation requests")] = 4,
    max_cv: Annotated[float, typer.Option("--max-cv", help="Maximum allowed coefficient of variation")] = 0.1,
    out: Annotated[Path, typer.Option("--out", "-o", help="Write baseline effort constant JSON")] = Path("reports/ar_constant.json"),
    model: Annotated[Optional[str], typer.Option("--model", envvar="OPENAI_MODEL")] = None,
    base_url: Annotated[Optional[str], typer.Option("--openai-base-url", envvar="OPENAI_BASE_URL")] = None,
) -> None:
    """Measure baseline translation effort by repeated full rerolls over a task set."""
    provider = OpenAIChatProvider(model=model, base_url=base_url)
    task_cases = load_task_cases(task_file)
    if not task_cases:
        raise typer.BadParameter("--task-file must resolve to a non-empty task set")
    result = measure_baseline_effort(
        provider=provider,
        task_cases=task_cases,
        rerolls=rerolls,
        max_cv=max_cv,
        max_workers=max_workers,
    )
    save_baseline_effort_constant(out, result)
    console.print(json.dumps(
        {
            "baseline_effort": result.baseline_effort,
            "mean_effort": result.mean_effort,
            "std_effort": result.std_effort,
            "cv_effort": result.cv_effort,
            "sample_count": len(result.samples),
            "reroll_count": len(result.rerolls),
            "task_count": len(result.source_cases),
            "model": result.model,
            "out": str(out),
        },
        indent=2,
        ensure_ascii=False,
    ))


def _load_adapter(
    adapter_file: Optional[Path],
    readme_url: Optional[str],
    doc_url: list[str],
    local_path: Optional[str],
    extra_text: Optional[str],
    model: Optional[str],
    base_url: Optional[str],
) -> tuple[AdapterSpec, list[dict]]:
    if adapter_file:
        data = json.loads(adapter_file.read_text(encoding="utf-8"))
        return validate_adapter_dict(data), []

    text, sources = collect_docs(
        readme_url=readme_url,
        doc_urls=list(doc_url) if doc_url else None,
        local_path=local_path,
        extra_text=extra_text,
    )
    if not text.strip():
        console.print("[red]No documentation text collected. Use --adapter-file, --readme-url, or --local-path.[/red]")
        raise typer.Exit(2)
    spec = extract_adapter_with_llm(text, model=model, base_url=base_url)
    return spec, sources


def _run_evaluation(
    *,
    spec: AdapterSpec,
    sources: list[dict],
    suite: str,
    yes: bool,
    no_probe: bool,
    model: str | None,
    base_url: str | None,
    report_dir: Path,
    honor_adapter_device: bool,
    adaptation_suites: str | None,
    include_smoke_in_adaptation: bool,
    numeric_rtol: float,
    numeric_atol: float,
    single_process: bool,
    agent_mode: str,
    agent_max_failures: int,
    agent_repair_attempts: int,
    ar_constant_file: Path | None,
):
    if not no_probe:
        hw = probe_hardware()
        console.print(format_hardware_summary(hw))
        if not yes:
            if not user_confirms("Confirm this machine for evaluation?"):
                console.print("Aborted.")
                raise typer.Exit(1)
    else:
        hw = probe_hardware()

    console.print(f"[bold]Library:[/bold] {spec.library_name}")

    skip_bench = True
    runner = execute_plan_inprocess if single_process else execute_plan
    summ = runner(
        spec,
        suite,
        skip_benchmark_if_smoke_fails=skip_bench,
        honor_device=honor_adapter_device,
        adaptation_suites=parse_adaptation_suites_arg(adaptation_suites),
        include_smoke_in_adaptation=include_smoke_in_adaptation,
        rtol=numeric_rtol,
        atol=numeric_atol,
    )

    if agent_mode.lower().strip() != "off":
        agent_report = run_agent_loop(
            adapter=spec,
            summary=summ,
            mode=agent_mode,
            model=model,
            base_url=base_url,
            max_failures=agent_max_failures,
            repair_attempts=agent_repair_attempts,
            honor_device=honor_adapter_device,
        )
        summ.agent_report = agent_report.model_dump(mode="json")
        summ.evaluation_summary = compute_evaluation_summary(
            summ,
            adaptation_suites=parse_adaptation_suites_arg(adaptation_suites),
            include_smoke_in_adaptation=include_smoke_in_adaptation,
        )

    _apply_ar_constant(summary=summ, ar_constant_file=ar_constant_file, model=model)
    summ.evaluation_summary = compute_evaluation_summary(
        summ,
        adaptation_suites=parse_adaptation_suites_arg(adaptation_suites),
        include_smoke_in_adaptation=include_smoke_in_adaptation,
    )

    payload = build_report_payload(
        adapter=spec,
        hardware=hw,
        summary=summ,
        doc_sources=sources,
    )
    jp, mp = write_reports(payload, report_dir, spec.library_name)
    return summ, jp, mp


@app.command("extract")
def cmd_extract(
    adapter_file: Annotated[
        Optional[Path],
        typer.Option("--adapter-file", exists=True, help="Skip LLM; validate JSON only"),
    ] = None,
    readme_url: Annotated[Optional[str], typer.Option("--readme-url")] = None,
    doc_url: Annotated[list[str], typer.Option("--doc-url", help="Extra doc URL; repeatable")] = [],
    local_path: Annotated[Optional[str], typer.Option("--local-path")] = None,
    extra_text: Annotated[Optional[str], typer.Option("--extra-text")] = None,
    model: Annotated[Optional[str], typer.Option("--model", envvar="OPENAI_MODEL")] = None,
    base_url: Annotated[Optional[str], typer.Option("--openai-base-url", envvar="OPENAI_BASE_URL")] = None,
    out: Annotated[Optional[Path], typer.Option("--out", "-o", help="Write AdapterSpec JSON")] = None,
) -> None:
    """Extract AdapterSpec from docs via LLM, or validate --adapter-file."""
    if adapter_file:
        spec = validate_adapter_dict(json.loads(adapter_file.read_text(encoding="utf-8")))
    else:
        text, _ = collect_docs(
            readme_url=readme_url,
            doc_urls=list(doc_url) if doc_url else None,
            local_path=local_path,
            extra_text=extra_text,
        )
        if not text.strip():
            console.print("[red]No documentation to extract from.[/red]")
            raise typer.Exit(2)
        spec = extract_adapter_with_llm(text, model=model, base_url=base_url)
    js = json.dumps(spec.model_dump(mode="json"), indent=2, ensure_ascii=False)
    console.print(js)
    if out:
        out.write_text(js, encoding="utf-8")
        console.print(f"[green]Wrote {out}[/green]")


@app.command("run")
def cmd_run(
    adapter_file: Annotated[Optional[Path], typer.Option("--adapter-file", exists=True)] = None,
    readme_url: Annotated[Optional[str], typer.Option("--readme-url")] = None,
    doc_url: Annotated[list[str], typer.Option("--doc-url")] = [],
    local_path: Annotated[Optional[str], typer.Option("--local-path")] = None,
    extra_text: Annotated[Optional[str], typer.Option("--extra-text")] = None,
    suite: Annotated[
        str,
        typer.Option("--suite", help="smoke|core|models|training|numeric|benchmark|all"),
    ] = "smoke",
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip hardware confirmation")] = False,
    no_probe: Annotated[bool, typer.Option("--no-probe", help="Skip hardware probe (not recommended)")] = False,
    model: Annotated[Optional[str], typer.Option("--model", envvar="OPENAI_MODEL")] = None,
    base_url: Annotated[Optional[str], typer.Option("--openai-base-url", envvar="OPENAI_BASE_URL")] = None,
    report_dir: Annotated[Path, typer.Option("--report-dir", "-r")] = Path("reports"),
    honor_adapter_device: Annotated[
        bool,
        typer.Option(
            "--honor-adapter-device",
            help="Set torch default device to AdapterSpec.target_device (e.g. jax for TorchAX)",
        ),
    ] = False,
    adaptation_suites: Annotated[
        Optional[str],
        typer.Option(
            "--adaptation-suites",
            help="Comma list for rate denominator (default: core,models,training,numeric). Example: core,models",
        ),
    ] = None,
    include_smoke_in_adaptation: Annotated[
        bool,
        typer.Option(
            "--include-smoke-in-adaptation",
            help="Also count smoke suite in adaptation_rate",
        ),
    ] = False,
    numeric_rtol: Annotated[float, typer.Option("--numeric-rtol", help="allclose rtol vs baseline")] = 1e-4,
    numeric_atol: Annotated[float, typer.Option("--numeric-atol", help="allclose atol vs baseline")] = 1e-5,
    single_process: Annotated[
        bool,
        typer.Option(
            "--single-process",
            help="Run all tests in a single process (fast; preamble executed once). "
            "Numeric and benchmark suites fall back to subprocess mode.",
        ),
    ] = False,
    agent_mode: Annotated[
        str,
        typer.Option(
            "--agent-mode",
            help="off|diagnose|revalidate|repair. Agent stages run after baseline failures.",
        ),
    ] = "off",
    agent_max_failures: Annotated[
        int,
        typer.Option("--agent-max-failures", help="Maximum failed cases for agent processing."),
    ] = 5,
    agent_repair_attempts: Annotated[
        int,
        typer.Option("--agent-repair-attempts", help="Adapter repair attempts per failed case in repair mode."),
    ] = 1,
    ar_constant_file: Annotated[
        Optional[Path],
        typer.Option("--ar-constant-file", exists=True, help="JSON file produced by calibrate-ar"),
    ] = None,
) -> None:
    """Probe hardware, load/extract adapter, run suite(s), write reports."""
    spec, sources = _load_adapter(adapter_file, readme_url, doc_url, local_path, extra_text, model, base_url)
    summ, jp, mp = _run_evaluation(
        spec=spec,
        sources=sources,
        suite=suite,
        yes=yes,
        no_probe=no_probe,
        model=model,
        base_url=base_url,
        report_dir=report_dir,
        honor_adapter_device=honor_adapter_device,
        adaptation_suites=adaptation_suites,
        include_smoke_in_adaptation=include_smoke_in_adaptation,
        numeric_rtol=numeric_rtol,
        numeric_atol=numeric_atol,
        single_process=single_process,
        agent_mode=agent_mode,
        agent_max_failures=agent_max_failures,
        agent_repair_attempts=agent_repair_attempts,
        ar_constant_file=ar_constant_file,
    )
    console.print(f"[green]Wrote {jp}[/green]")
    console.print(f"[green]Wrote {mp}[/green]")

    es = summ.evaluation_summary or {}
    console.print(
        f"[bold]Adaptation rate (overall):[/bold] {es.get('adaptation_rate_overall')}"
    )
    _emit_environment_issues(summ.agent_report)

    table = Table(title="Results")
    table.add_column("Suite")
    table.add_column("Pass", justify="right")
    table.add_column("Fail", justify="right")
    table.add_column("Skip", justify="right")
    for name, sr in summ.suites.items():
        if sr.skipped:
            table.add_row(name, "-", "skipped", "-")
        else:
            table.add_row(name, str(sr.passed), str(sr.failed), str(sr.skipped_n))
    console.print(table)
    if _has_unresolved_failures(summ):
        raise typer.Exit(1)


@app.command("eval")
def cmd_eval(
    bridge_id: Annotated[
        Optional[str],
        typer.Option("--bridge-id", help="Short bridge id for generated reports."),
    ] = None,
    docs: Annotated[
        Optional[Path],
        typer.Option("--docs", exists=True, help="README, docs file, or local bridge repo used to generate AdapterSpec."),
    ] = None,
    adapter_file: Annotated[
        Optional[Path],
        typer.Option("--adapter-file", exists=True, help="Use an existing AdapterSpec but still emit standardized eval artifacts."),
    ] = None,
    suite: Annotated[str, typer.Option("--suite", help="smoke|core|models|training|numeric|benchmark|all")] = "smoke",
    out: Annotated[Path, typer.Option("--out", "-o", help="Output directory containing adapter, ledger and reports.")] = Path("reports/eval"),
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip hardware confirmation")] = True,
    no_probe: Annotated[bool, typer.Option("--no-probe", help="Skip hardware probe output")] = False,
    model: Annotated[Optional[str], typer.Option("--model", envvar="OPENAI_MODEL")] = None,
    base_url: Annotated[Optional[str], typer.Option("--openai-base-url", envvar="OPENAI_BASE_URL")] = None,
    honor_adapter_device: Annotated[bool, typer.Option("--honor-adapter-device")] = False,
    agent_mode: Annotated[str, typer.Option("--agent-mode", help="off|diagnose|revalidate|repair")] = "revalidate",
    agent_max_failures: Annotated[int, typer.Option("--agent-max-failures")] = 5,
    agent_repair_attempts: Annotated[int, typer.Option("--agent-repair-attempts")] = 1,
    ar_constant_file: Annotated[
        Optional[Path],
        typer.Option("--ar-constant-file", exists=True, help="JSON file produced by calibrate-ar"),
    ] = None,
    single_process: Annotated[bool, typer.Option("--single-process")] = False,
    include_smoke_in_adaptation: Annotated[bool, typer.Option("--include-smoke-in-adaptation")] = True,
    numeric_rtol: Annotated[float, typer.Option("--numeric-rtol")] = 1e-4,
    numeric_atol: Annotated[float, typer.Option("--numeric-atol")] = 1e-5,
) -> None:
    """One-command evaluation: generate/load adapter, record effort, run tests, write focused reports."""
    if adapter_file is None and docs is None:
        raise typer.BadParameter("Provide --docs for automatic adapter generation, or --adapter-file to reuse an adapter.")

    out.mkdir(parents=True, exist_ok=True)
    adapter_out = out / "adapter.generated.json"
    ledger_out = out / "effort_ledger.json"
    report_out = out / "reports"

    adapt_stats = MigrationEffortStats()
    sources: list[dict] = []
    measurement = "provided_adapter"

    if adapter_file is not None:
        spec = validate_adapter_dict(json.loads(adapter_file.read_text(encoding="utf-8")))
        if bridge_id:
            spec.library_name = bridge_id
        adapter_out.write_text(
            json.dumps(adapter_spec_to_dict(spec), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    else:
        doc_path = str(docs)
        text, sources = collect_docs(local_path=doc_path)
        if not text.strip():
            console.print("[red]No documentation text collected from --docs.[/red]")
            raise typer.Exit(2)
        spec = extract_adapter_with_llm(text, model=model, base_url=base_url)
        if bridge_id:
            spec.library_name = bridge_id
        adapter_out.write_text(
            json.dumps(adapter_spec_to_dict(spec), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        adapt_stats.llm_calls = 1
        adapt_stats.agent_rounds = 1
        adapt_stats.prompt_chars = len(text)
        generated = adapter_out.read_text(encoding="utf-8")
        adapt_stats.completion_chars = len(generated)
        adapt_stats.patch_lines_added = len(generated.splitlines())
        measurement = "local_char_proxy"

    ledger = _stats_to_effort_ledger(
        bridge_id=spec.library_name,
        phase="adapt",
        stats=adapt_stats,
        measurement=measurement,
    )
    ledger_out.write_text(json.dumps(ledger, indent=2, ensure_ascii=False), encoding="utf-8")

    summ, jp, mp = _run_evaluation(
        spec=spec,
        sources=sources,
        suite=suite,
        yes=yes,
        no_probe=no_probe,
        model=model,
        base_url=base_url,
        report_dir=report_out,
        honor_adapter_device=honor_adapter_device,
        adaptation_suites=suite if include_smoke_in_adaptation and suite != "all" else None,
        include_smoke_in_adaptation=include_smoke_in_adaptation,
        numeric_rtol=numeric_rtol,
        numeric_atol=numeric_atol,
        single_process=single_process,
        agent_mode=agent_mode,
        agent_max_failures=agent_max_failures,
        agent_repair_attempts=agent_repair_attempts,
        ar_constant_file=None,
    )

    agent_payload = summ.agent_report or AgentRunReport(enabled=False, mode="off").model_dump(mode="json")
    runtime_stats = MigrationEffortStats.model_validate(agent_payload.get("migration_effort") or {})
    total_stats = _add_effort_stats(adapt_stats, runtime_stats)
    adapt_effort = calc_effort_from_stats(adapt_stats)
    repair_effort = calc_effort_from_stats(runtime_stats)
    total_effort = calc_effort_from_stats(total_stats)
    agent_payload["enabled"] = True
    agent_payload["mode"] = agent_payload.get("mode") or agent_mode
    agent_payload["migration_effort"] = total_stats.model_dump(mode="json")
    repair_metrics = agent_payload.get("repair_metrics") or RepairMetrics().model_dump(mode="json")
    repair_metrics["effort_total"] = total_effort
    repair_metrics.setdefault("repair_success_rate", None)
    agent_payload["effort_breakdown"] = {
        "effort_adapt": adapt_effort,
        "effort_repair": repair_effort,
        "ME": total_effort,
        "measurement": measurement,
    }
    agent_payload["repair_metrics"] = repair_metrics
    summ.agent_report = agent_payload
    _apply_ar_constant(summary=summ, ar_constant_file=ar_constant_file, model=model)

    payload = build_report_payload(
        adapter=spec,
        hardware=probe_hardware(),
        summary=summ,
        doc_sources=sources,
    )
    final_json, final_md = write_reports(payload, out, spec.library_name, stamp="summary")
    summary_json = out / "summary.json"
    summary_md = out / "summary.md"
    shutil.copyfile(final_json, summary_json)
    shutil.copyfile(final_md, summary_md)

    console.print(f"[green]Wrote {adapter_out}[/green]")
    console.print(f"[green]Wrote {ledger_out}[/green]")
    console.print(f"[green]Wrote {summary_json}[/green]")
    console.print(f"[green]Wrote {summary_md}[/green]")

    es = summ.evaluation_summary or {}
    repair_metrics = (summ.agent_report or {}).get("repair_metrics") or {}
    console.print(
        json.dumps(
            {
                "summary_md": str(summary_md),
                "summary_json": str(summary_json),
                "adapter": str(adapter_out),
                "effort_ledger": str(ledger_out),
                "compatibility_rate": es.get("adaptation_rate_overall"),
                "ME": repair_metrics.get("effort_total"),
                "AR": repair_metrics.get("AR"),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    _emit_environment_issues(summ.agent_report)
    if _has_unresolved_failures(summ):
        raise typer.Exit(1)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
