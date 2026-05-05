"""Typer CLI: probe, extract, run."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.table import Table

from agentorskill_cli.adapter_schema import AdapterSpec, validate_adapter_dict
from agentorskill_cli.doc_reader import collect_docs
from agentorskill_cli.hardware_probe import format_hardware_summary, probe_hardware, user_confirms
from agentorskill_cli.llm_adapter import extract_adapter_with_llm
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
) -> None:
    """Probe hardware, load/extract adapter, run suite(s), write reports."""
    if not no_probe:
        hw = probe_hardware()
        console.print(format_hardware_summary(hw))
        if not yes:
            if not user_confirms("Confirm this machine for evaluation?"):
                console.print("Aborted.")
                raise typer.Exit(1)
    else:
        hw = probe_hardware()

    spec, sources = _load_adapter(adapter_file, readme_url, doc_url, local_path, extra_text, model, base_url)
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

    payload = build_report_payload(
        adapter=spec,
        hardware=hw,
        summary=summ,
        doc_sources=sources,
    )
    jp, mp = write_reports(payload, report_dir, spec.library_name)
    console.print(f"[green]Wrote {jp}[/green]")
    console.print(f"[green]Wrote {mp}[/green]")

    es = summ.evaluation_summary or {}
    console.print(
        f"[bold]Adaptation rate (overall):[/bold] {es.get('adaptation_rate_overall')}"
    )

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
    if summ.suites and any(not v.skipped and v.failed for v in summ.suites.values()):
        raise typer.Exit(1)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
