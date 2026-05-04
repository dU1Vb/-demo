"""Orchestrate suites with tiered gates, numeric fidelity, and evaluation summary."""

from __future__ import annotations

import math
from typing import Any

from agentorskill_cli.adapter_schema import AdapterSpec
from agentorskill_cli.metrics import compute_evaluation_summary
from agentorskill_cli.test_suites.benchmark import BENCH_CASES
from agentorskill_cli.test_suites.core import CORE_CASES
from agentorskill_cli.test_suites.harness import run_case
from agentorskill_cli.test_suites.models import MODEL_CASES
from agentorskill_cli.test_suites.numeric import NUMERIC_CASES
from agentorskill_cli.test_suites.smoke import SMOKE_CASES
from agentorskill_cli.test_suites.training_loop import TRAINING_CASES
from agentorskill_cli.test_suites.types_summary import RunSummary, SuiteRun

DEFAULT_RTOL = 1e-4
DEFAULT_ATOL = 1e-5


def _vec_metrics(a: list[float], b: list[float]) -> dict[str, Any]:
    if len(a) != len(b) or not a:
        return {"match": False, "reason": "length mismatch"}
    diffs = [abs(x - y) for x, y in zip(a, b, strict=True)]
    max_abs = max(diffs)
    mean_abs = sum(diffs) / len(diffs)
    rmse = math.sqrt(sum(d * d for d in diffs) / len(diffs))
    return {"max_abs_err": max_abs, "mean_abs_err": mean_abs, "rmse": rmse, "n": len(a)}


def _allclose_vecs(a: list[float], b: list[float], rtol: float, atol: float) -> bool:
    for x, y in zip(a, b, strict=True):
        limit = atol + rtol * max(abs(x), abs(y))
        if abs(x - y) > limit:
            return False
    return True


def run_numeric_aggregate(
    adapter: AdapterSpec,
    *,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
    honor_device: bool = False,
) -> tuple[SuiteRun, dict[str, Any], dict[str, Any]]:
    baseline_result: dict[str, Any] | None = None
    adapter_result: dict[str, Any] | None = None
    sr = SuiteRun(name="numeric")
    for tc in NUMERIC_CASES:
        tr = run_case(adapter, tc, honor_device=honor_device)
        sr.cases.append(tr)
        if tr.skipped:
            sr.skipped_n += 1
        elif tr.ok:
            sr.passed += 1
        else:
            sr.failed += 1
        if tc.id == "numeric_baseline_ref" and tr.details:
            baseline_result = tr.details
        if tc.id == "numeric_with_adapter" and tr.details:
            adapter_result = tr.details

    delta: dict[str, Any] = {}
    fidelity: dict[str, Any] = {"baseline_ok": False, "rtol": rtol, "atol": atol}

    if baseline_result and adapter_result and "y_flat" in baseline_result and "y_flat" in adapter_result:
        ym = _vec_metrics(adapter_result["y_flat"], baseline_result["y_flat"])
        gm = _vec_metrics(adapter_result["g_flat"], baseline_result["g_flat"])
        delta["y_metrics"] = ym
        delta["g_metrics"] = gm
        if "max_abs_err" not in ym or "max_abs_err" not in gm:
            delta["baseline_ok"] = False
            delta["reason"] = ym.get("reason") or gm.get("reason") or "vector mismatch"
            fidelity["error"] = delta["reason"]
        else:
            delta["baseline_ok"] = True
            y_ok = _allclose_vecs(adapter_result["y_flat"], baseline_result["y_flat"], rtol, atol)
            g_ok = _allclose_vecs(adapter_result["g_flat"], baseline_result["g_flat"], rtol, atol)
            fidelity["within_tolerance_y"] = y_ok
            fidelity["within_tolerance_g"] = g_ok
            fidelity["within_tolerance_all"] = y_ok and g_ok
            fidelity["max_abs_err_y"] = ym["max_abs_err"]
            fidelity["max_abs_err_g"] = gm["max_abs_err"]
            fidelity["rmse_y"] = ym["rmse"]
            fidelity["rmse_g"] = gm["rmse"]
    else:
        delta["baseline_ok"] = False
        delta["reason"] = "missing baseline or adapter vectors"

    return sr, delta, fidelity


def run_cases(
    name: str,
    adapter: AdapterSpec,
    cases: list,
    *,
    honor_device: bool = False,
) -> SuiteRun:
    sr = SuiteRun(name=name)
    for tc in cases:
        tr = run_case(adapter, tc, honor_device=honor_device)
        sr.cases.append(tr)
        if tr.skipped:
            sr.skipped_n += 1
        elif tr.ok:
            sr.passed += 1
        else:
            sr.failed += 1
    return sr


SUITE_MAP = {
    "smoke": SMOKE_CASES,
    "core": CORE_CASES,
    "models": MODEL_CASES,
    "training": TRAINING_CASES,
    "benchmark": BENCH_CASES,
}


def parse_adaptation_suites_arg(s: str | None) -> frozenset[str] | None:
    if not s or not s.strip():
        return None
    parts = {p.strip().lower() for p in s.split(",") if p.strip()}
    return frozenset(parts)


def execute_plan(
    adapter: AdapterSpec,
    suite: str,
    *,
    skip_benchmark_if_smoke_fails: bool = True,
    honor_device: bool = False,
    adaptation_suites: frozenset[str] | None = None,
    include_smoke_in_adaptation: bool = False,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
) -> RunSummary:
    summary = RunSummary()
    name = suite.lower().strip()
    if name == "all":
        order = ["smoke", "core", "models", "training", "numeric", "benchmark"]
    elif name == "numeric":
        order = ["numeric"]
    else:
        order = [name]

    for sn in order:
        if sn == "numeric":
            sr, delta, fidelity = run_numeric_aggregate(
                adapter, rtol=rtol, atol=atol, honor_device=honor_device
            )
            summary.suites["numeric"] = sr
            summary.numeric_delta = delta
            summary.numeric_fidelity = fidelity
            continue

        cases = SUITE_MAP.get(sn)
        if not cases:
            summary.suites[sn] = SuiteRun(name=sn, skipped=True)
            continue

        if sn == "benchmark" and skip_benchmark_if_smoke_fails:
            sm = summary.suites.get("smoke")
            if sm and sm.failed > 0:
                summary.suites["benchmark"] = SuiteRun(name="benchmark", skipped=True)
                continue

        sr = run_cases(sn, adapter, cases, honor_device=honor_device)
        summary.suites[sn] = sr
        if sn == "smoke" and sr.failed > 0:
            summary.smoke_failed = True

    if name == "all" and summary.smoke_failed and skip_benchmark_if_smoke_fails:
        if "benchmark" not in summary.suites:
            summary.suites["benchmark"] = SuiteRun(name="benchmark", skipped=True)

    summ = compute_evaluation_summary(
        summary,
        adaptation_suites=adaptation_suites,
        include_smoke_in_adaptation=include_smoke_in_adaptation,
    )
    summary.evaluation_summary = summ

    return summary
