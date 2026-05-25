"""Agentic diagnostic, revalidation and adapter-repair loop."""

from __future__ import annotations

import difflib
import json
import time
from dataclasses import replace
from typing import Any

from pydantic import ValidationError

from agentorskill_cli.adapter_schema import AdapterSpec, adapter_spec_to_dict, validate_adapter_dict
from agentorskill_cli.agent_schema import (
    AgentRunReport,
    DiagnosticDecision,
    EnvironmentIssue,
    MigrationEffortStats,
    RepairAttempt,
    RepairMetrics,
    RevalidationAction,
    RevalidationResult,
)
from agentorskill_cli.llm_provider import OpenAIChatProvider
from agentorskill_cli.test_suites.harness import EvalCaseResult, TestCase, run_case
from agentorskill_cli.test_suites.runner import SUITE_MAP, execute_plan
from agentorskill_cli.test_suites.types_summary import RunSummary

AGENT_TEMPERATURE = 0.1
ENVIRONMENT_FAILURE_CLASSES = {"EnvironmentFailure", "DependencyMissing", "ImportOrderError"}


DIAGNOSTIC_SYSTEM_PROMPT = """You are a diagnostic agent for PyTorch-to-MindSpore migration evaluation.
Classify one failed test case. Return only JSON with:
{
  "failure_class": one of ["EnvironmentFailure","DependencyMissing","ImportOrderError","OperatorNotFound","TypeMismatch","ShapeMismatch","DeviceMismatch","AutogradFailure","NumericMismatch","RuntimeCrash"],
  "evidence": ["short concrete evidence strings"],
  "is_environment_or_config": boolean,
  "counts_toward_compatibility": boolean,
  "suggested_revalidation": [{"action": one of ["rerun_same_case_subprocess","rerun_case_without_adapter","rerun_case_honor_adapter_device","rerun_suite_subprocess"], "reason": "why"}],
  "recommended_next_step": "short action"
}

Rules:
- EnvironmentFailure, DependencyMissing and ImportOrderError usually do not count toward migration compatibility.
- OperatorNotFound, TypeMismatch, ShapeMismatch, DeviceMismatch, AutogradFailure and NumericMismatch usually count.
- RuntimeCrash counts unless evidence points to dependency, driver, device initialization or import-order problems.
- Only suggest actions from the allowed list.
"""


REPAIR_SYSTEM_PROMPT = """You are a repair agent for an AdapterSpec, not for arbitrary source code.
Given a failed case, diagnostic evidence and current AdapterSpec, propose ONE complete replacement AdapterSpec JSON.
Return only JSON with:
{
  "rationale": "why this adapter change may fix the case",
  "adapter": { ... complete AdapterSpec ... }
}

Rules:
- Only change adapter fields such as enable.preamble, device, install notes, constraints or extra.
- Do not invent large framework patches.
- The adapter.enable.preamble must be valid Python and non-empty.
- If no safe adapter repair exists, return the original adapter unchanged and explain why.
"""


def _failed_cases(summary: RunSummary, *, max_failures: int) -> list[tuple[str, EvalCaseResult]]:
    failed: list[tuple[str, EvalCaseResult]] = []
    for suite_name, suite in summary.suites.items():
        if suite.skipped:
            continue
        for case in suite.cases:
            if not case.ok and not case.skipped:
                failed.append((suite_name, case))
                if len(failed) >= max_failures:
                    return failed
    return failed


def _case_payload(suite_name: str, case: EvalCaseResult) -> dict[str, Any]:
    return {
        "suite": suite_name,
        "case_id": case.case_id,
        "name": case.name,
        "category": case.category,
        "error": case.error,
        "details": case.details,
        "stdout": case.stdout[-1000:],
        "stderr": case.stderr[-1000:],
    }


def _heuristic_diagnosis(suite_name: str, case: EvalCaseResult) -> DiagnosticDecision:
    text = " ".join(
        [
            str(case.error or ""),
            str(case.stderr or ""),
            json.dumps(case.details, ensure_ascii=False, default=str),
        ]
    ).lower()
    failure_class = "RuntimeCrash"
    env = False
    if "no module named" in text or "modulenotfounderror" in text:
        failure_class = "DependencyMissing"
        env = True
    elif "import" in text and ("before" in text or "order" in text):
        failure_class = "ImportOrderError"
        env = True
    elif "not implemented" in text or "operator" in text and "not" in text:
        failure_class = "OperatorNotFound"
    elif "shape" in text or "size mismatch" in text:
        failure_class = "ShapeMismatch"
    elif "dtype" in text or "type" in text:
        failure_class = "TypeMismatch"
    elif "device" in text or "npu" in text or "cuda" in text:
        failure_class = "DeviceMismatch"
    elif "grad" in text or "autograd" in text or "backward" in text:
        failure_class = "AutogradFailure"
    elif "numeric" in text or "allclose" in text or "tolerance" in text:
        failure_class = "NumericMismatch"

    actions = [RevalidationAction(action="rerun_same_case_subprocess", reason="Check whether the failure is stable.")]
    if failure_class == "DeviceMismatch":
        actions.append(
            RevalidationAction(
                action="rerun_case_honor_adapter_device",
                reason="Check whether adapter target device placement changes the result.",
            )
        )
    if env:
        actions.append(
            RevalidationAction(
                action="rerun_case_without_adapter",
                reason="Check whether the failure also appears without the migration adapter.",
            )
        )

    return DiagnosticDecision(
        suite=suite_name,
        case_id=case.case_id,
        failure_class=failure_class,  # type: ignore[arg-type]
        evidence=[str(case.error or "")[:500] or "No error text captured."],
        is_environment_or_config=env,
        counts_toward_compatibility=not env,
        suggested_revalidation=actions,
        recommended_next_step="Run controlled revalidation actions.",
    )


def _diagnose_with_llm(
    provider: OpenAIChatProvider,
    *,
    adapter: AdapterSpec,
    suite_name: str,
    case: EvalCaseResult,
) -> tuple[DiagnosticDecision, dict[str, Any]]:
    user_prompt = json.dumps(
        {
            "adapter": adapter_spec_to_dict(adapter),
            "failed_case": _case_payload(suite_name, case),
        },
        indent=2,
        ensure_ascii=False,
        default=str,
    )
    resp = provider.complete_json(
        system_prompt=DIAGNOSTIC_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        temperature=AGENT_TEMPERATURE,
    )
    payload = dict(resp.data)
    payload["suite"] = suite_name
    payload["case_id"] = case.case_id
    payload["prompt"] = user_prompt
    payload["raw_response"] = resp.raw_text
    return DiagnosticDecision.model_validate(payload), {
        "provider": resp.provider,
        "protocol": resp.protocol,
        "model": resp.model,
    }


def _find_test_case(suite_name: str, case_id: str) -> TestCase | None:
    cases = SUITE_MAP.get(suite_name)
    if not cases:
        return None
    for case in cases:
        if case.id == case_id:
            return case
    return None


def _run_revalidation_action(
    *,
    adapter: AdapterSpec,
    suite_name: str,
    case_id: str,
    action: RevalidationAction,
    honor_device: bool,
) -> RevalidationResult:
    t0 = time.perf_counter()
    test_case = _find_test_case(suite_name, case_id)
    if test_case is None:
        return RevalidationResult(
            suite=suite_name,
            case_id=case_id,
            action=action.action,
            ok=False,
            conclusion="not_run",
            error="case not found for controlled revalidation",
        )

    result: EvalCaseResult | None = None
    if action.action == "rerun_same_case_subprocess":
        result = run_case(adapter, test_case, honor_device=honor_device)
    elif action.action == "rerun_case_without_adapter":
        result = run_case(adapter, replace(test_case, include_adapter=False), honor_device=False)
    elif action.action == "rerun_case_honor_adapter_device":
        result = run_case(adapter, test_case, honor_device=True)
    elif action.action == "rerun_suite_subprocess":
        suite_summary = execute_plan(adapter, suite_name, honor_device=honor_device)
        suite = suite_summary.suites.get(suite_name)
        if suite:
            for case in suite.cases:
                if case.case_id == case_id:
                    result = case
                    break

    if result is None:
        return RevalidationResult(
            suite=suite_name,
            case_id=case_id,
            action=action.action,
            ok=False,
            duration_s=time.perf_counter() - t0,
            conclusion="not_run",
            error="action did not produce a case result",
        )

    return RevalidationResult(
        suite=suite_name,
        case_id=case_id,
        action=action.action,
        ok=result.ok,
        duration_s=time.perf_counter() - t0,
        error=result.error,
        conclusion="unstable_or_environment" if result.ok else "stable_failure",
        details={
            "reason": action.reason,
            "category": result.category,
            "case_duration_s": result.duration_s,
            "details": result.details,
        },
    )


def _adapter_diff(before: AdapterSpec, after: AdapterSpec) -> str:
    old = json.dumps(adapter_spec_to_dict(before), indent=2, ensure_ascii=False).splitlines()
    new = json.dumps(adapter_spec_to_dict(after), indent=2, ensure_ascii=False).splitlines()
    return "\n".join(
        difflib.unified_diff(old, new, fromfile="adapter.before.json", tofile="adapter.after.json", lineterm="")
    )


def _count_patch_lines(diff: str) -> tuple[int, int]:
    added = 0
    deleted = 0
    for line in diff.splitlines():
        if line.startswith("+++") or line.startswith("---"):
            continue
        if line.startswith("+"):
            added += 1
        elif line.startswith("-"):
            deleted += 1
    return added, deleted


def _repair_case(
    provider: OpenAIChatProvider,
    *,
    adapter: AdapterSpec,
    suite_name: str,
    case: EvalCaseResult,
    diagnostic: DiagnosticDecision,
    attempt_index: int,
    honor_device: bool,
    effort: MigrationEffortStats,
) -> RepairAttempt:
    t0 = time.perf_counter()
    user_prompt = json.dumps(
        {
            "current_adapter": adapter_spec_to_dict(adapter),
            "diagnostic": diagnostic.model_dump(mode="json"),
            "failed_case": _case_payload(suite_name, case),
        },
        indent=2,
        ensure_ascii=False,
        default=str,
    )
    try:
        resp = provider.complete_json(
            system_prompt=REPAIR_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=AGENT_TEMPERATURE,
        )
        effort.llm_calls += 1
        effort.agent_rounds += 1
        effort.prompt_chars += resp.prompt_chars
        effort.completion_chars += resp.completion_chars
    except Exception as exc:
        return RepairAttempt(
            suite=suite_name,
            case_id=case.case_id,
            attempt_index=attempt_index,
            ok=False,
            validation_error=str(exc),
            duration_s=time.perf_counter() - t0,
            prompt=user_prompt,
        )

    rationale = str(resp.data.get("rationale", ""))
    try:
        patched = validate_adapter_dict(resp.data.get("adapter") or {})
    except ValidationError as exc:
        return RepairAttempt(
            suite=suite_name,
            case_id=case.case_id,
            attempt_index=attempt_index,
            ok=False,
            rationale=rationale,
            validation_error=str(exc),
            duration_s=time.perf_counter() - t0,
            prompt=user_prompt,
            raw_response=resp.raw_text,
        )

    diff = _adapter_diff(adapter, patched)
    effort.patches_generated += 1
    added, deleted = _count_patch_lines(diff)
    effort.patch_lines_added += added
    effort.patch_lines_deleted += deleted

    test_case = _find_test_case(suite_name, case.case_id)
    if test_case is None:
        test_error = "case not found for repair validation"
        ok = False
    else:
        validation = run_case(patched, test_case, honor_device=honor_device)
        effort.revalidation_runs += 1
        ok = validation.ok
        test_error = validation.error

    return RepairAttempt(
        suite=suite_name,
        case_id=case.case_id,
        attempt_index=attempt_index,
        ok=ok,
        rationale=rationale,
        patch_unified_diff=diff,
        patched_adapter=adapter_spec_to_dict(patched),
        test_error=test_error,
        duration_s=time.perf_counter() - t0,
        prompt=user_prompt,
        raw_response=resp.raw_text,
    )


def _suggest_user_actions(diag: DiagnosticDecision, case: EvalCaseResult) -> list[str]:
    text = " ".join(
        [
            str(case.error or ""),
            str(case.stderr or ""),
            json.dumps(case.details, ensure_ascii=False, default=str),
        ]
    ).lower()

    actions: list[str] = []
    if diag.failure_class == "DependencyMissing":
        actions.append("Install the missing dependency in the evaluation environment and rerun the failed suite.")
    elif diag.failure_class == "ImportOrderError":
        actions.append("Adjust import order or adapter preamble initialization sequence, then rerun the failed suite.")
    elif diag.failure_class == "EnvironmentFailure":
        actions.append("Verify the base Python/runtime environment before evaluating bridge compatibility.")

    if "cuda" in text or "npu" in text or "ascend" in text:
        actions.append("Check device runtime availability, drivers, and backend initialization on the target machine.")
    if not actions:
        actions.append("Inspect the captured error and environment configuration, then rerun the failed case.")
    return actions


def _environment_issue_from_case(
    suite_name: str,
    case: EvalCaseResult,
    diag: DiagnosticDecision,
    actions: list[RevalidationAction],
) -> EnvironmentIssue:
    return EnvironmentIssue(
        suite=suite_name,
        case_id=case.case_id,
        failure_class=diag.failure_class,
        resolved=False,
        attempted_actions=[a.action for a in actions],
        evidence=list(diag.evidence),
        suggested_user_actions=_suggest_user_actions(diag, case),
        final_error=case.error,
    )


def _compute_repair_metrics(attempts: list[RepairAttempt], max_attempts: int) -> RepairMetrics:
    by_case: dict[tuple[str, str], list[RepairAttempt]] = {}
    for attempt in attempts:
        by_case.setdefault((attempt.suite, attempt.case_id), []).append(attempt)

    attempted_cases = len(by_case)
    success_attempt_by_case: dict[tuple[str, str], int] = {}
    for key, case_attempts in by_case.items():
        successes = [a.attempt_index for a in case_attempts if a.ok]
        if successes:
            success_attempt_by_case[key] = min(successes)

    successful = len(success_attempt_by_case)
    migrate_at_k: dict[str, float | None] = {}
    for k in range(1, max_attempts + 1):
        if attempted_cases == 0:
            migrate_at_k[f"migrate@{k}"] = None
        else:
            migrate_at_k[f"migrate@{k}"] = sum(1 for idx in success_attempt_by_case.values() if idx <= k) / attempted_cases

    repair_rate = (successful / attempted_cases) if attempted_cases else None
    return RepairMetrics(
        attempted_cases=attempted_cases,
        successful_repairs=successful,
        repair_success_rate=repair_rate,
        AR=None,
        migrate_at_k=migrate_at_k,
    )


def run_agent_loop(
    *,
    adapter: AdapterSpec,
    summary: RunSummary,
    mode: str,
    model: str | None = None,
    base_url: str | None = None,
    max_failures: int = 5,
    repair_attempts: int = 0,
    honor_device: bool = False,
) -> AgentRunReport:
    """Run diagnostic -> revalidation -> adapter repair stages on failed cases."""
    mode = mode.lower().strip()
    if mode not in {"off", "diagnose", "revalidate", "repair"}:
        raise ValueError("agent mode must be one of: off, diagnose, revalidate, repair")
    if mode == "off":
        return AgentRunReport(enabled=False, mode="off")

    t0 = time.perf_counter()
    effort = MigrationEffortStats()
    report = AgentRunReport(enabled=True, mode=mode, migration_effort=effort)
    failed = _failed_cases(summary, max_failures=max_failures)
    if not failed:
        report.migration_effort.elapsed_s = time.perf_counter() - t0
        return report

    try:
        provider = OpenAIChatProvider(model=model, base_url=base_url)
    except RuntimeError as exc:
        report.diagnostics.append(
            DiagnosticDecision(
                suite="agent",
                case_id="agent_mode_bootstrap",
                failure_class="EnvironmentFailure",
                evidence=[str(exc)],
                is_environment_or_config=True,
                counts_toward_compatibility=False,
                suggested_revalidation=[],
                recommended_next_step="Set OPENAI_API_KEY or rerun with --agent-mode off.",
            )
        )
        report.environment_issues.append(
            EnvironmentIssue(
                suite="agent",
                case_id="agent_mode_bootstrap",
                failure_class="EnvironmentFailure",
                resolved=False,
                attempted_actions=[],
                evidence=[str(exc)],
                suggested_user_actions=[
                    "Set OPENAI_API_KEY before enabling agent mode, or rerun with --agent-mode off.",
                ],
                final_error=str(exc),
            )
        )
        report.migration_effort.elapsed_s = time.perf_counter() - t0
        return report
    report.provider = provider.provider
    report.protocol = provider.protocol
    report.model = provider.model
    report.temperature = AGENT_TEMPERATURE

    diagnostic_by_case: dict[tuple[str, str], DiagnosticDecision] = {}
    environment_cases: set[tuple[str, str]] = set()

    for suite_name, case in failed:
        try:
            diag, provider_meta = _diagnose_with_llm(
                provider,
                adapter=adapter,
                suite_name=suite_name,
                case=case,
            )
            report.provider = provider_meta["provider"]
            report.protocol = provider_meta["protocol"]
            report.model = provider_meta["model"]
        except Exception:
            diag = _heuristic_diagnosis(suite_name, case)
        diagnostic_by_case[(suite_name, case.case_id)] = diag
        report.diagnostics.append(diag)

        case.details["agent_diagnostic"] = diag.model_dump(mode="json")
        if not diag.counts_toward_compatibility:
            case.counts_toward_adaptation = False
            case.details["agent_outcome"] = "excluded_from_compatibility"
            environment_cases.add((suite_name, case.case_id))
            report.environment_issues.append(
                _environment_issue_from_case(
                    suite_name,
                    case,
                    diag,
                    diag.suggested_revalidation,
                )
            )

    if mode in {"revalidate", "repair"}:
        for suite_name, case in failed:
            diag = diagnostic_by_case[(suite_name, case.case_id)]
            actions = diag.suggested_revalidation or [
                RevalidationAction(action="rerun_same_case_subprocess", reason="Default stability check.")
            ]
            is_environment_case = (suite_name, case.case_id) in environment_cases
            for action in actions[:3]:
                rv = _run_revalidation_action(
                    adapter=adapter,
                    suite_name=suite_name,
                    case_id=case.case_id,
                    action=action,
                    honor_device=honor_device,
                )
                if not is_environment_case:
                    effort.revalidation_runs += 1
                report.revalidations.append(rv)
                report.tool_calls.append(
                    {
                        "tool": "revalidation",
                        "action": action.action,
                        "suite": suite_name,
                        "case_id": case.case_id,
                        "ok": rv.ok,
                        "duration_s": rv.duration_s,
                    }
                )
                if rv.conclusion == "unstable_or_environment":
                    case.counts_toward_adaptation = False
                    case.details["agent_outcome"] = "excluded_after_revalidation"
                    case.details["agent_revalidation_conclusion"] = rv.model_dump(mode="json")
                    if is_environment_case:
                        for issue in report.environment_issues:
                            if issue.suite == suite_name and issue.case_id == case.case_id:
                                issue.resolved = True
                                issue.final_error = None
                    break
            if is_environment_case:
                for issue in report.environment_issues:
                    if issue.suite == suite_name and issue.case_id == case.case_id and not issue.resolved:
                        issue.final_error = case.error

    if mode == "repair":
        attempts_n = max(1, repair_attempts)
        for suite_name, case in failed:
            diag = diagnostic_by_case[(suite_name, case.case_id)]
            if not diag.counts_toward_compatibility or not case.counts_toward_adaptation:
                continue
            for attempt_idx in range(1, attempts_n + 1):
                effort.repair_attempts += 1
                attempt = _repair_case(
                    provider,
                    adapter=adapter,
                    suite_name=suite_name,
                    case=case,
                    diagnostic=diag,
                    attempt_index=attempt_idx,
                    honor_device=honor_device,
                    effort=effort,
                )
                report.repair_attempts.append(attempt)
                report.tool_calls.append(
                    {
                        "tool": "adapter_repair",
                        "suite": suite_name,
                        "case_id": case.case_id,
                        "attempt_index": attempt_idx,
                        "ok": attempt.ok,
                        "duration_s": attempt.duration_s,
                    }
                )
                if attempt.ok:
                    case.details["agent_successful_repair"] = attempt.model_dump(mode="json")
                    case.details["agent_outcome"] = "repaired_by_adapter_patch"
                    break
        report.repair_metrics = _compute_repair_metrics(report.repair_attempts, attempts_n)
    else:
        report.repair_metrics = _compute_repair_metrics([], max(1, repair_attempts))

    effort.elapsed_s = time.perf_counter() - t0
    return report
