"""Tests for the agent diagnostic/revalidation/repair loop."""

from __future__ import annotations

import pytest

from agentorskill_cli.adapter_schema import validate_adapter_dict
from agentorskill_cli.ar_calibration import measure_baseline_effort
from agentorskill_cli.agent_loop import run_agent_loop
from agentorskill_cli.agent_schema import MigrationEffortStats
from agentorskill_cli.effort import calc_ar, calc_effort_from_stats
from agentorskill_cli.main import _has_unresolved_failures
from agentorskill_cli.llm_provider import LLMJsonResponse
from agentorskill_cli.test_suites.harness import EvalCaseResult
from agentorskill_cli.test_suites.types_summary import RunSummary, SuiteRun


def _adapter():
    return validate_adapter_dict(
        {
            "library_name": "baseline_pytorch",
            "enable": {"preamble": "import torch", "import_order_note": ""},
            "device": {"target_device": "cpu"},
        }
    )


def _bad_adapter():
    return validate_adapter_dict(
        {
            "library_name": "bad_adapter",
            "enable": {"preamble": "raise RuntimeError('bad preamble')", "import_order_note": ""},
            "device": {"target_device": "cpu"},
        }
    )


def _failed_summary(case_id: str = "smoke_import_torch") -> tuple[RunSummary, EvalCaseResult]:
    case = EvalCaseResult(
        case_id=case_id,
        name="failed case",
        ok=False,
        duration_s=0.1,
        error="ModuleNotFoundError: No module named 'missing_dep'",
        category="smoke",
    )
    suite = SuiteRun(name="smoke", cases=[case], failed=1)
    summary = RunSummary(suites={"smoke": suite})
    return summary, case


class _FakeProvider:
    provider = "fake"
    protocol = "chat_completions"
    model = "fake-model"

    def __init__(self, *args, **kwargs) -> None:
        self.calls = 0

    def complete_json(self, *, system_prompt: str, user_prompt: str, temperature: float = 0.1):
        self.calls += 1
        if "propose ONE complete replacement AdapterSpec" in system_prompt:
            data = {
                "rationale": "No source patch needed; use native torch preamble.",
                "adapter": {
                    "library_name": "baseline_pytorch",
                    "enable": {"preamble": "import torch", "import_order_note": ""},
                    "device": {"target_device": "cpu"},
                },
            }
        else:
            data = {
                "failure_class": "DependencyMissing",
                "evidence": ["missing dependency"],
                "is_environment_or_config": True,
                "counts_toward_compatibility": False,
                "suggested_revalidation": [
                    {"action": "rerun_same_case_subprocess", "reason": "stability check"}
                ],
                "recommended_next_step": "install dependency or revalidate environment",
            }
        return LLMJsonResponse(
            data=data,
            raw_text="{}",
            model=self.model,
            provider=self.provider,
            protocol=self.protocol,
            prompt_chars=len(system_prompt) + len(user_prompt),
            completion_chars=2,
        )


class _RepairProvider(_FakeProvider):
    def complete_json(self, *, system_prompt: str, user_prompt: str, temperature: float = 0.1):
        resp = super().complete_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
        )
        if "propose ONE complete replacement AdapterSpec" not in system_prompt:
            resp.data["counts_toward_compatibility"] = True
            resp.data["is_environment_or_config"] = False
        return resp


class _TranslationProvider:
    provider = "fake"
    protocol = "chat_completions"
    model = "fake-model"

    def complete_text(self, *, system_prompt: str, user_prompt: str, temperature: float = 0.1):
        text = "import mindspore as ms\n"
        return type(
            "Resp",
            (),
            {
                "text": text,
                "model": self.model,
                "provider": self.provider,
                "protocol": self.protocol,
                "prompt_chars": len(system_prompt) + len(user_prompt),
                "completion_chars": len(text),
            },
        )()


def test_agent_no_failures_does_not_require_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    summary = RunSummary(suites={"smoke": SuiteRun(name="smoke", passed=1)})
    report = run_agent_loop(adapter=_adapter(), summary=summary, mode="diagnose")
    assert report.enabled
    assert report.diagnostics == []
    assert report.migration_effort.llm_calls == 0


def test_agent_diagnose_marks_environment_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("agentorskill_cli.agent_loop.OpenAIChatProvider", _FakeProvider)
    summary, case = _failed_summary()
    report = run_agent_loop(adapter=_adapter(), summary=summary, mode="diagnose")
    assert report.diagnostics[0].failure_class == "DependencyMissing"
    assert report.diagnostics[0].counts_toward_compatibility is False
    assert case.counts_toward_adaptation is False
    assert report.environment_issues[0].resolved is False
    assert report.environment_issues[0].failure_class == "DependencyMissing"
    assert report.migration_effort.llm_calls == 0
    assert _has_unresolved_failures(summary) is False


def test_agent_repair_tracks_ar_and_migrate_at_k(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("torch")
    monkeypatch.setattr("agentorskill_cli.agent_loop.OpenAIChatProvider", _RepairProvider)
    summary, _ = _failed_summary("smoke_import_torch")
    report = run_agent_loop(
        adapter=_bad_adapter(),
        summary=summary,
        mode="repair",
        repair_attempts=2,
    )
    assert report.repair_metrics.attempted_cases == 1
    assert report.repair_metrics.successful_repairs == 1
    assert report.repair_metrics.AR is None
    assert report.repair_metrics.migrate_at_k["migrate@1"] == pytest.approx(1.0)
    assert report.migration_effort.repair_attempts == 1
    assert _has_unresolved_failures(summary) is False


def test_environment_revalidation_does_not_count_toward_effort(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("agentorskill_cli.agent_loop.OpenAIChatProvider", _FakeProvider)
    summary, _ = _failed_summary()
    report = run_agent_loop(adapter=_adapter(), summary=summary, mode="revalidate")
    assert report.environment_issues
    assert report.migration_effort.revalidation_runs == 0


def test_repair_metrics_exposes_success_rate_and_ar(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("agentorskill_cli.agent_loop.OpenAIChatProvider", _RepairProvider)
    summary, _ = _failed_summary("smoke_import_torch")
    report = run_agent_loop(
        adapter=_bad_adapter(),
        summary=summary,
        mode="repair",
        repair_attempts=1,
    )
    assert report.repair_metrics.repair_success_rate == pytest.approx(1.0)
    assert report.repair_metrics.AR is None


def test_baseline_effort_is_measured_per_reroll() -> None:
    result = measure_baseline_effort(
        provider=_TranslationProvider(),
        task_cases=[
            {"case_id": "a", "code": "print('a')"},
            {"case_id": "b", "code": "print('b')"},
        ],
        rerolls=2,
        max_cv=1.0,
    )
    assert result.rerolls[0].case_count == 2
    assert len(result.rerolls) == 2
    assert result.baseline_effort == pytest.approx(result.rerolls[0].total_effort)


def test_ar_formula_uses_shared_effort_helper() -> None:
    stats = MigrationEffortStats(
        llm_calls=2,
        revalidation_runs=1,
        repair_attempts=3,
        prompt_chars=10,
        completion_chars=5,
        patch_lines_added=2,
        patch_lines_deleted=1,
    )
    me = calc_effort_from_stats(stats)
    assert me == pytest.approx(6.018)
    assert calc_ar(me, 12.0) == pytest.approx(0.4985)
