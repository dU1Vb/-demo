"""Schemas for diagnostic, revalidation and repair agent reports."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


FailureClass = Literal[
    "EnvironmentFailure",
    "DependencyMissing",
    "ImportOrderError",
    "OperatorNotFound",
    "TypeMismatch",
    "ShapeMismatch",
    "DeviceMismatch",
    "AutogradFailure",
    "NumericMismatch",
    "RuntimeCrash",
]

RevalidationActionName = Literal[
    "rerun_same_case_subprocess",
    "rerun_case_without_adapter",
    "rerun_case_honor_adapter_device",
    "rerun_suite_subprocess",
]


class RevalidationAction(BaseModel):
    action: RevalidationActionName
    reason: str = ""


class DiagnosticDecision(BaseModel):
    suite: str
    case_id: str
    failure_class: FailureClass
    evidence: list[str] = Field(default_factory=list)
    is_environment_or_config: bool = False
    counts_toward_compatibility: bool = True
    suggested_revalidation: list[RevalidationAction] = Field(default_factory=list)
    recommended_next_step: str = ""
    prompt: str = ""
    raw_response: str = ""


class RevalidationResult(BaseModel):
    suite: str
    case_id: str
    action: RevalidationActionName
    ok: bool
    duration_s: float = 0.0
    error: str | None = None
    conclusion: Literal["stable_failure", "unstable_or_environment", "not_run"] = "not_run"
    details: dict[str, Any] = Field(default_factory=dict)


class RepairAttempt(BaseModel):
    suite: str
    case_id: str
    attempt_index: int
    ok: bool
    rationale: str = ""
    patch_unified_diff: str = ""
    patched_adapter: dict[str, Any] | None = None
    validation_error: str | None = None
    test_error: str | None = None
    duration_s: float = 0.0
    prompt: str = ""
    raw_response: str = ""


class MigrationEffortStats(BaseModel):
    agent_rounds: int = 0
    llm_calls: int = 0
    prompt_chars: int = 0
    completion_chars: int = 0
    revalidation_runs: int = 0
    repair_attempts: int = 0
    patches_generated: int = 0
    patch_lines_added: int = 0
    patch_lines_deleted: int = 0
    elapsed_s: float = 0.0


class RepairMetrics(BaseModel):
    attempted_cases: int = 0
    successful_repairs: int = 0
    repair_success_rate: float | None = None
    AR: float | None = None
    baseline_effort: float | None = None
    effort_total: float | None = None
    ar_constant_model: str | None = None
    ar_agent_system_version: str | None = None
    ar_constant_compatible: bool | None = None
    ar_constant_issues: list[str] = Field(default_factory=list)
    migrate_at_k: dict[str, float | None] = Field(default_factory=dict)


class EnvironmentIssue(BaseModel):
    suite: str
    case_id: str
    failure_class: FailureClass
    resolved: bool = False
    attempted_actions: list[RevalidationActionName] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    suggested_user_actions: list[str] = Field(default_factory=list)
    final_error: str | None = None


class AgentRunReport(BaseModel):
    enabled: bool = False
    mode: Literal["off", "diagnose", "revalidate", "repair"] = "off"
    provider: str | None = None
    protocol: str | None = None
    model: str | None = None
    temperature: float | None = None
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    diagnostics: list[DiagnosticDecision] = Field(default_factory=list)
    revalidations: list[RevalidationResult] = Field(default_factory=list)
    repair_attempts: list[RepairAttempt] = Field(default_factory=list)
    environment_issues: list[EnvironmentIssue] = Field(default_factory=list)
    migration_effort: MigrationEffortStats = Field(default_factory=MigrationEffortStats)
    repair_metrics: RepairMetrics = Field(default_factory=RepairMetrics)
