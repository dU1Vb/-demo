"""Run summary types shared by runner and metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agentorskill_cli.test_suites.harness import EvalCaseResult


@dataclass
class SuiteRun:
    name: str
    cases: list[EvalCaseResult] = field(default_factory=list)
    passed: int = 0
    failed: int = 0
    skipped_n: int = 0
    skipped: bool = False


@dataclass
class RunSummary:
    suites: dict[str, SuiteRun] = field(default_factory=dict)
    numeric_delta: dict[str, Any] = field(default_factory=dict)
    numeric_fidelity: dict[str, Any] = field(default_factory=dict)
    smoke_failed: bool = False
    evaluation_summary: dict[str, Any] = field(default_factory=dict)
    agent_report: dict[str, Any] = field(default_factory=dict)
