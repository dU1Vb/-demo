"""Tests for evaluation summary aggregation."""

import json
from pathlib import Path

import pytest

from agentorskill_cli.metrics import compute_evaluation_summary, DEFAULT_ADAPTATION_SUITES
from agentorskill_cli.test_suites.harness import EvalCaseResult
from agentorskill_cli.test_suites.types_summary import RunSummary, SuiteRun


def _make_summary() -> RunSummary:
    s = RunSummary()
    sr = SuiteRun(name="core")
    sr.cases = [
        EvalCaseResult(
            case_id="a",
            name="n",
            ok=True,
            duration_s=1.0,
            error=None,
            category="tensor",
        ),
        EvalCaseResult(
            case_id="b",
            name="n2",
            ok=False,
            duration_s=1.0,
            error="e",
            category="tensor",
        ),
        EvalCaseResult(
            case_id="c",
            name="skip",
            ok=True,
            duration_s=1.0,
            error=None,
            category="nn",
            skipped=True,
        ),
    ]
    sr.passed = 1
    sr.failed = 1
    sr.skipped_n = 1
    s.suites["core"] = sr
    return s


def test_adaptation_rate_excludes_skipped() -> None:
    summ = _make_summary()
    out = compute_evaluation_summary(summ, adaptation_suites=frozenset({"core"}))
    assert out["counts"]["failed_n"] == 1
    assert out["counts"]["skipped_n"] == 1
    assert out["counts"]["passed"] == 1
    assert out["counts"]["total"] == 2
    assert out["adaptation_rate_overall"] == pytest.approx(0.5)


def test_default_suites_exclude_benchmark() -> None:
    assert "benchmark" not in DEFAULT_ADAPTATION_SUITES
