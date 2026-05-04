"""Aggregate adaptation rates and breakdowns from suite runs."""

from __future__ import annotations

from typing import Any

from agentorskill_cli.test_suites.types_summary import RunSummary, SuiteRun

DEFAULT_ADAPTATION_SUITES = frozenset({"core", "models", "training", "numeric"})


def compute_evaluation_summary(
    summary: RunSummary,
    *,
    adaptation_suites: frozenset[str] | None = None,
    include_smoke_in_adaptation: bool = False,
) -> dict[str, Any]:
    """
    Unweighted adaptation rate: passed / (total - skipped) per category and suite.
    Respects counts_toward_adaptation on each case.
    """
    suites_for_rate = adaptation_suites if adaptation_suites is not None else DEFAULT_ADAPTATION_SUITES
    if include_smoke_in_adaptation:
        suites_for_rate = frozenset(suites_for_rate | {"smoke"})

    category_pass: dict[str, int] = {}
    category_total: dict[str, int] = {}
    suite_pass: dict[str, int] = {}
    suite_total: dict[str, int] = {}
    failed_ids: list[str] = []
    skipped_ids: list[str] = []

    overall_pass = 0
    overall_total = 0

    for sname, sr in summary.suites.items():
        if sr.skipped or sname not in suites_for_rate:
            continue
        for c in sr.cases:
            if not c.counts_toward_adaptation:
                continue
            if c.skipped:
                skipped_ids.append(c.case_id)
                continue
            cat = c.category or "misc"
            category_total[cat] = category_total.get(cat, 0) + 1
            suite_total[sname] = suite_total.get(sname, 0) + 1
            overall_total += 1
            if c.ok:
                category_pass[cat] = category_pass.get(cat, 0) + 1
                suite_pass[sname] = suite_pass.get(sname, 0) + 1
                overall_pass += 1
            else:
                failed_ids.append(c.case_id)

    def _rate(p: int, t: int) -> float | None:
        if t <= 0:
            return None
        return p / t

    by_category: dict[str, Any] = {}
    for cat, tt in category_total.items():
        pp = category_pass.get(cat, 0)
        by_category[cat] = {
            "adaptation_rate": _rate(pp, tt),
            "passed": pp,
            "total": tt,
        }

    by_suite: dict[str, Any] = {}
    for sn, tt in suite_total.items():
        pp = suite_pass.get(sn, 0)
        by_suite[sn] = {
            "adaptation_rate": _rate(pp, tt),
            "passed": pp,
            "total": tt,
        }

    return {
        "adaptation_rate_overall": _rate(overall_pass, overall_total),
        "counts": {
            "passed": overall_pass,
            "total": overall_total,
            "failed_n": len(failed_ids),
            "skipped_n": len(skipped_ids),
        },
        "by_category": by_category,
        "by_suite": by_suite,
        "failed_case_ids": failed_ids,
        "skipped_case_ids": skipped_ids,
        "suites_included": sorted(suites_for_rate),
    }
