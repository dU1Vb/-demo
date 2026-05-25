"""Shared migration-effort calculations for agent runs and AR calibration."""

from __future__ import annotations

from agentorskill_cli.agent_schema import MigrationEffortStats


def calc_effort_total(
    *,
    rounds: int,
    prompt_chars: int,
    completion_chars: int,
    edit_units: int,
) -> float:
    """Heuristic ME total using one shared formula across calibration and runtime."""
    return float(rounds + (prompt_chars + completion_chars + edit_units) / 1000.0)


def calc_effort_from_stats(stats: MigrationEffortStats) -> float:
    edit_units = stats.patch_lines_added + stats.patch_lines_deleted
    rounds = stats.llm_calls + stats.revalidation_runs + stats.repair_attempts
    return calc_effort_total(
        rounds=rounds,
        prompt_chars=stats.prompt_chars,
        completion_chars=stats.completion_chars,
        edit_units=edit_units,
    )


def calc_ar(me: float | None, baseline_effort: float | None) -> float | None:
    if me is None or baseline_effort is None:
        return None
    if baseline_effort <= 0:
        return 1.0
    return max(0.0, 1.0 - (me / baseline_effort))
