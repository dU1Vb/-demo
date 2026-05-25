"""AR calibration utilities for baseline translation effort."""

from __future__ import annotations

import json
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from agentorskill_cli.ar_identity import AR_AGENT_SYSTEM_VERSION, AR_EFFORT_FORMULA_VERSION
from agentorskill_cli.effort import calc_effort_total
from agentorskill_cli.llm_provider import LLMTextResponse, OpenAIChatProvider


@dataclass
class CalibrationSample:
    case_id: str
    reroll_index: int
    prompt_chars: int
    completion_chars: int
    translation_chars: int
    rounds: int
    total_effort: float


@dataclass
class CalibrationReroll:
    reroll_index: int
    case_count: int
    total_effort: float
    mean_case_effort: float


@dataclass
class CalibrationResult:
    baseline_effort: float
    mean_effort: float
    std_effort: float
    cv_effort: float | None
    samples: list[CalibrationSample]
    rerolls: list[CalibrationReroll]
    source_cases: list[str]
    model: str
    provider: str
    protocol: str
    task_set_digest: str
    calibration_prompt_digest: str
    agent_system_version: str
    effort_formula_version: str


SYSTEM_PROMPT = """You are translating PyTorch code to MindSpore code.
Return only the translated code, no markdown, no explanation.
Preserve behavior as closely as possible.
"""


def _effort_from_response(case_id: str, resp: LLMTextResponse, *, reroll_index: int) -> CalibrationSample:
    translation_chars = len(resp.text.strip())
    rounds = 1
    prompt_chars = resp.prompt_chars
    completion_chars = resp.completion_chars
    total_effort = calc_effort_total(
        rounds=rounds,
        prompt_chars=prompt_chars,
        completion_chars=completion_chars,
        edit_units=translation_chars,
    )
    return CalibrationSample(
        case_id=case_id,
        reroll_index=reroll_index,
        prompt_chars=prompt_chars,
        completion_chars=completion_chars,
        translation_chars=translation_chars,
        rounds=rounds,
        total_effort=total_effort,
    )


def measure_baseline_effort(
    *,
    provider: OpenAIChatProvider,
    task_cases: list[dict[str, str]],
    rerolls: int,
    max_cv: float,
    max_workers: int = 1,
) -> CalibrationResult:
    samples: list[CalibrationSample] = []
    reroll_summaries: list[CalibrationReroll] = []
    source_cases: list[str] = [case["case_id"] for case in task_cases]
    task_set_digest = hashlib.sha256(
        json.dumps(task_cases, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    calibration_prompt_digest = hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()

    def _translate_one(reroll_index: int, case: dict[str, str]) -> CalibrationSample:
        clone_fn = getattr(provider, "clone", None)
        local_provider = clone_fn() if callable(clone_fn) else provider
        resp = local_provider.complete_text(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=case["code"],
            temperature=0.2,
        )
        return _effort_from_response(case["case_id"], resp, reroll_index=reroll_index)

    effective_workers = max(1, max_workers or 1)
    if effective_workers == 1:
        for idx in range(1, rerolls + 1):
            reroll_efforts: list[float] = []
            for case in task_cases:
                sample = _translate_one(idx, case)
                samples.append(sample)
                reroll_efforts.append(sample.total_effort)
            total_effort = sum(reroll_efforts)
            reroll_summaries.append(
                CalibrationReroll(
                    reroll_index=idx,
                    case_count=len(reroll_efforts),
                    total_effort=total_effort,
                    mean_case_effort=(total_effort / len(reroll_efforts)) if reroll_efforts else 0.0,
                )
            )
    else:
        per_reroll: dict[int, list[CalibrationSample]] = {idx: [] for idx in range(1, rerolls + 1)}
        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            futures = [
                executor.submit(_translate_one, idx, case)
                for idx in range(1, rerolls + 1)
                for case in task_cases
            ]
            for future in as_completed(futures):
                sample = future.result()
                per_reroll[sample.reroll_index].append(sample)
        for idx in range(1, rerolls + 1):
            reroll_samples = sorted(per_reroll[idx], key=lambda item: item.case_id)
            samples.extend(reroll_samples)
            reroll_efforts = [sample.total_effort for sample in reroll_samples]
            total_effort = sum(reroll_efforts)
            reroll_summaries.append(
                CalibrationReroll(
                    reroll_index=idx,
                    case_count=len(reroll_efforts),
                    total_effort=total_effort,
                    mean_case_effort=(total_effort / len(reroll_efforts)) if reroll_efforts else 0.0,
                )
            )

    efforts = [r.total_effort for r in reroll_summaries]
    mean_effort = mean(efforts) if efforts else 0.0
    std_effort = pstdev(efforts) if len(efforts) > 1 else 0.0
    cv_effort = (std_effort / mean_effort) if mean_effort > 0 else None
    if cv_effort is not None and cv_effort > max_cv:
        raise RuntimeError(
            f"baseline effort variance too high: cv={cv_effort:.4f} > max_cv={max_cv:.4f}"
        )
    return CalibrationResult(
        baseline_effort=mean_effort,
        mean_effort=mean_effort,
        std_effort=std_effort,
        cv_effort=cv_effort,
        samples=samples,
        rerolls=reroll_summaries,
        source_cases=source_cases,
        model=provider.model,
        provider=provider.provider,
        protocol=provider.protocol,
        task_set_digest=task_set_digest,
        calibration_prompt_digest=calibration_prompt_digest,
        agent_system_version=AR_AGENT_SYSTEM_VERSION,
        effort_formula_version=AR_EFFORT_FORMULA_VERSION,
    )


def load_task_cases(task_source: Path) -> list[dict[str, str]]:
    if task_source.is_dir():
        cases: list[dict[str, str]] = []
        for path in sorted(task_source.rglob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and "id" in data and "code" in data:
                cases.append({"case_id": str(data["id"]), "code": str(data["code"])})
        return cases
    data = json.loads(task_source.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [{"case_id": str(item["case_id"]), "code": str(item["code"])} for item in data]
    if isinstance(data, dict) and "cases" in data and isinstance(data["cases"], list):
        return [{"case_id": str(item["case_id"]), "code": str(item["code"])} for item in data["cases"]]
    raise ValueError("task source must be a JSON list, JSON object with cases, or a directory of JSON cases")


def save_baseline_effort_constant(path: Path, result: CalibrationResult) -> None:
    payload: dict[str, Any] = {
        "baseline_effort": result.baseline_effort,
        "mean_effort": result.mean_effort,
        "std_effort": result.std_effort,
        "cv_effort": result.cv_effort,
        "sample_count": len(result.samples),
        "reroll_count": len(result.rerolls),
        "source_cases": result.source_cases,
        "model": result.model,
        "provider": result.provider,
        "protocol": result.protocol,
        "task_set_digest": result.task_set_digest,
        "calibration_prompt_digest": result.calibration_prompt_digest,
        "agent_system_version": result.agent_system_version,
        "effort_formula_version": result.effort_formula_version,
        "rerolls": [
            {
                "reroll_index": r.reroll_index,
                "case_count": r.case_count,
                "total_effort": r.total_effort,
                "mean_case_effort": r.mean_case_effort,
            }
            for r in result.rerolls
        ],
        "samples": [
            {
                "case_id": s.case_id,
                "reroll_index": s.reroll_index,
                "prompt_chars": s.prompt_chars,
                "completion_chars": s.completion_chars,
                "translation_chars": s.translation_chars,
                "rounds": s.rounds,
                "total_effort": s.total_effort,
            }
            for s in result.samples
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
