#!/usr/bin/env python3
"""TorchBridgeBench Claude Code plugin deterministic core.

This MVP intentionally uses only the Python standard library. Framework-specific
cases can still run because user code is executed in an isolated Python process;
the harness only requires the case to expose a JSON-normalizable RESULT or run().
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import datetime as _dt
import importlib.util
import json
import math
import os
import platform
import re
import statistics
import subprocess
import sys
import tempfile
import textwrap
import traceback
from pathlib import Path
from typing import Any


FAILURE_CLASSES = {
    "EnvironmentFailure",
    "DependencyMissing",
    "ImportOrderError",
    "OperatorNotFound",
    "TypeMismatch",
    "ShapeMismatch",
    "DeviceMismatch",
    "AutogradFailure",
    "NumericMismatch",
    "TrainingDivergence",
    "RuntimeCrash",
    "TranslationError",
    "Unknown",
}


@dataclasses.dataclass
class TestCase:
    id: str
    level: str
    track: str
    code: str
    expected_ops: list[str]
    training_config: dict[str, Any]
    ground_truth: dict[str, Any]
    difficulty: str | None
    failure_mode: str | None
    seed: int | None
    source_path: str


@dataclasses.dataclass
class AdapterSpec:
    bridge_id: str
    track: str
    preamble: str
    source_preamble: str
    env: dict[str, str]
    docs: str
    known_gaps: list[str]
    op_api_map: dict[str, str]
    atol: float
    rtol: float
    timeout_seconds: int
    fne_threshold: float | None
    gc_threshold: float | None
    tca_threshold: float | None
    dtw_threshold: float | None
    source_path: str


@dataclasses.dataclass
class ExecResult:
    role: str
    ok: bool
    result: Any | None
    activations: Any | None
    gradients: Any | None
    task_metrics: Any | None
    returncode: int
    duration_seconds: float
    stdout: str
    stderr: str
    traceback: str | None


def _load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise SystemExit(f"Expected JSON object in {path}")
    return data


def load_case(path: Path) -> TestCase:
    data = _load_json(path)
    required = ["id", "level", "track", "code"]
    missing = [k for k in required if k not in data]
    if missing:
        raise SystemExit(f"TestCase {path} missing required field(s): {', '.join(missing)}")
    return TestCase(
        id=str(data["id"]),
        level=str(data["level"]),
        track=str(data["track"]),
        code=str(data["code"]),
        expected_ops=list(data.get("expected_ops") or []),
        training_config=dict(data.get("training_config") or {}),
        ground_truth=dict(data.get("ground_truth") or {}),
        difficulty=data.get("difficulty"),
        failure_mode=data.get("failure_mode"),
        seed=data.get("seed"),
        source_path=str(path),
    )


def load_adapter(path: Path) -> AdapterSpec:
    data = _load_json(path)
    required = ["bridge_id", "track"]
    missing = [k for k in required if k not in data]
    if missing:
        raise SystemExit(f"AdapterSpec {path} missing required field(s): {', '.join(missing)}")
    timeout = int(data.get("timeout_seconds") or 60)
    if timeout <= 0:
        raise SystemExit(f"AdapterSpec {path} timeout_seconds must be positive")
    env = data.get("env") or {}
    if not isinstance(env, dict):
        raise SystemExit(f"AdapterSpec {path} env must be an object")
    return AdapterSpec(
        bridge_id=str(data["bridge_id"]),
        track=str(data["track"]),
        preamble=str(data.get("preamble") or ""),
        source_preamble=str(data.get("source_preamble") or ""),
        env={str(k): str(v) for k, v in env.items()},
        docs=str(data.get("docs") or ""),
        known_gaps=list(data.get("known_gaps") or []),
        op_api_map=dict(data.get("op_api_map") or {}),
        atol=float(data.get("atol", 1e-5)),
        rtol=float(data.get("rtol", 1e-5)),
        timeout_seconds=timeout,
        fne_threshold=_optional_float(data.get("fne_threshold")),
        gc_threshold=_optional_float(data.get("gc_threshold")),
        tca_threshold=_optional_float(data.get("tca_threshold")),
        dtw_threshold=_optional_float(data.get("dtw_threshold")),
        source_path=str(path),
    )


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _runner_source(preamble: str, code: str) -> str:
    preamble_block = textwrap.indent(preamble, "    ") if preamble.strip() else "    pass"
    code_block = textwrap.indent(code, "    ")
    return f'''import json
import math
import sys
import traceback

def _normalize(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        if isinstance(value, float):
            if math.isnan(value):
                return {{"__float__": "nan"}}
            if math.isinf(value):
                return {{"__float__": "inf" if value > 0 else "-inf"}}
        return value
    if isinstance(value, dict):
        return {{str(k): _normalize(v) for k, v in value.items()}}
    if isinstance(value, (list, tuple)):
        return [_normalize(v) for v in value]
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        return _normalize(value.tolist())
    if hasattr(value, "item"):
        try:
            return _normalize(value.item())
        except Exception:
            pass
    return repr(value)

try:
{preamble_block}

{code_block}

    if "RESULT" in globals():
        result = globals()["RESULT"]
    elif "run" in globals() and callable(globals()["run"]):
        result = globals()["run"]()
    else:
        raise RuntimeError("Test code must define RESULT or run().")
    payload = {{"result": _normalize(result)}}
    if "ACTIVATIONS" in globals():
        payload["activations"] = _normalize(globals()["ACTIVATIONS"])
    if "GRADIENTS" in globals():
        payload["gradients"] = _normalize(globals()["GRADIENTS"])
    if "TASK_METRICS" in globals():
        payload["task_metrics"] = _normalize(globals()["TASK_METRICS"])
    print("TBBCC_PAYLOAD_JSON=" + json.dumps(payload, sort_keys=True))
except Exception:
    print("TBBCC_TRACEBACK_START", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    print("TBBCC_TRACEBACK_END", file=sys.stderr)
    sys.exit(1)
'''


def execute_python(role: str, preamble: str, code: str, env: dict[str, str], timeout: int) -> ExecResult:
    source = _runner_source(preamble, code)
    start = _dt.datetime.now(_dt.UTC)
    with tempfile.TemporaryDirectory(prefix="tbbcc_") as td:
        script = Path(td) / f"{role}.py"
        script.write_text(source, encoding="utf-8")
        merged_env = os.environ.copy()
        merged_env.update(env)
        try:
            proc = subprocess.run(
                [sys.executable, str(script)],
                text=True,
                capture_output=True,
                env=merged_env,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            end = _dt.datetime.now(_dt.UTC)
            return ExecResult(
                role=role,
                ok=False,
                result=None,
                activations=None,
                gradients=None,
                task_metrics=None,
                returncode=124,
                duration_seconds=(end - start).total_seconds(),
                stdout=_coerce_text(exc.stdout),
                stderr=_coerce_text(exc.stderr) or f"Timeout after {timeout}s",
                traceback=f"Timeout after {timeout}s",
            )
    end = _dt.datetime.now(_dt.UTC)
    payload = _extract_payload(proc.stdout)
    tb = _extract_traceback(proc.stderr)
    return ExecResult(
        role=role,
        ok=proc.returncode == 0 and payload is not _MISSING,
        result=None if payload is _MISSING else payload.get("result"),
        activations=None if payload is _MISSING else payload.get("activations"),
        gradients=None if payload is _MISSING else payload.get("gradients"),
        task_metrics=None if payload is _MISSING else payload.get("task_metrics"),
        returncode=proc.returncode,
        duration_seconds=(end - start).total_seconds(),
        stdout=proc.stdout,
        stderr=proc.stderr,
        traceback=tb,
    )


def _pair_runner_source(source_preamble: str, target_preamble: str, code: str, target_env: dict[str, str], timeout: int) -> str:
    return f"""import contextlib
import io
import json
import math
import os
import signal
import sys
import time
import traceback

SOURCE_PREAMBLE = {source_preamble!r}
TARGET_PREAMBLE = {target_preamble!r}
CODE = {code!r}
TARGET_ENV = {target_env!r}
ROLE_TIMEOUT = {timeout!r}

def _normalize(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        if isinstance(value, float):
            if math.isnan(value):
                return {{"__float__": "nan"}}
            if math.isinf(value):
                return {{"__float__": "inf" if value > 0 else "-inf"}}
        return value
    if isinstance(value, dict):
        return {{str(k): _normalize(v) for k, v in value.items()}}
    if isinstance(value, (list, tuple)):
        return [_normalize(v) for v in value]
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        return _normalize(value.tolist())
    if hasattr(value, "item"):
        try:
            return _normalize(value.item())
        except Exception:
            pass
    return repr(value)

class _RoleTimeout(Exception):
    pass

def _timeout_handler(signum, frame):
    raise _RoleTimeout(f"Timeout after {{ROLE_TIMEOUT}}s")

def _run_role(role, preamble, env_updates):
    started = time.perf_counter()
    ns = {{"__name__": "__main__"}}
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    previous_env = {{}}
    added_keys = []
    try:
        for key, value in env_updates.items():
            if key in os.environ:
                previous_env[key] = os.environ[key]
            else:
                added_keys.append(key)
            os.environ[key] = value
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(int(ROLE_TIMEOUT))
        try:
            with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
                if preamble:
                    exec(preamble, ns)
                exec(CODE, ns)
                if "RESULT" in ns:
                    result = ns["RESULT"]
                elif "run" in ns and callable(ns["run"]):
                    result = ns["run"]()
                else:
                    raise RuntimeError("Test code must define RESULT or run().")
            payload = {{"result": _normalize(result)}}
            if "ACTIVATIONS" in ns:
                payload["activations"] = _normalize(ns["ACTIVATIONS"])
            if "GRADIENTS" in ns:
                payload["gradients"] = _normalize(ns["GRADIENTS"])
            if "TASK_METRICS" in ns:
                payload["task_metrics"] = _normalize(ns["TASK_METRICS"])
            ok = True
            returncode = 0
            tb = None
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
    except Exception:
        payload = {{}}
        ok = False
        returncode = 1
        tb = traceback.format_exc().strip()
        stderr_buf.write(tb + "\\n")
    finally:
        for key, value in previous_env.items():
            os.environ[key] = value
        for key in added_keys:
            os.environ.pop(key, None)
    return {{
        "role": role,
        "ok": ok,
        "result": payload.get("result"),
        "activations": payload.get("activations"),
        "gradients": payload.get("gradients"),
        "task_metrics": payload.get("task_metrics"),
        "returncode": returncode,
        "duration_seconds": time.perf_counter() - started,
        "stdout": stdout_buf.getvalue(),
        "stderr": stderr_buf.getvalue(),
        "traceback": tb,
    }}

pair = {{
    "source": _run_role("source", SOURCE_PREAMBLE, {{}}),
    "target": _run_role("target", TARGET_PREAMBLE, TARGET_ENV),
}}
print("TBBCC_PAIR_JSON=" + json.dumps(pair, sort_keys=True))
"""


def _extract_pair_payload(stdout: str) -> Any:
    prefix = "TBBCC_PAIR_JSON="
    for line in reversed(stdout.splitlines()):
        if line.startswith(prefix):
            return json.loads(line[len(prefix) :])
    return _MISSING


def execute_python_pair(source_preamble: str, target_preamble: str, code: str, env: dict[str, str], timeout: int) -> tuple[ExecResult, ExecResult]:
    pair_source = _pair_runner_source(source_preamble, target_preamble, code, env, timeout)
    start = _dt.datetime.now(_dt.UTC)
    with tempfile.TemporaryDirectory(prefix="tbbcc_pair_") as td:
        script = Path(td) / "pair.py"
        script.write_text(pair_source, encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, str(script)],
                text=True,
                capture_output=True,
                env=os.environ.copy(),
                timeout=max(timeout * 2 + 5, timeout + 5),
            )
        except subprocess.TimeoutExpired as exc:
            end = _dt.datetime.now(_dt.UTC)
            duration = (end - start).total_seconds()
            timeout_text = f"Timeout after {timeout}s"
            src = ExecResult("source", False, None, None, None, None, 124, duration, _coerce_text(exc.stdout), _coerce_text(exc.stderr) or timeout_text, timeout_text)
            tgt = ExecResult("target", False, None, None, None, None, 124, duration, _coerce_text(exc.stdout), _coerce_text(exc.stderr) or timeout_text, timeout_text)
            return src, tgt
    payload = _extract_pair_payload(proc.stdout)
    if payload is _MISSING:
        fallback = ExecResult("source", False, None, None, None, None, proc.returncode, 0.0, proc.stdout, proc.stderr, _extract_traceback(proc.stderr))
        fallback_target = dataclasses.replace(fallback, role="target")
        return fallback, fallback_target

    def _convert(role: str) -> ExecResult:
        item = payload.get(role) or {}
        return ExecResult(
            role=role,
            ok=bool(item.get("ok")),
            result=item.get("result"),
            activations=item.get("activations"),
            gradients=item.get("gradients"),
            task_metrics=item.get("task_metrics"),
            returncode=int(item.get("returncode", 1)),
            duration_seconds=float(item.get("duration_seconds", 0.0)),
            stdout=_coerce_text(item.get("stdout")),
            stderr=_coerce_text(item.get("stderr")),
            traceback=item.get("traceback"),
        )

    return _convert("source"), _convert("target")


class _Missing:
    pass


_MISSING = _Missing()


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _extract_payload(stdout: str) -> Any:
    prefix = "TBBCC_PAYLOAD_JSON="
    for line in reversed(stdout.splitlines()):
        if line.startswith(prefix):
            return json.loads(line[len(prefix) :])
    legacy_prefix = "TBBCC_RESULT_JSON="
    for line in reversed(stdout.splitlines()):
        if line.startswith(legacy_prefix):
            return {"result": json.loads(line[len(legacy_prefix) :])}
    return _MISSING


def _extract_traceback(stderr: str) -> str | None:
    if "TBBCC_TRACEBACK_START" not in stderr:
        return stderr.strip() or None
    m = re.search(r"TBBCC_TRACEBACK_START\n(?P<body>.*?)\nTBBCC_TRACEBACK_END", stderr, re.S)
    if m:
        return m.group("body").strip()
    return stderr.strip() or None


def _flatten_numbers(value: Any, prefix: str = "$") -> tuple[list[float], list[str]]:
    numbers: list[float] = []
    paths: list[str] = []
    if isinstance(value, dict):
        if set(value.keys()) == {"__float__"}:
            marker = value["__float__"]
            if marker == "nan":
                numbers.append(float("nan"))
            elif marker == "inf":
                numbers.append(float("inf"))
            elif marker == "-inf":
                numbers.append(float("-inf"))
            paths.append(prefix)
            return numbers, paths
        for key in sorted(value):
            child_numbers, child_paths = _flatten_numbers(value[key], f"{prefix}.{key}")
            numbers.extend(child_numbers)
            paths.extend(child_paths)
    elif isinstance(value, list):
        for idx, item in enumerate(value):
            child_numbers, child_paths = _flatten_numbers(item, f"{prefix}[{idx}]")
            numbers.extend(child_numbers)
            paths.extend(child_paths)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        numbers.append(float(value))
        paths.append(prefix)
    return numbers, paths


def compare_values(src: Any, tgt: Any, atol: float, rtol: float) -> dict[str, Any]:
    if _shape_signature(src) != _shape_signature(tgt):
        return {
            "passed": False,
            "reason": "ShapeMismatch",
            "source_shape": _shape_signature(src),
            "target_shape": _shape_signature(tgt),
        }

    src_nums, paths = _flatten_numbers(src)
    tgt_nums, _ = _flatten_numbers(tgt)
    if not src_nums and not tgt_nums:
        return {
            "passed": src == tgt,
            "reason": None if src == tgt else "SemanticMismatch",
            "exact_match": src == tgt,
        }
    if len(src_nums) != len(tgt_nums):
        return {
            "passed": False,
            "reason": "ShapeMismatch",
            "source_count": len(src_nums),
            "target_count": len(tgt_nums),
        }

    diffs: list[float] = []
    failed_paths: list[str] = []
    dot = 0.0
    src_norm = 0.0
    tgt_norm = 0.0
    for idx, (a, b) in enumerate(zip(src_nums, tgt_nums)):
        if math.isnan(a) or math.isnan(b) or math.isinf(a) or math.isinf(b):
            close = a == b
            diff = 0.0 if close else float("inf")
        else:
            diff = abs(a - b)
            close = diff <= (atol + rtol * abs(b))
            dot += a * b
            src_norm += a * a
            tgt_norm += b * b
        diffs.append(diff)
        if not close:
            failed_paths.append(paths[idx])

    finite_diffs = [d for d in diffs if math.isfinite(d)]
    mae = statistics.fmean(finite_diffs) if finite_diffs else float("inf")
    max_error = max(diffs) if diffs else 0.0
    p95 = _percentile(finite_diffs, 95.0) if finite_diffs else float("inf")
    cosine = None
    if src_norm > 0 and tgt_norm > 0:
        cosine = dot / (math.sqrt(src_norm) * math.sqrt(tgt_norm))
    return {
        "passed": not failed_paths,
        "reason": None if not failed_paths else "NumericMismatch",
        "count": len(src_nums),
        "mae": mae,
        "max_error": max_error,
        "p95": p95,
        "cosine": cosine,
        "failed_paths": failed_paths[:20],
        "failed_count": len(failed_paths),
        "atol": atol,
        "rtol": rtol,
    }


def _shape_signature(value: Any) -> Any:
    if isinstance(value, dict):
        if set(value.keys()) == {"__float__"}:
            return "number"
        return {k: _shape_signature(value[k]) for k in sorted(value)}
    if isinstance(value, list):
        return ["list", len(value), _shape_signature(value[0]) if value else "empty"]
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return "number"
    return type(value).__name__


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = (len(ordered) - 1) * (pct / 100.0)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[int(pos)]
    weight = pos - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


def compare_optional_channel(source_value: Any, target_value: Any, atol: float, rtol: float) -> dict[str, Any]:
    if source_value is None and target_value is None:
        return {"implemented": False, "passed": None, "metrics": {}}
    if source_value is None or target_value is None:
        return {
            "implemented": True,
            "passed": False,
            "metrics": {
                "reason": "MissingChannel",
                "source_present": source_value is not None,
                "target_present": target_value is not None,
            },
        }
    metrics = compare_values(source_value, target_value, atol, rtol)
    return {"implemented": True, "passed": bool(metrics.get("passed")), "metrics": metrics}


def build_t2_result(source: ExecResult, target: ExecResult, atol: float, rtol: float) -> dict[str, Any]:
    activations = compare_optional_channel(source.activations, target.activations, atol, rtol)
    gradients = compare_optional_channel(source.gradients, target.gradients, atol, rtol)
    implemented = bool(activations["implemented"] or gradients["implemented"])
    if not implemented:
        return {
            "name": "ActivationGradient",
            "implemented": False,
            "passed": None,
            "metrics": {},
        }
    channel_results = {
        "activations": activations,
        "gradients": gradients,
    }
    passed = all(
        result["passed"] is not False
        for result in channel_results.values()
        if result["implemented"]
    )
    first_fail = None
    for name, result in channel_results.items():
        if result["implemented"] and result["passed"] is False:
            first_fail = name
            break
    return {
        "name": "ActivationGradient",
        "implemented": True,
        "passed": passed,
        "metrics": {
            "channels": channel_results,
            "first_fail": first_fail,
        },
    }


def build_t3_result(source: ExecResult, target: ExecResult, atol: float, rtol: float) -> dict[str, Any]:
    task = compare_optional_channel(source.task_metrics, target.task_metrics, atol, rtol)
    if not task["implemented"]:
        return {
            "name": "Task",
            "implemented": False,
            "passed": None,
            "metrics": {},
        }
    return {
        "name": "Task",
        "implemented": True,
        "passed": task["passed"],
        "metrics": task["metrics"],
    }


def auto_classify(
    source: ExecResult,
    target: ExecResult,
    comparison: dict[str, Any] | None,
    t2_result: dict[str, Any] | None = None,
    t3_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    evidence = []
    tb = "\n".join(x for x in [source.traceback, target.traceback, source.stderr, target.stderr] if x)
    lower = tb.lower()
    cls = "NoFailure"

    if not source.ok:
        cls = "EnvironmentFailure"
        evidence.append("baseline execution failed")
    elif not target.ok:
        if "no module named" in lower or "modulenotfounderror" in lower:
            cls = "DependencyMissing"
        elif "import" in lower and ("before" in lower or "order" in lower):
            cls = "ImportOrderError"
        elif "has no attribute" in lower or "operator not found" in lower or "unsupported" in lower:
            cls = "OperatorNotFound"
        elif "dtype" in lower or "type mismatch" in lower or "expected scalar type" in lower:
            cls = "TypeMismatch"
        elif "shape" in lower or "broadcast" in lower or "size mismatch" in lower:
            cls = "ShapeMismatch"
        elif "device" in lower or "cuda" in lower or "npu" in lower or "ascend" in lower:
            cls = "DeviceMismatch"
        elif "grad" in lower or "autograd" in lower or "backward" in lower:
            cls = "AutogradFailure"
        else:
            cls = "RuntimeCrash"
        evidence.append("target execution failed")
    elif comparison and not comparison.get("passed", False):
        reason = comparison.get("reason")
        if reason in FAILURE_CLASSES:
            cls = reason
        elif reason == "SemanticMismatch":
            cls = "NumericMismatch"
        else:
            cls = "NumericMismatch"
        evidence.append(f"comparison failed: {reason}")
    elif t2_result and t2_result.get("implemented") and not t2_result.get("passed"):
        first_fail = t2_result.get("metrics", {}).get("first_fail")
        cls = "AutogradFailure" if first_fail == "gradients" else "NumericMismatch"
        evidence.append(f"Tier-2 failed: {first_fail}")
    elif t3_result and t3_result.get("implemented") and not t3_result.get("passed"):
        cls = "TrainingDivergence"
        evidence.append("Tier-3 task metrics failed")
    else:
        cls = "NoFailure"
        evidence.append("no failure detected")

    return {
        "class": cls,
        "evidence": evidence,
        "counts_toward_compatibility": cls not in {"EnvironmentFailure", "DependencyMissing", "ImportOrderError"},
    }


def build_report(case: TestCase, adapter: AdapterSpec, source: ExecResult, target: ExecResult, comparison: dict[str, Any]) -> dict[str, Any]:
    atol = float(comparison.get("atol", adapter.atol))
    rtol = float(comparison.get("rtol", adapter.rtol))
    t2_result = build_t2_result(source, target, atol, rtol) if source.ok and target.ok else {
        "name": "ActivationGradient",
        "implemented": False,
        "passed": None,
        "metrics": {},
    }
    t3_result = build_t3_result(source, target, atol, rtol) if source.ok and target.ok else {
        "name": "Task",
        "implemented": False,
        "passed": None,
        "metrics": {},
    }
    classification = auto_classify(source, target, comparison, t2_result, t3_result)
    implemented_tiers_passed = bool(comparison.get("passed")) and (
        not t2_result.get("implemented") or bool(t2_result.get("passed"))
    ) and (
        not t3_result.get("implemented") or bool(t3_result.get("passed"))
    )
    final_state = "ALL_PASS" if source.ok and target.ok and implemented_tiers_passed else "MARK_UNFIXABLE"
    return {
        "schema_version": "tbbcc.report.v0.1",
        "created_at": _dt.datetime.now(_dt.UTC).isoformat(),
        "final_state": final_state,
        "case": dataclasses.asdict(case),
        "adapter": dataclasses.asdict(adapter),
        "environment": inspect_environment(),
        "timing": {
            "source_duration_seconds": source.duration_seconds,
            "target_duration_seconds": target.duration_seconds,
            "combined_execution_seconds": source.duration_seconds + target.duration_seconds,
        },
        "tiers": {
            "T1": {
                "name": "Tensor",
                "implemented": True,
                "passed": bool(source.ok and target.ok and comparison.get("passed")),
                "source": dataclasses.asdict(source),
                "target": dataclasses.asdict(target),
                "metrics": comparison,
                "auto_classification": classification,
            },
            "T2": {
                **t2_result,
            },
            "T3": {
                **t3_result,
            },
        },
        "effort": {
            "effort_adapt": None,
            "effort_repair": None,
            "effort_total": None,
            "repair_attempts": 0,
            "note": "Agent effort accounting is reserved for the Claude Code workflow layer.",
        },
    }


def inspect_environment(import_versions: bool = False) -> dict[str, Any]:
    modules = [
        "torch",
        "torch_npu",
        "mindspore",
        "torch4ms",
        "mindtorch",
        "mindtorch_v2",
        "mindnlp",
        "numpy",
    ]
    info: dict[str, Any] = {
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cwd": os.getcwd(),
        "version_imports_enabled": import_versions,
        "modules": {},
    }
    for name in modules:
        spec = importlib.util.find_spec(name)
        if spec is None:
            info["modules"][name] = {"available": False}
            continue
        module_info = {
            "available": True,
            "origin": spec.origin,
        }
        if import_versions:
            try:
                mod = __import__(name)
                module_info["version"] = getattr(mod, "__version__", "present")
            except Exception as exc:
                module_info["version"] = f"import_error: {exc}"
        info["modules"][name] = module_info
    return info


def write_report(report: dict[str, Any], out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "report.json"
    md_path = out_dir / "report.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, md_path


def write_suite_summary(summary: dict[str, Any], out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "summary.json"
    md_path = out_dir / "summary.md"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(render_suite_markdown(summary), encoding="utf-8")
    return json_path, md_path


def render_markdown(report: dict[str, Any]) -> str:
    t1 = report["tiers"]["T1"]
    t2 = report["tiers"]["T2"]
    t3 = report["tiers"]["T3"]
    metrics = t1["metrics"]
    cls = t1["auto_classification"]
    timing = report.get("timing") or {}
    lines = [
        "# TorchBridgeBench Report",
        "",
        f"- Case: `{report['case']['id']}`",
        f"- Adapter: `{report['adapter']['bridge_id']}`",
        f"- Final state: `{report['final_state']}`",
        f"- Tier-1 passed: `{t1['passed']}`",
        f"- Failure class: `{cls['class']}`",
        f"- Counts toward compatibility: `{cls['counts_toward_compatibility']}`",
        f"- Combined execution seconds: `{timing.get('combined_execution_seconds')}`",
        "",
        "## Tier-1 Metrics",
        "",
    ]
    for key in ["count", "mae", "p95", "max_error", "cosine", "atol", "rtol", "failed_count"]:
        if key in metrics:
            lines.append(f"- {key}: `{metrics[key]}`")
    if metrics.get("failed_paths"):
        lines.append(f"- failed_paths: `{metrics['failed_paths']}`")
    lines.extend(["", "## Tier-2 Metrics", ""])
    if not t2.get("implemented"):
        lines.append("- implemented: `False`")
    else:
        lines.append(f"- passed: `{t2.get('passed')}`")
        lines.append(f"- first_fail: `{t2.get('metrics', {}).get('first_fail')}`")
        for channel, result in t2.get("metrics", {}).get("channels", {}).items():
            if result.get("implemented"):
                channel_metrics = result.get("metrics", {})
                lines.append(
                    f"- {channel}: passed=`{result.get('passed')}`, "
                    f"mae=`{channel_metrics.get('mae')}`, failed_count=`{channel_metrics.get('failed_count')}`"
                )
    lines.extend(["", "## Tier-3 Metrics", ""])
    if not t3.get("implemented"):
        lines.append("- implemented: `False`")
    else:
        t3_metrics = t3.get("metrics", {})
        lines.append(f"- passed: `{t3.get('passed')}`")
        for key in ["mae", "p95", "max_error", "cosine", "failed_count"]:
            if key in t3_metrics:
                lines.append(f"- {key}: `{t3_metrics[key]}`")
        if t3_metrics.get("failed_paths"):
            lines.append(f"- failed_paths: `{t3_metrics['failed_paths']}`")
    lines.extend(["", "## Evidence", ""])
    for item in cls.get("evidence", []):
        lines.append(f"- {item}")
    if report["tiers"]["T1"]["target"].get("traceback"):
        lines.extend(["", "## Target Traceback", "", "```text", report["tiers"]["T1"]["target"]["traceback"], "```"])
    lines.extend(
        [
            "",
            "## Scope Notes",
            "",
            "- Tier-2 is implemented when a case exposes ACTIVATIONS or GRADIENTS.",
            "- Tier-3 is implemented when a case exposes TASK_METRICS.",
            "- Agent effort accounting is handled by the Claude Code workflow layer.",
            "",
        ]
    )
    return "\n".join(lines)


def render_suite_markdown(summary: dict[str, Any]) -> str:
    totals = summary["totals"]
    compatibility_rate = totals.get("compatibility_rate")
    raw_pass_rate = totals.get("raw_pass_rate")
    compatibility_text = "n/a" if compatibility_rate is None else f"{compatibility_rate:.4f}"
    raw_pass_text = "n/a" if raw_pass_rate is None else f"{raw_pass_rate:.4f}"
    lines = [
        "# TorchBridgeBench Suite Summary",
        "",
        f"- Suite: `{summary['suite_id']}`",
        f"- Total runs: `{totals['total']}`",
        f"- Passed: `{totals['passed']}`",
        f"- Failed: `{totals['failed']}`",
        f"- Compatibility-counted runs: `{totals.get('compatibility_counted_total')}`",
        f"- Compatibility-counted passes: `{totals.get('compatibility_counted_passed')}`",
        f"- Compatibility rate: `{compatibility_text}`",
        f"- Raw pass rate: `{raw_pass_text}`",
        f"- Total duration seconds: `{totals.get('duration_seconds')}`",
        "",
        "## Executive Summary",
        "",
    ]
    if compatibility_rate is None:
        lines.append("- No runs counted toward compatibility; review environment failures first.")
    else:
        lines.append(
            f"- Compatibility is measured after excluding environment/config-only failures; current rate is `{compatibility_text}`."
        )
    if summary["failure_classes"]:
        top_failure = sorted(summary["failure_classes"].items(), key=lambda item: (-item[1], item[0]))[0]
        lines.append(f"- Most frequent failure class: `{top_failure[0]}` x `{top_failure[1]}`.")
    else:
        lines.append("- No failure classes were recorded.")
    lines.extend(
        [
            "",
            "## Failure Classes",
            "",
        ]
    )
    if summary["failure_classes"]:
        for name, count in sorted(summary["failure_classes"].items()):
            lines.append(f"- {name}: `{count}`")
    else:
        lines.append("- none")
    lines.extend(["", "## Run Details", ""])
    for item in summary["runs"]:
        counted = item.get("counts_toward_compatibility")
        counted_text = "yes" if counted else "no"
        lines.append(
            f"- `{item['case_id']}` x `{item['bridge_id']}`: "
            f"`{item['final_state']}` / `{item['failure_class']}` / counted=`{counted_text}` / duration=`{item.get('duration_seconds')}` "
            f"([json]({item['report_json']}), [md]({item['report_md']}))"
        )
    lines.append("")
    return "\n".join(lines)


def _slug(value: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return clean or "item"


def _resolve_path(base: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    candidate = (base / path).resolve()
    if candidate.exists():
        return candidate
    return path.resolve()


def run_eval(case_path: Path, adapter_path: Path, out_dir: Path, timeout_override: int | None = None, atol_override: float | None = None, rtol_override: float | None = None) -> dict[str, Any]:
    started = _dt.datetime.now(_dt.UTC)
    case = load_case(case_path)
    adapter = load_adapter(adapter_path)
    timeout = int(timeout_override or adapter.timeout_seconds)
    atol = float(atol_override if atol_override is not None else case.ground_truth.get("atol", adapter.atol))
    rtol = float(rtol_override if rtol_override is not None else case.ground_truth.get("rtol", adapter.rtol))
    source, target = execute_python_pair(adapter.source_preamble, adapter.preamble, case.code, adapter.env, timeout)
    comparison: dict[str, Any]
    if source.ok and target.ok:
        comparison = compare_values(source.result, target.result, atol, rtol)
    else:
        comparison = {"passed": False, "reason": "ExecutionFailure", "atol": atol, "rtol": rtol}
    report = build_report(case, adapter, source, target, comparison)
    report["timing"]["wall_clock_seconds"] = (_dt.datetime.now(_dt.UTC) - started).total_seconds()
    json_path, md_path = write_report(report, out_dir)
    report["_paths"] = {"report_json": str(json_path), "report_md": str(md_path)}
    return report


def cmd_eval(args: argparse.Namespace) -> int:
    report = run_eval(
        Path(args.case),
        Path(args.adapter),
        Path(args.out),
        timeout_override=args.timeout,
        atol_override=args.atol,
        rtol_override=args.rtol,
    )
    paths = report.pop("_paths")
    print(json.dumps({"report_json": paths["report_json"], "report_md": paths["report_md"], "final_state": report["final_state"]}, indent=2))
    return 0 if report["final_state"] == "ALL_PASS" else 1


def cmd_eval_suite(args: argparse.Namespace) -> int:
    suite_started = _dt.datetime.now(_dt.UTC)
    suite_path = Path(args.suite).resolve()
    suite = _load_json(suite_path)
    suite_id = str(suite.get("suite_id") or suite_path.stem)
    cases = suite.get("cases") or []
    adapters = suite.get("adapters") or []
    if not isinstance(cases, list) or not cases:
        raise SystemExit(f"Suite {suite_path} must define a non-empty cases array")
    if not isinstance(adapters, list) or not adapters:
        raise SystemExit(f"Suite {suite_path} must define a non-empty adapters array")
    base = suite_path.parent
    out_dir = Path(args.out).resolve()
    runs: list[dict[str, Any]] = []
    failure_classes: dict[str, int] = {}
    passed = 0
    failed = 0
    compatibility_total = 0
    compatibility_passed = 0
    for case_item in cases:
        case_path = _resolve_path(base, str(case_item))
        case_obj = load_case(case_path)
        for adapter_item in adapters:
            adapter_path = _resolve_path(base, str(adapter_item))
            adapter_obj = load_adapter(adapter_path)
            run_dir = out_dir / "runs" / f"{_slug(case_obj.id)}__{_slug(adapter_obj.bridge_id)}"
            report = run_eval(
                case_path,
                adapter_path,
                run_dir,
                timeout_override=args.timeout,
                atol_override=args.atol,
                rtol_override=args.rtol,
            )
            paths = report.pop("_paths")
            cls = report["tiers"]["T1"]["auto_classification"]["class"]
            counts_toward_compatibility = report["tiers"]["T1"]["auto_classification"]["counts_toward_compatibility"]
            run_duration = report.get("timing", {}).get("wall_clock_seconds")
            if report["final_state"] == "ALL_PASS":
                passed += 1
                if counts_toward_compatibility:
                    compatibility_passed += 1
            else:
                failed += 1
                failure_classes[cls] = failure_classes.get(cls, 0) + 1
            if counts_toward_compatibility:
                compatibility_total += 1
            runs.append(
                {
                    "case_id": case_obj.id,
                    "case_path": str(case_path),
                    "bridge_id": adapter_obj.bridge_id,
                    "adapter_path": str(adapter_path),
                    "final_state": report["final_state"],
                    "failure_class": cls,
                    "report_json": paths["report_json"],
                    "report_md": paths["report_md"],
                    "counts_toward_compatibility": counts_toward_compatibility,
                    "duration_seconds": run_duration,
                }
            )
    total = passed + failed
    suite_duration = (_dt.datetime.now(_dt.UTC) - suite_started).total_seconds()
    summary = {
        "schema_version": "tbbcc.suite.v0.1",
        "suite_id": suite_id,
        "created_at": _dt.datetime.now(_dt.UTC).isoformat(),
        "suite_path": str(suite_path),
        "totals": {
            "total": total,
            "passed": passed,
            "failed": failed,
            "compatibility_counted_total": compatibility_total,
            "compatibility_counted_passed": compatibility_passed,
            "compatibility_rate": (compatibility_passed / compatibility_total) if compatibility_total else None,
            "raw_pass_rate": (passed / total) if total else 0.0,
            "duration_seconds": suite_duration,
        },
        "failure_classes": failure_classes,
        "runs": runs,
    }
    json_path, md_path = write_suite_summary(summary, out_dir)
    print(json.dumps({"summary_json": str(json_path), "summary_md": str(md_path), "totals": summary["totals"]}, indent=2))
    return 0 if failed == 0 else 1


def cmd_validate(args: argparse.Namespace) -> int:
    errors: list[str] = []
    for case_path in args.case or []:
        try:
            load_case(Path(case_path))
        except SystemExit as exc:
            errors.append(str(exc))
    for adapter_path in args.adapter or []:
        try:
            load_adapter(Path(adapter_path))
        except SystemExit as exc:
            errors.append(str(exc))
    if errors:
        print(json.dumps({"ok": False, "errors": errors}, indent=2))
        return 1
    print(json.dumps({"ok": True}, indent=2))
    return 0


def cmd_inspect_env(args: argparse.Namespace) -> int:
    print(json.dumps(inspect_environment(import_versions=bool(args.import_versions)), indent=2, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tbbcc", description="TorchBridgeBench Claude Code plugin core")
    sub = parser.add_subparsers(dest="command", required=True)

    p_eval = sub.add_parser("eval", help="Run a Tier-1 benchmark evaluation")
    p_eval.add_argument("--case", required=True, help="Path to TestCase JSON")
    p_eval.add_argument("--adapter", required=True, help="Path to AdapterSpec JSON")
    p_eval.add_argument("--out", required=True, help="Output directory")
    p_eval.add_argument("--timeout", type=int, default=None, help="Override timeout seconds")
    p_eval.add_argument("--atol", type=float, default=None, help="Override absolute tolerance")
    p_eval.add_argument("--rtol", type=float, default=None, help="Override relative tolerance")
    p_eval.set_defaults(func=cmd_eval)

    p_suite = sub.add_parser("eval-suite", help="Run a benchmark suite matrix")
    p_suite.add_argument("--suite", required=True, help="Path to suite JSON")
    p_suite.add_argument("--out", required=True, help="Output directory")
    p_suite.add_argument("--timeout", type=int, default=None, help="Override timeout seconds")
    p_suite.add_argument("--atol", type=float, default=None, help="Override absolute tolerance")
    p_suite.add_argument("--rtol", type=float, default=None, help="Override relative tolerance")
    p_suite.set_defaults(func=cmd_eval_suite)

    p_validate = sub.add_parser("validate-inputs", help="Validate case and adapter JSON files")
    p_validate.add_argument("--case", action="append", default=[])
    p_validate.add_argument("--adapter", action="append", default=[])
    p_validate.set_defaults(func=cmd_validate)

    p_env = sub.add_parser("inspect-env", help="Print local benchmark environment summary")
    p_env.add_argument(
        "--import-versions",
        action="store_true",
        help="Import modules to read __version__. Slower and may initialize heavy frameworks.",
    )
    p_env.set_defaults(func=cmd_inspect_env)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        return 130
    except Exception:
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
