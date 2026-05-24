"""Smoke validation for the mirrored TorchBridgeBench deterministic core."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def test_tbbcc_validate_and_eval_smoke() -> None:
    gen = subprocess.run(
        [sys.executable, "scripts/generate_benchmark_library.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert '"total_cases": 175' in gen.stdout

    validate = subprocess.run(
        [
            sys.executable,
            "scripts/tbbcc.py",
            "validate-inputs",
            "--case",
            "examples/cases/pure_python_vector.json",
            "--adapter",
            "examples/adapters/noop.json",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(validate.stdout)
    assert payload["ok"] is True

    out_dir = ROOT / "reports_tbbcc_test"
    run = subprocess.run(
        [
            sys.executable,
            "scripts/tbbcc.py",
            "eval",
            "--case",
            "examples/cases/pure_python_vector.json",
            "--adapter",
            "examples/adapters/noop.json",
            "--out",
            str(out_dir),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    result = json.loads(run.stdout)
    assert result["final_state"] == "ALL_PASS"


def test_tbbcc_smoke_suite_manifest_exists() -> None:
    manifest = ROOT / "benchmarks/v1.0.0/manifest.json"
    if not manifest.exists():
        subprocess.run(
            [sys.executable, "scripts/generate_benchmark_library.py"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    data = json.loads(manifest.read_text(encoding="utf-8"))
    assert data["totals"]["total_cases"] == 175
    assert data["suites"]["dev"] == "suites/dev_noop.json"
    assert data["suites"]["smoke"] == "suites/smoke_noop.json"
