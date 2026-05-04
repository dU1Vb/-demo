"""Tests for adapter schema validation."""

import json
from pathlib import Path

import pytest

from agentorskill_cli.adapter_schema import validate_adapter_dict
from agentorskill_cli.test_suites.runner import execute_plan


def test_validate_example_torchax() -> None:
    p = Path(__file__).resolve().parent.parent / "examples" / "torchax_adapter.json"
    data = json.loads(p.read_text(encoding="utf-8"))
    spec = validate_adapter_dict(data)
    assert spec.library_name == "torchax"


def test_noop_smoke_runs() -> None:
    pytest.importorskip("torch")
    p = Path(__file__).resolve().parent.parent / "examples" / "noop_adapter.json"
    spec = validate_adapter_dict(json.loads(p.read_text(encoding="utf-8")))
    summ = execute_plan(spec, "smoke")
    assert "smoke" in summ.suites
    sr = summ.suites["smoke"]
    assert sr.failed == 0, [c.error for c in sr.cases if not c.ok]
