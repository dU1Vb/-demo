"""Smoke tests: import, enable, simple tensor and forward."""

from __future__ import annotations

from agentorskill_cli.test_suites.harness import TestCase

SMOKE_CASES: list[TestCase] = [
    TestCase(
        id="smoke_import_torch",
        name="Import torch and create tensor",
        body="""
    import torch
    x = torch.ones(2, 3)
    return {"ok": True, "shape": list(x.shape), "dtype": str(x.dtype)}
""",
        category="smoke",
    ),
    TestCase(
        id="smoke_addmm",
        name="matmul path",
        body="""
    import torch
    a = torch.randn(4, 8)
    b = torch.randn(8, 2)
    y = a @ b
    return {"ok": True, "shape": list(y.shape)}
""",
        category="smoke",
    ),
]
