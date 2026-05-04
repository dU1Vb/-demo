"""Lightweight performance micro-benchmarks (optional suite)."""

from __future__ import annotations

from agentorskill_cli.test_suites.harness import TestCase

BENCH_CASES: list[TestCase] = [
    TestCase(
        id="bench_matmul",
        name="Matmul throughput (rough)",
        body="""
    import time
    import torch
    a = torch.randn(512, 512)
    b = torch.randn(512, 512)
    for _ in range(3):
        _ = a @ b
    t0 = time.perf_counter()
    for _ in range(20):
        c = a @ b
    dt = time.perf_counter() - t0
    if hasattr(c, "cpu"):
        _ = c.sum().item()
    return {"ok": True, "total_s": dt, "iters": 20}
""",
        timeout=300.0,
        category="perf",
    ),
    TestCase(
        id="bench_conv",
        name="Conv2d micro",
        body="""
    import time
    import torch
    import torch.nn as nn
    m = nn.Conv2d(64, 64, 3, padding=1)
    x = torch.randn(4, 64, 64, 64)
    y = m(x)
    t0 = time.perf_counter()
    for _ in range(10):
        y = m(x)
    dt = time.perf_counter() - t0
    return {"ok": True, "out_shape": list(y.shape), "total_s": dt, "iters": 10}
""",
        timeout=300.0,
        category="perf",
    ),
]
