"""Elementwise / reduction / linalg ops."""

from __future__ import annotations

from agentorskill_cli.test_suites.harness import TestCase

_M = "math"


def _case(i: str, title: str, body: str, timeout: float = 120.0) -> TestCase:
    return TestCase(id=i, name=title, body=body, category=_M, timeout=timeout)


MATH_CASES: list[TestCase] = [
    _case(
        "math_add_mul",
        "add mul div",
        """
    import torch
    a = torch.randn(3, 4)
    b = torch.randn(3, 4)
    y = (a + b) * 0.5 / (torch.ones_like(a) + 0.1)
    return {"ok": True, "mean": float(y.mean())}
""",
    ),
    _case(
        "math_pow_sqrt",
        "pow sqrt",
        """
    import torch
    x = torch.rand(2, 3) + 0.1
    y = torch.sqrt(torch.pow(x, 2))
    return {"ok": True, "shape": list(y.shape)}
""",
    ),
    _case(
        "math_exp_log",
        "exp log",
        """
    import torch
    x = torch.randn(2, 3)
    y = torch.log(torch.exp(x.clamp(-5, 5)))
    return {"ok": True, "shape": list(y.shape)}
""",
    ),
    _case(
        "math_sin_clamp",
        "sin where clamp",
        """
    import torch
    x = torch.randn(4, 4)
    m = x > 0
    y = torch.where(m, torch.sin(x), torch.clamp(x, -1, 1))
    return {"ok": True, "shape": list(y.shape)}
""",
    ),
    _case(
        "math_reductions",
        "mean sum max min",
        """
    import torch
    x = torch.randn(2, 5, 8)
    return {
        "ok": True,
        "mean": float(x.mean()),
        "sum": float(x.sum()),
        "mx": float(x.max()),
        "mn": float(x.min()),
    }
""",
    ),
    _case(
        "math_matmul_bmm",
        "matmul bmm",
        """
    import torch
    a = torch.randn(2, 4, 8)
    b = torch.randn(2, 8, 3)
    y = torch.bmm(a, b)
    z = a[0] @ b[0]
    return {"ok": True, "y_shape": list(y.shape), "z_shape": list(z.shape)}
""",
    ),
    _case(
        "math_einsum",
        "einsum batch matmul",
        """
    import torch
    a = torch.randn(3, 5, 4)
    b = torch.randn(3, 4, 6)
    y = torch.einsum("bij,bjk->bik", a, b)
    return {"ok": True, "shape": list(y.shape)}
""",
    ),
]
