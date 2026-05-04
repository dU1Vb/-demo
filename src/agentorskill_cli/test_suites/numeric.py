"""Full-vector numeric fidelity vs CPU baseline (no adapter preamble on baseline)."""

from __future__ import annotations

from agentorskill_cli.test_suites.harness import TestCase

# Same graph, fixed seed; baseline vs adapter runs compare full flattened outputs + grad sample

NUMERIC_CASES: list[TestCase] = [
    TestCase(
        id="numeric_baseline_ref",
        name="Reference MLP on CPU (no adapter)",
        body="""
    import torch
    import torch.nn as nn
    torch.manual_seed(42)
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = nn.Linear(16, 32)
            self.l2 = nn.Linear(32, 8)
        def forward(self, x):
            return self.l2(torch.relu(self.l1(x)))
    m = M()
    x = torch.randn(4, 16)
    y = m(x)
    loss = y.pow(2).mean()
    loss.backward()
    y_flat = y.detach().float().cpu().reshape(-1).tolist()
    g_flat = m.l1.weight.grad.detach().float().cpu().reshape(-1).tolist()
    return {"ok": True, "y_flat": y_flat, "g_flat": g_flat, "n_y": len(y_flat), "n_g": len(g_flat)}
""",
        include_adapter=False,
        category="numeric_fidelity",
        timeout=240.0,
    ),
    TestCase(
        id="numeric_with_adapter",
        name="Same MLP with adapter preamble",
        body="""
    import torch
    import torch.nn as nn
    torch.manual_seed(42)
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = nn.Linear(16, 32)
            self.l2 = nn.Linear(32, 8)
        def forward(self, x):
            return self.l2(torch.relu(self.l1(x)))
    m = M()
    x = torch.randn(4, 16)
    y = m(x)
    loss = y.pow(2).mean()
    loss.backward()
    y_flat = y.detach().float().cpu().reshape(-1).tolist()
    g_flat = m.l1.weight.grad.detach().float().cpu().reshape(-1).tolist()
    return {"ok": True, "y_flat": y_flat, "g_flat": g_flat, "n_y": len(y_flat), "n_g": len(g_flat)}
""",
        include_adapter=True,
        category="numeric_fidelity",
        timeout=240.0,
    ),
]
