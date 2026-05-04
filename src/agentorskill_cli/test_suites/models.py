"""Small model graphs: MLP, small CNN, transformer block."""

from __future__ import annotations

from agentorskill_cli.test_suites.harness import TestCase

_MD = "models"

MODEL_CASES: list[TestCase] = [
    TestCase(
        id="model_mlp",
        name="2-layer MLP",
        body="""
    import torch
    import torch.nn as nn
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = nn.Linear(32, 64)
            self.l2 = nn.Linear(64, 8)
        def forward(self, x):
            return self.l2(torch.relu(self.l1(x)))
    m = M()
    x = torch.randn(4, 32)
    y = m(x)
    return {"ok": True, "shape": list(y.shape)}
""",
        timeout=180.0,
        category=_MD,
    ),
    TestCase(
        id="model_cnn",
        name="Small CNN",
        body="""
    import torch
    import torch.nn as nn
    class C(nn.Module):
        def __init__(self):
            super().__init__()
            self.c = nn.Conv2d(3, 8, 3, padding=1)
            self.p = nn.AdaptiveAvgPool2d(1)
            self.f = nn.Linear(8, 2)
        def forward(self, x):
            x = torch.relu(self.c(x))
            x = self.p(x).flatten(1)
            return self.f(x)
    m = C()
    x = torch.randn(2, 3, 16, 16)
    y = m(x)
    return {"ok": True, "shape": list(y.shape)}
""",
        timeout=180.0,
        category=_MD,
    ),
    TestCase(
        id="model_transformer_block",
        name="Single transformer encoder layer",
        body="""
    import torch
    import torch.nn as nn
    enc = nn.TransformerEncoderLayer(
        d_model=64, nhead=4, dim_feedforward=128, batch_first=True
    )
    x = torch.randn(2, 8, 64)
    y = enc(x)
    return {"ok": True, "shape": list(y.shape)}
""",
        timeout=240.0,
        category=_MD,
    ),
]
