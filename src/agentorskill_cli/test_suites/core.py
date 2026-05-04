"""Legacy small core cases (kept for compatibility) + re-export full core list."""

from __future__ import annotations

from agentorskill_cli.test_suites.harness import TestCase
from agentorskill_cli.test_suites.ops_loss_opt import LOSS_OPT_CASES
from agentorskill_cli.test_suites.ops_math import MATH_CASES
from agentorskill_cli.test_suites.ops_nn_layers import NN_LAYERS_CASES
from agentorskill_cli.test_suites.ops_tensor_shape import TENSOR_SHAPE_CASES

LEGACY_CORE_CASES: list[TestCase] = [
    TestCase(
        id="core_conv2d",
        name="nn.Conv2d forward",
        body="""
    import torch
    import torch.nn as nn
    m = nn.Conv2d(3, 16, 3, padding=1)
    x = torch.randn(2, 3, 32, 32)
    y = m(x)
    return {"ok": True, "out_shape": list(y.shape)}
""",
        timeout=180.0,
        category="nn",
    ),
    TestCase(
        id="core_layernorm",
        name="LayerNorm",
        body="""
    import torch
    import torch.nn as nn
    m = nn.LayerNorm(64)
    x = torch.randn(2, 10, 64)
    y = m(x)
    return {"ok": True, "out_shape": list(y.shape)}
""",
        category="nn",
    ),
    TestCase(
        id="core_dropout_train",
        name="Dropout train mode",
        body="""
    import torch
    import torch.nn as nn
    m = nn.Dropout(0.5)
    m.train()
    x = torch.ones(2, 10)
    y = m(x)
    return {"ok": True, "shape": list(y.shape)}
""",
        category="nn",
    ),
    TestCase(
        id="core_indexing",
        name="Advanced indexing",
        body="""
    import torch
    x = torch.arange(12).reshape(3, 4)
    y = x[[0, 2], :]
    return {"ok": True, "shape": list(y.shape)}
""",
        category="tensor",
    ),
    TestCase(
        id="core_autograd",
        name="Autograd matmul grad",
        body="""
    import torch
    a = torch.randn(4, 4, requires_grad=True)
    b = torch.randn(4, 4, requires_grad=True)
    loss = (a @ b).sum()
    loss.backward()
    return {"ok": True, "ga_sum": float(a.grad.abs().sum())}
""",
        category="autograd",
    ),
]

CORE_CASES: list[TestCase] = (
    TENSOR_SHAPE_CASES
    + MATH_CASES
    + NN_LAYERS_CASES
    + LOSS_OPT_CASES
    + LEGACY_CORE_CASES
)
