"""Tensor shape / manipulation ops."""

from __future__ import annotations

from agentorskill_cli.test_suites.harness import TestCase

_T = "tensor"

TENSOR_SHAPE_CASES: list[TestCase] = [
    TestCase(
        id="ts_view",
        name="Tensor.view",
        body="""
    import torch
    x = torch.arange(24).reshape(2, 3, 4)
    y = x.view(-1)
    return {"ok": True, "shape": list(y.shape)}
""",
        category=_T,
    ),
    TestCase(
        id="ts_reshape",
        name="reshape",
        body="""
    import torch
    x = torch.randn(2, 12)
    y = x.reshape(4, 6)
    return {"ok": True, "shape": list(y.shape)}
""",
        category=_T,
    ),
    TestCase(
        id="ts_transpose",
        name="transpose",
        body="""
    import torch
    x = torch.randn(2, 3, 5)
    y = x.transpose(0, 1)
    return {"ok": True, "shape": list(y.shape)}
""",
        category=_T,
    ),
    TestCase(
        id="ts_permute",
        name="permute",
        body="""
    import torch
    x = torch.randn(2, 3, 4)
    y = x.permute(2, 0, 1)
    return {"ok": True, "shape": list(y.shape)}
""",
        category=_T,
    ),
    TestCase(
        id="ts_squeeze_unsqueeze",
        name="squeeze / unsqueeze",
        body="""
    import torch
    x = torch.randn(1, 3, 1, 5)
    y = x.squeeze().unsqueeze(-1)
    return {"ok": True, "shape": list(y.shape)}
""",
        category=_T,
    ),
    TestCase(
        id="ts_cat_stack",
        name="cat and stack",
        body="""
    import torch
    a = torch.randn(2, 4)
    b = torch.randn(2, 4)
    c = torch.cat([a, b], dim=1)
    s = torch.stack([a, b], dim=0)
    return {"ok": True, "c_shape": list(c.shape), "s_shape": list(s.shape)}
""",
        category=_T,
    ),
    TestCase(
        id="ts_split_chunk",
        name="split chunk",
        body="""
    import torch
    x = torch.randn(4, 8)
    parts = x.split(4, dim=1)
    ch = torch.chunk(x, 2, dim=0)
    return {"ok": True, "n_split": len(parts), "n_chunk": len(ch)}
""",
        category=_T,
    ),
    TestCase(
        id="ts_expand_repeat",
        name="expand / repeat",
        body="""
    import torch
    x = torch.randn(1, 3)
    y = x.expand(4, 3)
    z = x.repeat(4, 1)
    return {"ok": True, "y_shape": list(y.shape), "z_shape": list(z.shape)}
""",
        category=_T,
    ),
    TestCase(
        id="ts_narrow_clone",
        name="narrow clone detach",
        body="""
    import torch
    x = torch.randn(4, 8)
    y = x.narrow(1, 2, 3).clone().detach()
    return {"ok": True, "shape": list(y.shape)}
""",
        category=_T,
    ),
]
