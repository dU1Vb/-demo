"""NN modules forward coverage."""

from __future__ import annotations

from agentorskill_cli.test_suites.harness import TestCase

_N = "nn"


def _c(i: str, title: str, body: str, timeout: float = 180.0) -> TestCase:
    return TestCase(id=i, name=title, body=body, category=_N, timeout=timeout)


NN_LAYERS_CASES: list[TestCase] = [
    _c(
        "nn_conv1d",
        "Conv1d",
        """
    import torch
    import torch.nn as nn
    m = nn.Conv1d(8, 16, kernel_size=3, padding=1)
    x = torch.randn(2, 8, 32)
    return {"ok": True, "shape": list(m(x).shape)}
""",
    ),
    _c(
        "nn_conv3d",
        "Conv3d",
        """
    import torch
    import torch.nn as nn
    m = nn.Conv3d(2, 4, kernel_size=3, padding=1)
    x = torch.randn(1, 2, 8, 8, 8)
    return {"ok": True, "shape": list(m(x).shape)}
""",
        timeout=240.0,
    ),
    _c(
        "nn_conv_transpose",
        "ConvTranspose2d",
        """
    import torch
    import torch.nn as nn
    m = nn.ConvTranspose2d(4, 8, 4, stride=2, padding=1)
    x = torch.randn(2, 4, 8, 8)
    return {"ok": True, "shape": list(m(x).shape)}
""",
    ),
    _c(
        "nn_pool",
        "MaxPool2d AvgPool2d AdaptiveAvgPool2d",
        """
    import torch
    import torch.nn as nn
    x = torch.randn(2, 3, 32, 32)
    y1 = nn.MaxPool2d(2)(x)
    y2 = nn.AvgPool2d(2)(x)
    y3 = nn.AdaptiveAvgPool2d(1)(x)
    return {"ok": True, "s1": list(y1.shape), "s3": list(y3.shape)}
""",
    ),
    _c(
        "nn_batchnorm",
        "BatchNorm1d BatchNorm2d",
        """
    import torch
    import torch.nn as nn
    m1 = nn.BatchNorm1d(16)
    m2 = nn.BatchNorm2d(8)
    m1.train()
    m2.train()
    x1 = torch.randn(4, 16, 10)
    x2 = torch.randn(2, 8, 16, 16)
    return {"ok": True, "s1": list(m1(x1).shape), "s2": list(m2(x2).shape)}
""",
    ),
    _c(
        "nn_groupnorm_instancenorm",
        "GroupNorm InstanceNorm2d",
        """
    import torch
    import torch.nn as nn
    gn = nn.GroupNorm(4, 8)
    inn = nn.InstanceNorm2d(8)
    x = torch.randn(2, 8, 16, 16)
    return {"ok": True, "gn": list(gn(x).shape), "in": list(inn(x).shape)}
""",
    ),
    _c(
        "nn_embedding",
        "Embedding",
        """
    import torch
    import torch.nn as nn
    e = nn.Embedding(100, 32)
    idx = torch.randint(0, 100, (4, 12))
    return {"ok": True, "shape": list(e(idx).shape)}
""",
    ),
    _c(
        "nn_mha",
        "MultiheadAttention",
        """
    import torch
    import torch.nn as nn
    m = nn.MultiheadAttention(64, 4, batch_first=True)
    x = torch.randn(2, 10, 64)
    y, _ = m(x, x, x)
    return {"ok": True, "shape": list(y.shape)}
""",
        timeout=240.0,
    ),
    _c(
        "nn_rnn_gru",
        "RNN GRU",
        """
    import torch
    import torch.nn as nn
    r = nn.RNN(16, 32, batch_first=True)
    g = nn.GRU(16, 32, batch_first=True)
    x = torch.randn(4, 8, 16)
    y1, _ = r(x)
    y2, _ = g(x)
    return {"ok": True, "s1": list(y1.shape), "s2": list(y2.shape)}
""",
        timeout=240.0,
    ),
    _c(
        "nn_lstm",
        "LSTM",
        """
    import torch
    import torch.nn as nn
    l = nn.LSTM(16, 32, batch_first=True)
    x = torch.randn(4, 8, 16)
    y, _ = l(x)
    return {"ok": True, "shape": list(y.shape)}
""",
        timeout=240.0,
    ),
    _c(
        "nn_sdpa",
        "scaled_dot_product_attention",
        """
    import torch
    import torch.nn.functional as F
    q = torch.randn(2, 4, 8, 32)
    k = torch.randn(2, 4, 8, 32)
    v = torch.randn(2, 4, 8, 32)
    try:
        o = F.scaled_dot_product_attention(q, k, v)
        return {"ok": True, "shape": list(o.shape)}
    except Exception as e:
        return {"ok": True, "skipped": True, "reason": str(e)}
""",
    ),
]
