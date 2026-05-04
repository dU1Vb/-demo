"""Loss functions and single optimizer steps."""

from __future__ import annotations

from agentorskill_cli.test_suites.harness import TestCase

_L = "loss_opt"


def _c(i: str, title: str, body: str) -> TestCase:
    return TestCase(id=i, name=title, body=body, category=_L, timeout=180.0)


LOSS_OPT_CASES: list[TestCase] = [
    _c(
        "lo_mse_ce",
        "MSELoss CrossEntropyLoss",
        """
    import torch
    import torch.nn as nn
    mse = nn.MSELoss()
    x = torch.randn(4, 5)
    y = torch.randn(4, 5)
    l1 = mse(x, y)
    logits = torch.randn(4, 10)
    tgt = torch.randint(0, 10, (4,))
    l2 = nn.CrossEntropyLoss()(logits, tgt)
    return {"ok": True, "l1": float(l1), "l2": float(l2)}
""",
    ),
    _c(
        "lo_bce_kl",
        "BCEWithLogits KLDivLoss",
        """
    import torch
    import torch.nn as nn
    x = torch.randn(4)
    t = torch.randint(0, 2, (4,)).float()
    l1 = nn.BCEWithLogitsLoss()(x, t)
    log_p = torch.randn(4, 8).log_softmax(dim=-1)
    p = torch.softmax(torch.randn(4, 8), dim=-1)
    l2 = nn.KLDivLoss(reduction="batchmean")(log_p, p)
    return {"ok": True, "l1": float(l1), "l2": float(l2)}
""",
    ),
    _c(
        "lo_nll_l1",
        "NLLLoss L1Loss",
        """
    import torch
    import torch.nn as nn
    log_sm = torch.randn(4, 7).log_softmax(dim=-1)
    tgt = torch.randint(0, 7, (4,))
    l1 = nn.NLLLoss()(log_sm, tgt)
    l2 = nn.L1Loss()(torch.randn(3, 4), torch.randn(3, 4))
    return {"ok": True, "l1": float(l1), "l2": float(l2)}
""",
    ),
    _c(
        "opt_sgd_step",
        "SGD single step",
        """
    import torch
    import torch.nn as nn
    m = nn.Linear(8, 4)
    opt = torch.optim.SGD(m.parameters(), lr=0.1)
    x = torch.randn(2, 8)
    loss = m(x).sum()
    loss.backward()
    opt.step()
    opt.zero_grad()
    return {"ok": True, "loss": float(loss.detach())}
""",
    ),
    _c(
        "opt_adamw_step",
        "AdamW single step",
        """
    import torch
    import torch.nn as nn
    m = nn.Linear(8, 4)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    x = torch.randn(2, 8)
    loss = m(x).pow(2).mean()
    loss.backward()
    opt.step()
    opt.zero_grad()
    return {"ok": True, "loss": float(loss.detach())}
""",
    ),
]
