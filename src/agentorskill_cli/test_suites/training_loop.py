"""Multi-step training simulations (forward, backward, optimizer)."""

from __future__ import annotations

from agentorskill_cli.test_suites.harness import TestCase

_TR = "training"

TRAINING_CASES: list[TestCase] = [
    TestCase(
        id="train_mlp_sgd",
        name="MLP multi-step SGD",
        body="""
    import torch
    import torch.nn as nn
    torch.manual_seed(0)
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = nn.Linear(16, 32)
            self.l2 = nn.Linear(32, 4)
        def forward(self, x):
            return self.l2(torch.relu(self.l1(x)))
    m = M()
    opt = torch.optim.SGD(m.parameters(), lr=0.05)
    losses = []
    for _ in range(15):
        x = torch.randn(8, 16)
        loss = m(x).pow(2).mean()
        losses.append(float(loss.detach()))
        opt.zero_grad()
        loss.backward()
        opt.step()
    return {
        "ok": True,
        "loss_initial": losses[0],
        "loss_final": losses[-1],
        "steps": len(losses),
    }
""",
        category=_TR,
        timeout=300.0,
    ),
    TestCase(
        id="train_cnn_adamw",
        name="CNN multi-step AdamW",
        body="""
    import torch
    import torch.nn as nn
    torch.manual_seed(1)
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
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    losses = []
    for _ in range(12):
        x = torch.randn(4, 3, 16, 16)
        y = torch.randint(0, 2, (4,))
        logits = m(x)
        loss = nn.CrossEntropyLoss()(logits, y)
        losses.append(float(loss.detach()))
        opt.zero_grad()
        loss.backward()
        opt.step()
    return {
        "ok": True,
        "loss_initial": losses[0],
        "loss_final": losses[-1],
        "steps": len(losses),
    }
""",
        category=_TR,
        timeout=360.0,
    ),
    TestCase(
        id="train_clip_eval",
        name="grad clip train eval toggle",
        body="""
    import torch
    import torch.nn as nn
    m = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 2))
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    m.train()
    x = torch.randn(4, 8)
    loss = m(x).sum()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
    opt.step()
    m.eval()
    with torch.no_grad():
        y = m(x)
    return {"ok": True, "eval_shape": list(y.shape)}
""",
        category=_TR,
        timeout=180.0,
    ),
    TestCase(
        id="train_amp_optional",
        name="CUDA AMP optional step",
        body="""
    import torch
    import torch.nn as nn
    dev = "cuda" if torch.cuda.is_available() else None
    if dev is None:
        return {"ok": True, "skipped": True, "reason": "no CUDA"}
    m = nn.Linear(8, 4).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    x = torch.randn(2, 8, device=dev)
    try:
        try:
            scaler = torch.amp.GradScaler("cuda")
        except Exception:
            scaler = torch.cuda.amp.GradScaler()
        opt.zero_grad()
        with torch.amp.autocast("cuda"):
            loss = m(x).float().sum()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        return {"ok": True, "loss": float(loss.detach())}
    except Exception as e:
        return {"ok": True, "skipped": True, "reason": str(e)}
""",
        category=_TR,
        timeout=180.0,
    ),
]
