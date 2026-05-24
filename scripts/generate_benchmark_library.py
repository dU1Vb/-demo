#!/usr/bin/env python3
"""Build the static TorchBridgeBench benchmark library from Appendix A.

The runtime contract remains static JSON. This script is only a build-time tool
that materializes the benchmark tree under benchmarks/v1.0.0/.
"""

from __future__ import annotations

import json
import shutil
import textwrap
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BENCH_ROOT = ROOT / "benchmarks" / "v1.0.0"
CASES_ROOT = BENCH_ROOT / "cases"
SUITES_ROOT = BENCH_ROOT / "suites"
MANIFEST_PATH = BENCH_ROOT / "manifest.json"
ADAPTER_REF = "../../../examples/adapters/noop.json"


COMMON = textwrap.dedent(
    """
    from contextlib import nullcontext
    import math
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def select_device():
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def precision_dtype(label, device):
        if "FP16" in label and device.type != "cpu":
            return torch.float16
        return torch.float32

    def maybe_autocast(device, precision):
        if "AMP" in precision and device.type != "cpu":
            return torch.autocast(device_type=device.type, dtype=torch.float16)
        return nullcontext()

    def grad_norm(value):
        if value is None:
            return 0.0
        return float(value.detach().float().norm().cpu())

    def tensor_mean(value):
        return float(value.detach().float().mean().cpu())

    def tensor_sample(value, limit=8):
        flat = value.detach().float().reshape(-1).cpu()
        return [float(x) for x in flat[:limit]]

    def first_grad(module):
        for param in module.parameters():
            if param.grad is not None:
                return param.grad
        return None

    def freeze_batch_norm(module):
        for child in module.modules():
            if isinstance(child, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                child.eval()
                if child.weight is not None:
                    child.weight.requires_grad_(False)
                if child.bias is not None:
                    child.bias.requires_grad_(False)

    def nms_torch(boxes, scores, iou_threshold=0.5):
        if boxes.numel() == 0:
            return torch.empty((0,), dtype=torch.long, device=boxes.device)
        order = torch.argsort(scores, descending=True)
        keep = []
        while order.numel() > 0:
            idx = order[0]
            keep.append(idx)
            if order.numel() == 1:
                break
            rest = order[1:]
            x1 = torch.maximum(boxes[idx, 0], boxes[rest, 0])
            y1 = torch.maximum(boxes[idx, 1], boxes[rest, 1])
            x2 = torch.minimum(boxes[idx, 2], boxes[rest, 2])
            y2 = torch.minimum(boxes[idx, 3], boxes[rest, 3])
            inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
            area_idx = (boxes[idx, 2] - boxes[idx, 0]).clamp(min=0) * (boxes[idx, 3] - boxes[idx, 1]).clamp(min=0)
            area_rest = (boxes[rest, 2] - boxes[rest, 0]).clamp(min=0) * (boxes[rest, 3] - boxes[rest, 1]).clamp(min=0)
            union = area_idx + area_rest - inter + 1e-6
            iou = inter / union
            order = rest[iou <= iou_threshold]
        return torch.stack(keep) if keep else torch.empty((0,), dtype=torch.long, device=boxes.device)
    """
).strip()


@dataclass
class CaseSpec:
    id: str
    level: str
    track: str
    difficulty: str
    code: str
    expected_ops: list[str]
    training_config: dict[str, Any] | None = None
    ground_truth: dict[str, Any] | None = None
    seed: int = 42
    failure_mode: str | None = None

    def to_json(self) -> dict[str, Any]:
        payload = {
            "id": self.id,
            "level": self.level,
            "track": self.track,
            "difficulty": self.difficulty,
            "code": self.code,
            "expected_ops": self.expected_ops,
            "training_config": self.training_config,
            "ground_truth": self.ground_truth,
            "seed": self.seed,
        }
        if self.failure_mode:
            payload["failure_mode"] = self.failure_mode
        return payload


def indent(block: str, prefix: str = "    ") -> str:
    return textwrap.indent(textwrap.dedent(block).strip(), prefix)


def wrap_code(body: str) -> str:
    return f"{COMMON}\n\n{textwrap.dedent(body).strip()}\n"


def default_ground_truth(level: str, rationale: str, *, training: bool = False) -> dict[str, Any]:
    base: dict[str, Any] = {
        "atol": 1e-5,
        "rtol": 1e-5,
        "fne_threshold": 0.995,
        "gc_threshold": 0.95,
        "tca_threshold": 0.02,
        "annotator": "TBD_after_pilot",
        "rationale": rationale,
    }
    if training:
        base["dtw_threshold"] = 0.05
    return base


def make_case(
    identifier: str,
    level: str,
    difficulty: str,
    code: str,
    expected_ops: list[str],
    *,
    track: str = "both",
    training_config: dict[str, Any] | None = None,
    rationale: str,
    failure_mode: str | None = None,
) -> CaseSpec:
    return CaseSpec(
        id=identifier,
        level=level,
        track=track,
        difficulty=difficulty,
        code=wrap_code(code),
        expected_ops=expected_ops,
        training_config=training_config,
        ground_truth=default_ground_truth(level, rationale, training=bool(training_config)),
        failure_mode=failure_mode,
    )


def simple_case_code(setup: str, result_expr: str, *, precision: str = "FP32") -> str:
    return f"""
device = select_device()
precision = "{precision}"
dtype = precision_dtype(precision, device)
{textwrap.dedent(setup).strip()}
RESULT = {result_expr}
"""


def l1_case(
    identifier: str,
    setup: str,
    result_expr: str,
    expected_ops: list[str],
    *,
    precision: str = "FP32",
    difficulty: str = "basic",
    rationale: str,
    failure_mode: str | None = None,
) -> CaseSpec:
    return make_case(
        identifier,
        "L1",
        difficulty,
        simple_case_code(setup, result_expr, precision=precision),
        expected_ops,
        rationale=rationale,
        failure_mode=failure_mode,
    )


def l2_case(
    identifier: str,
    setup: str,
    run_body: str,
    expected_ops: list[str],
    *,
    difficulty: str = "intermediate",
    rationale: str,
    failure_mode: str | None = None,
) -> CaseSpec:
    code = f"""
{textwrap.dedent(setup).strip()}

def run():
{indent(run_body)}
"""
    return make_case(
        identifier,
        "L2",
        difficulty,
        code,
        expected_ops,
        rationale=rationale,
        failure_mode=failure_mode,
    )


def l3_case(
    identifier: str,
    setup: str,
    run_body: str,
    expected_ops: list[str],
    *,
    rationale: str,
    failure_mode: str | None = None,
) -> CaseSpec:
    code = f"""
{textwrap.dedent(setup).strip()}

def run():
{indent(run_body)}
"""
    return make_case(
        identifier,
        "L3",
        "advanced",
        code,
        expected_ops,
        rationale=rationale,
        failure_mode=failure_mode,
    )


def l4_case(
    identifier: str,
    setup: str,
    run_body: str,
    expected_ops: list[str],
    training_config: dict[str, Any],
    *,
    rationale: str,
    failure_mode: str | None = None,
) -> CaseSpec:
    code = f"""
{textwrap.dedent(setup).strip()}

def run():
{indent(run_body)}
"""
    difficulty = "advanced" if training_config.get("steps", 0) >= 50 or training_config.get("epochs", 0) else "intermediate"
    return make_case(
        identifier,
        "L4",
        difficulty,
        code,
        expected_ops,
        training_config=training_config,
        rationale=rationale,
        failure_mode=failure_mode,
    )


RESNET_BLOCK = """
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_ch, out_ch, stride=1, bn_momentum=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch, momentum=bn_momentum)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=True)
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch, momentum=bn_momentum),
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x):
        identity = self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + identity)
        return out


class ResNet18(nn.Module):
    def __init__(self, num_classes=1000, bn_momentum=0.1, trap_mode=None):
        super().__init__()
        self.trap_mode = trap_mode
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1),
        )
        self.layer1 = self._make_layer(64, 64, 2, 1, bn_momentum)
        self.layer2 = self._make_layer(64, 128, 2, 2, bn_momentum)
        self.layer3 = self._make_layer(128, 256, 2, 2, bn_momentum)
        self.layer4 = self._make_layer(256, 512, 2, 2, bn_momentum)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)
        self.debug = {}

    def _make_layer(self, in_ch, out_ch, blocks, stride, bn_momentum):
        layers = [BasicBlock(in_ch, out_ch, stride=stride, bn_momentum=bn_momentum)]
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_ch, out_ch, bn_momentum=bn_momentum))
        return nn.Sequential(*layers)

    def _apply_trap(self, x):
        if self.trap_mode == "smr1":
            trap = (x + x) ** 2 - 4 * x ** 2
            return x + trap * 0
        return x

    def forward(self, x):
        x = self.stem(x)
        self.debug["stem"] = x
        x = self.layer1(x)
        x = self.layer2(x)
        x = self._apply_trap(x)
        self.debug["layer2"] = x
        x = self.layer3(x)
        x = self.layer4(x)
        self.debug["layer4"] = x
        x = torch.flatten(self.pool(x), 1)
        logits = self.fc(x)
        self.debug["logits"] = logits
        return logits
"""


VIT_BLOCK = """
class AttentionBlock(nn.Module):
    def __init__(self, dim=192, heads=3, mlp_dim=768, activation="gelu"):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        act = nn.GELU() if activation == "gelu" else nn.SiLU()
        self.ffn = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            act,
            nn.Dropout(0.1),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        attn_out = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x, attn_out


class ViTTiny(nn.Module):
    def __init__(self, num_classes=1000, activation="gelu", trap_mode=None, use_rope=False):
        super().__init__()
        self.patch = nn.Conv2d(3, 192, 16, 16)
        self.cls = nn.Parameter(torch.zeros(1, 1, 192))
        self.pos = nn.Parameter(torch.zeros(1, 197, 192))
        self.blocks = nn.ModuleList([AttentionBlock(192, 3, 768, activation=activation) for _ in range(12)])
        self.norm = nn.LayerNorm(192)
        self.head = nn.Linear(192, num_classes)
        self.trap_mode = trap_mode
        self.use_rope = use_rope
        self.debug = {}

    def _apply_rope(self, x):
        if not self.use_rope:
            return x
        dim = x.shape[-1]
        freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=x.device, dtype=x.dtype) / dim))
        pos = torch.arange(x.shape[1], device=x.device, dtype=x.dtype)
        theta = pos[:, None] * freq[None, :]
        cos = theta.cos()[None, :, :]
        sin = theta.sin()[None, :, :]
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        rot_even = x_even * cos - x_odd * sin
        rot_odd = x_even * sin + x_odd * cos
        y = torch.empty_like(x)
        y[..., 0::2] = rot_even
        y[..., 1::2] = rot_odd
        return y

    def forward(self, x):
        x = self.patch(x).flatten(2).transpose(1, 2)
        cls = self.cls.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos
        x = self._apply_rope(x)
        self.debug["patch"] = x
        for idx, block in enumerate(self.blocks):
            x, attn_out = block(x)
            if idx == 0:
                self.debug["attn0"] = attn_out
            if idx == 5:
                self.debug["mid"] = x
        if self.trap_mode == "smr3":
            a = x[:, :, :96]
            b = x[:, :, 96:192]
            trap = torch.abs(a) + torch.abs(b) - torch.abs(a + b)
            x = x + trap.mean() * 1e-6
        x = self.norm(x)
        self.debug["norm"] = x
        logits = self.head(x[:, 0])
        self.debug["logits"] = logits
        return logits
"""


MOBILENET_BLOCK = """
class InvertedResidual(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, expand_ratio=6):
        super().__init__()
        hidden = int(in_ch * expand_ratio)
        self.use_res = stride == 1 and in_ch == out_ch
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden, hidden, 3, stride, 1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden, out_ch, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x):
        out = self.block(x)
        if self.use_res:
            out = out + x
        return out


class MobileNetV2Small(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.0, trap_mode=None):
        super().__init__()
        c = lambda v: max(8, int(v * width_mult))
        self.stem = nn.Sequential(
            nn.Conv2d(3, c(32), 3, 2, 1, bias=False),
            nn.BatchNorm2d(c(32)),
            nn.ReLU6(inplace=True),
        )
        self.blocks = nn.ModuleList([
            InvertedResidual(c(32), c(16), 1, 1),
            InvertedResidual(c(16), c(24), 2, 6),
            InvertedResidual(c(24), c(32), 2, 6),
            InvertedResidual(c(32), c(64), 2, 6),
            InvertedResidual(c(64), c(96), 1, 6),
            InvertedResidual(c(96), c(160), 2, 6),
            InvertedResidual(c(160), c(320), 1, 6),
        ])
        self.conv_last = nn.Sequential(
            nn.Conv2d(c(320), c(1280), 1, 1, 0, bias=False),
            nn.BatchNorm2d(c(1280)),
            nn.ReLU6(inplace=True),
        )
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c(1280), num_classes, 1, 1), nn.Flatten())
        self.trap_mode = trap_mode
        self.debug = {}

    def forward(self, x):
        x = self.stem(x)
        self.debug["stem"] = x
        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx == 0:
                self.debug["irb0"] = x
                if self.trap_mode == "smr4":
                    trap_value = x.mean(dim=(2, 3), keepdim=True)
                    x = x + F.relu(trap_value) + F.relu(-trap_value) - torch.abs(trap_value)
            if idx == len(self.blocks) - 1:
                self.debug["tail"] = x
        x = self.conv_last(x)
        self.debug["conv_last"] = x
        logits = self.head(x)
        self.debug["logits"] = logits
        return logits
"""


YOLO_BLOCK = """
class Focus(nn.Module):
    def __init__(self, in_ch=3, out_ch=16):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch * 4, out_ch, 3, 2, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(),
        )

    def forward(self, x):
        x = torch.cat(
            [
                x[..., ::2, ::2],
                x[..., 1::2, ::2],
                x[..., ::2, 1::2],
                x[..., 1::2, 1::2],
            ],
            dim=1,
        )
        return self.conv(x)


class BottleneckCSP(nn.Module):
    def __init__(self, channels, repeats):
        super().__init__()
        layers = []
        for _ in range(repeats):
            layers.extend(
                [
                    nn.Conv2d(channels, channels, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.SiLU(),
                    nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.SiLU(),
                ]
            )
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.net(x)


class SPP(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pools = nn.ModuleList([nn.MaxPool2d(k, 1, k // 2) for k in (5, 9, 13)])
        self.mix = nn.Sequential(
            nn.Conv2d(channels * 4, channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(),
        )

    def forward(self, x):
        stacked = torch.cat([x] + [pool(x) for pool in self.pools], dim=1)
        return self.mix(stacked)


class YOLOv5nBackbone(nn.Module):
    def __init__(self, trap_mode=None, with_head=False):
        super().__init__()
        self.focus = Focus(3, 16)
        self.conv1 = nn.Sequential(nn.Conv2d(16, 32, 3, 2, 1, bias=False), nn.BatchNorm2d(32), nn.SiLU())
        self.c3_1 = BottleneckCSP(32, 1)
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, 3, 2, 1, bias=False), nn.BatchNorm2d(64), nn.SiLU())
        self.c3_2 = BottleneckCSP(64, 2)
        self.conv3 = nn.Sequential(nn.Conv2d(64, 128, 3, 2, 1, bias=False), nn.BatchNorm2d(128), nn.SiLU())
        self.c3_3 = BottleneckCSP(128, 3)
        self.spp = SPP(128)
        self.with_head = with_head
        self.head = nn.Conv2d(128, 24, 1, 1) if with_head else nn.Identity()
        self.trap_mode = trap_mode
        self.debug = {}

    def forward(self, x):
        x = self.focus(x)
        self.debug["focus"] = x
        x = self.conv1(x)
        x = self.c3_1(x)
        x = self.conv2(x)
        x = self.c3_2(x)
        x = self.conv3(x)
        x = self.c3_3(x)
        x = self.spp(x)
        if self.trap_mode == "smr2":
            a = x[:, :, :, : x.shape[-1] // 2]
            b = x[:, :, :, x.shape[-1] // 2 :]
            crop = min(a.shape[-1], b.shape[-1])
            trap = torch.abs(a[..., :crop]) + torch.abs(b[..., :crop]) - torch.abs(a[..., :crop] + b[..., :crop])
            x = x + trap.mean() * 0
        self.debug["spp"] = x
        x = self.head(x)
        self.debug["head"] = x
        return x
"""


UNET_BLOCK = """
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, with_bn=False):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, 3, 1, 1)]
        if with_bn:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(out_ch, out_ch, 3, 1, 1))
        if with_bn:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class UNetSmall(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, depth=2, with_bn=False, bilinear=False):
        super().__init__()
        channels = [64, 128, 256, 512]
        self.depth = depth
        self.enc1 = DoubleConv(in_ch, channels[0], with_bn)
        self.enc2 = DoubleConv(channels[0], channels[1], with_bn)
        self.bottleneck = DoubleConv(channels[1], channels[2], with_bn)
        self.pool = nn.MaxPool2d(2)
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False) if bilinear else nn.ConvTranspose2d(channels[2], channels[1], 2, 2)
        self.dec2 = DoubleConv(channels[1] * 2, channels[1], with_bn)
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False) if bilinear else nn.ConvTranspose2d(channels[1], channels[0], 2, 2)
        self.dec1 = DoubleConv(channels[0] * 2, channels[0], with_bn)
        if depth == 3:
            self.enc3 = DoubleConv(channels[1], channels[2], with_bn)
            self.bottleneck = DoubleConv(channels[2], channels[3], with_bn)
            self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False) if bilinear else nn.ConvTranspose2d(channels[3], channels[2], 2, 2)
            self.dec3 = DoubleConv(channels[2] * 2, channels[2], with_bn)
            self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False) if bilinear else nn.ConvTranspose2d(channels[2], channels[1], 2, 2)
            self.dec2 = DoubleConv(channels[1] * 2, channels[1], with_bn)
        self.out = nn.Conv2d(channels[0], out_ch, 1)
        self.debug = {}

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        if self.depth == 2:
            b = self.bottleneck(self.pool(e2))
            d2 = self.up2(b)
            d2 = self.dec2(torch.cat([d2, e2], dim=1))
        else:
            e3 = self.enc3(self.pool(e2))
            b = self.bottleneck(self.pool(e3))
            d3 = self.up3(b)
            d3 = self.dec3(torch.cat([d3, e3], dim=1))
            d2 = self.up2(d3)
            d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        self.debug["enc1"] = e1
        self.debug["bottleneck"] = b
        self.debug["dec1"] = d1
        logits = self.out(d1)
        self.debug["logits"] = logits
        return logits
"""


LSTM_TRAIN_BLOCK = """
class LSTMClassifier(nn.Module):
    def __init__(self, input_size=64, hidden_size=128, num_layers=2, num_classes=10):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.debug = {}

    def forward(self, x):
        out, (h, c) = self.lstm(x)
        self.debug["sequence"] = out
        self.debug["hidden"] = h
        logits = self.fc(out[:, -1])
        self.debug["logits"] = logits
        return logits
"""


def build_l1_cases() -> list[CaseSpec]:
    cases = [
        l1_case("bench_v1.0.0/L1/conv/conv2d_fp32", "conv = nn.Conv2d(3, 64, 3, 1, 1).to(device=device, dtype=dtype)\nx = torch.randn(1, 3, 224, 224, device=device, dtype=dtype)", "conv(x)", ["torch.nn.Conv2d"], rationale="Conv2d FP32 smoke."),
        l1_case("bench_v1.0.0/L1/conv/conv2d_fp16", "conv = nn.Conv2d(3, 64, 3, 2, 1).to(device=device, dtype=dtype)\nx = torch.randn(1, 3, 224, 224, device=device, dtype=dtype)", "conv(x)", ["torch.nn.Conv2d"], precision="FP16", rationale="Conv2d FP16 path with CPU fallback."),
        l1_case("bench_v1.0.0/L1/conv/conv2d_stride2", "conv = nn.Conv2d(64, 128, 3, 2, 1).to(device=device, dtype=dtype)\nx = torch.randn(1, 64, 112, 112, device=device, dtype=dtype)", "conv(x)", ["torch.nn.Conv2d"], rationale="Stride-2 convolution."),
        l1_case("bench_v1.0.0/L1/conv/conv_transpose2d", "conv = nn.ConvTranspose2d(64, 32, 4, 2, 1).to(device=device, dtype=dtype)\nx = torch.randn(1, 64, 14, 14, device=device, dtype=dtype)", "conv(x)", ["torch.nn.ConvTranspose2d"], rationale="Transposed convolution."),
        l1_case("bench_v1.0.0/L1/conv/conv1d_fp32", "conv = nn.Conv1d(16, 33, 3, 1, 1).to(device=device, dtype=dtype)\nx = torch.randn(20, 16, 50, device=device, dtype=dtype)", "conv(x)", ["torch.nn.Conv1d"], rationale="Conv1d FP32."),
        l1_case("bench_v1.0.0/L1/conv/conv1d_causal", "conv = nn.Conv1d(64, 128, 5, 1, 2).to(device=device, dtype=dtype)\nx = torch.randn(8, 64, 100, device=device, dtype=dtype)", "conv(x)", ["torch.nn.Conv1d"], rationale="Long-sequence temporal Conv1d."),
        l1_case("bench_v1.0.0/L1/conv/conv3d", "conv = nn.Conv3d(3, 16, 3, 1, 1).to(device=device, dtype=dtype)\nx = torch.randn(1, 3, 16, 224, 224, device=device, dtype=dtype)", "conv(x)", ["torch.nn.Conv3d"], rationale="Conv3d FP32."),
        l1_case("bench_v1.0.0/L1/conv/conv3d_stride2", "conv = nn.Conv3d(16, 32, 3, 2, 1).to(device=device, dtype=dtype)\nx = torch.randn(2, 16, 14, 56, 56, device=device, dtype=dtype)", "conv(x)", ["torch.nn.Conv3d"], rationale="Conv3d stride-2."),
        l1_case("bench_v1.0.0/L1/conv/conv_transpose2d_stride4", "conv = nn.ConvTranspose2d(16, 3, 4, 2, 1).to(device=device, dtype=dtype)\nx = torch.randn(1, 16, 56, 56, device=device, dtype=dtype)", "conv(x)", ["torch.nn.ConvTranspose2d"], rationale="Generator-style transposed convolution."),
        l1_case("bench_v1.0.0/L1/conv/conv2d_depthwise", "conv = nn.Conv2d(64, 64, 3, 1, 1, groups=64).to(device=device, dtype=dtype)\nx = torch.randn(1, 64, 56, 56, device=device, dtype=dtype)", "conv(x)", ["torch.nn.Conv2d"], rationale="Depthwise convolution."),
        l1_case("bench_v1.0.0/L1/norm/batchnorm2d", "bn = nn.BatchNorm2d(64).to(device=device, dtype=dtype)\nbn.train()\nx = torch.randn(4, 64, 56, 56, device=device, dtype=dtype)", "bn(x)", ["torch.nn.BatchNorm2d"], rationale="BatchNorm2d train mode."),
        l1_case("bench_v1.0.0/L1/norm/batchnorm2d_momentum", "bn = nn.BatchNorm2d(64, momentum=0.05).to(device=device, dtype=dtype)\nbn.train()\nx = torch.randn(4, 64, 56, 56, device=device, dtype=dtype)", "bn(x)", ["torch.nn.BatchNorm2d"], rationale="BatchNorm2d custom momentum."),
        l1_case("bench_v1.0.0/L1/norm/layernorm", "ln = nn.LayerNorm(768).to(device=device, dtype=dtype)\nx = torch.randn(32, 197, 768, device=device, dtype=dtype)", "ln(x)", ["torch.nn.LayerNorm"], rationale="LayerNorm on token sequence."),
        l1_case("bench_v1.0.0/L1/norm/groupnorm", "gn = nn.GroupNorm(32, 128).to(device=device, dtype=dtype)\nx = torch.randn(2, 128, 32, 32, device=device, dtype=dtype)", "gn(x)", ["torch.nn.GroupNorm"], rationale="GroupNorm spatial case."),
        l1_case("bench_v1.0.0/L1/norm/instancenorm2d", "norm = nn.InstanceNorm2d(64).to(device=device, dtype=dtype)\nnorm.train()\nx = torch.randn(2, 64, 32, 32, device=device, dtype=dtype)", "norm(x)", ["torch.nn.InstanceNorm2d"], rationale="InstanceNorm2d default affine semantics.", failure_mode="instance_norm_affine_semantics"),
        l1_case("bench_v1.0.0/L1/norm/rmsnorm", "norm = nn.RMSNorm(512).to(device=device, dtype=dtype)\nx = torch.randn(4, 512, device=device, dtype=dtype)", "norm(x)", ["torch.nn.RMSNorm"], rationale="RMSNorm forward."),
        l1_case("bench_v1.0.0/L1/act/relu", "x = torch.randn(4, 64, 56, 56, device=device, dtype=dtype)", "nn.ReLU()(x)", ["torch.nn.ReLU"], rationale="ReLU activation."),
        l1_case("bench_v1.0.0/L1/act/gelu", "x = torch.randn(4, 768, device=device, dtype=dtype)", "nn.GELU()(x)", ["torch.nn.GELU"], rationale="GELU activation."),
        l1_case("bench_v1.0.0/L1/act/silu", "x = torch.randn(4, 512, device=device, dtype=dtype)", "nn.SiLU()(x)", ["torch.nn.SiLU"], rationale="SiLU activation."),
        l1_case("bench_v1.0.0/L1/act/leakyrelu_default", "x = torch.randn(4, 64, 28, 28, device=device, dtype=dtype)", "nn.LeakyReLU()(x)", ["torch.nn.LeakyReLU"], rationale="LeakyReLU default slope."),
        l1_case("bench_v1.0.0/L1/act/leakyrelu_neg01", "x = torch.randn(4, 64, 28, 28, device=device, dtype=dtype)", "nn.LeakyReLU(negative_slope=0.01)(x)", ["torch.nn.LeakyReLU"], rationale="LeakyReLU explicit slope."),
        l1_case("bench_v1.0.0/L1/act/sigmoid", "x = torch.randn(4, 256, device=device, dtype=dtype)", "torch.sigmoid(x)", ["torch.sigmoid"], rationale="Sigmoid activation."),
        l1_case("bench_v1.0.0/L1/act/tanh", "x = torch.randn(4, 256, device=device, dtype=dtype)", "torch.tanh(x)", ["torch.tanh"], rationale="Tanh activation."),
        l1_case("bench_v1.0.0/L1/act/softmax", "x = torch.randn(4, 1000, device=device, dtype=dtype)", "nn.Softmax(dim=-1)(x)", ["torch.nn.Softmax"], rationale="Softmax over class dimension."),
        l1_case("bench_v1.0.0/L1/act/mish", "x = torch.randn(4, 64, 28, 28, device=device, dtype=dtype)", "nn.Mish()(x)", ["torch.nn.Mish"], rationale="Mish activation."),
        l1_case("bench_v1.0.0/L1/act/hardswish", "x = torch.randn(4, 64, 28, 28, device=device, dtype=dtype)", "nn.Hardswish()(x)", ["torch.nn.Hardswish"], rationale="HardSwish activation."),
        l1_case("bench_v1.0.0/L1/linear/linear_fp32", "fc = nn.Linear(768, 1000).to(device=device, dtype=dtype)\nx = torch.randn(32, 768, device=device, dtype=dtype)", "fc(x)", ["torch.nn.Linear"], rationale="Linear layer FP32."),
        l1_case("bench_v1.0.0/L1/linear/linear_bias_false", "fc = nn.Linear(512, 2048, bias=False).to(device=device, dtype=dtype)\nx = torch.randn(16, 512, device=device, dtype=dtype)", "fc(x)", ["torch.nn.Linear"], rationale="Linear without bias."),
        l1_case("bench_v1.0.0/L1/attn/matmul", "a = torch.randn(4, 8, 64, 64, device=device, dtype=dtype)\nb = torch.randn(4, 8, 64, 64, device=device, dtype=dtype).transpose(-2, -1)", "torch.matmul(a, b)", ["torch.matmul"], rationale="Attention-style matmul."),
        l1_case("bench_v1.0.0/L1/attn/scaled_dot_product", "q = torch.randn(4, 8, 100, 64, device=device, dtype=dtype)\nk = torch.randn(4, 8, 100, 64, device=device, dtype=dtype)\nv = torch.randn(4, 8, 100, 64, device=device, dtype=dtype)", "F.scaled_dot_product_attention(q, k, v)", ["torch.nn.functional.scaled_dot_product_attention"], rationale="Native scaled dot-product attention."),
        l1_case("bench_v1.0.0/L1/attn/bmm", "a = torch.randn(10, 3, 4, device=device, dtype=dtype)\nb = torch.randn(10, 4, 5, device=device, dtype=dtype)", "torch.bmm(a, b)", ["torch.bmm"], rationale="Batch matrix multiplication."),
        l1_case("bench_v1.0.0/L1/attn/einsum", "a = torch.randn(2, 10, 64, device=device, dtype=dtype)\nb = torch.randn(2, 10, 64, device=device, dtype=dtype)", "torch.einsum('bqd,bkd->bqk', a, b)", ["torch.einsum"], rationale="Attention score einsum."),
        l1_case("bench_v1.0.0/L1/pool/maxpool2d", "x = torch.randn(4, 64, 112, 112, device=device, dtype=dtype)", "nn.MaxPool2d(2, 2)(x)", ["torch.nn.MaxPool2d"], rationale="MaxPool2d."),
        l1_case("bench_v1.0.0/L1/pool/avgpool2d", "x = torch.randn(4, 2048, 7, 7, device=device, dtype=dtype)", "nn.AvgPool2d(7)(x)", ["torch.nn.AvgPool2d"], rationale="AvgPool2d global-ish pooling."),
        l1_case("bench_v1.0.0/L1/pool/adaptive_avgpool2d", "x = torch.randn(4, 2048, 7, 7, device=device, dtype=dtype)", "nn.AdaptiveAvgPool2d(1)(x)", ["torch.nn.AdaptiveAvgPool2d"], rationale="AdaptiveAvgPool2d."),
        l1_case("bench_v1.0.0/L1/pool/adaptive_maxpool2d", "x = torch.randn(4, 512, 14, 14, device=device, dtype=dtype)", "nn.AdaptiveMaxPool2d(1)(x)", ["torch.nn.AdaptiveMaxPool2d"], rationale="AdaptiveMaxPool2d."),
        l1_case("bench_v1.0.0/L1/reg/dropout_train", "drop = nn.Dropout(p=0.5).to(device=device)\ndrop.train()\nx = torch.randn(4, 1024, device=device, dtype=dtype)", "drop(x)", ["torch.nn.Dropout"], rationale="Dropout in train mode.", failure_mode="dropout_semantics"),
        l1_case("bench_v1.0.0/L1/reg/dropout2d", "drop = nn.Dropout2d(p=0.2).to(device=device)\ndrop.train()\nx = torch.randn(4, 64, 32, 32, device=device, dtype=dtype)", "drop(x)", ["torch.nn.Dropout2d"], rationale="Spatial dropout."),
        l1_case("bench_v1.0.0/L1/tensor/concat", "a = torch.randn(4, 256, device=device, dtype=dtype)\nb = torch.randn(4, 512, device=device, dtype=dtype)", "torch.cat([a, b], dim=1)", ["torch.cat"], rationale="Tensor concatenation."),
        l1_case("bench_v1.0.0/L1/tensor/reshape", "x = torch.randn(4, 768, device=device, dtype=dtype)", "torch.reshape(x, (4, 12, 64))", ["torch.reshape"], rationale="Tensor reshape."),
        l1_case("bench_v1.0.0/L1/tensor/permute", "x = torch.randn(4, 197, 768, device=device, dtype=dtype)", "torch.permute(x, (0, 2, 1))", ["torch.permute"], rationale="Tensor permute."),
        l1_case("bench_v1.0.0/L1/tensor/index_select", "x = torch.randn(4, 1000, device=device, dtype=dtype)\nindex = torch.tensor([1, 3, 5, 7], device=device)", "torch.index_select(x, dim=1, index=index)", ["torch.index_select"], rationale="Index select gather."),
        l1_case("bench_v1.0.0/L1/tensor/expand_repeat", "x = torch.randn(1, 10, 64, device=device, dtype=dtype)", "{\"expand\": x.expand(4, 10, 64), \"repeat\": x.repeat(1, 2, 1)}", ["Tensor.expand", "Tensor.repeat"], rationale="Expand and repeat semantics."),
        l1_case("bench_v1.0.0/L1/tensor/scatter", "x = torch.zeros(5, 5, device=device, dtype=dtype)\nidx = torch.tensor([[0, 1, 2, 0, 0], [1, 2, 3, 0, 0], [2, 3, 4, 0, 0], [3, 4, 0, 0, 0], [4, 0, 1, 0, 0]], device=device)\nsrc = torch.randn(5, 5, device=device, dtype=dtype)", "x.scatter(0, idx, src)", ["Tensor.scatter_"], rationale="Scatter semantics."),
        l1_case("bench_v1.0.0/L1/tensor/gather", "x = torch.randn(4, 8, device=device, dtype=dtype)\nidx = torch.tensor([[0, 2, 4, 6], [1, 3, 5, 7], [7, 5, 3, 1], [6, 4, 2, 0]], device=device)", "torch.gather(x, dim=1, index=idx)", ["torch.gather"], rationale="Gather semantics."),
        l1_case("bench_v1.0.0/L1/reduce/sum", "x = torch.randn(4, 256, device=device, dtype=dtype)", "torch.sum(x, dim=-1)", ["torch.sum"], rationale="Reduction sum."),
        l1_case("bench_v1.0.0/L1/reduce/mean", "x = torch.randn(4, 8, 64, device=device, dtype=dtype)", "torch.mean(x, dim=(1, 2))", ["torch.mean"], rationale="Reduction mean over tuple dims."),
        l1_case("bench_v1.0.0/L1/reduce/argmax", "x = torch.randn(4, 1000, device=device, dtype=dtype)", "torch.argmax(x, dim=-1)", ["torch.argmax"], rationale="Argmax reduction."),
        l1_case("bench_v1.0.0/L1/reduce/topk", "x = torch.randn(4, 1000, device=device, dtype=dtype)", "torch.topk(x, k=5, dim=-1)", ["torch.topk"], rationale="Top-k ranking."),
        l1_case("bench_v1.0.0/L1/reduce/sort", "x = torch.randn(4, 256, device=device, dtype=dtype)", "torch.sort(x, dim=-1, descending=True)", ["torch.sort"], rationale="Descending sort."),
        l1_case("bench_v1.0.0/L1/risk/dropout_p_inverted", "drop = nn.Dropout(p=0.3).to(device=device)\ndrop.train()\nx = torch.randn(4, 1024, device=device, dtype=dtype)", "drop(x)", ["torch.nn.Dropout"], rationale="Dropout p-vs-keep_prob semantic risk.", failure_mode="dropout_semantics"),
        l1_case("bench_v1.0.0/L1/risk/bn_momentum_inverted", "bn = nn.BatchNorm2d(64, momentum=0.1).to(device=device, dtype=dtype)\nbn.train()\nx = torch.randn(4, 64, 32, 32, device=device, dtype=dtype)", "bn(x)", ["torch.nn.BatchNorm2d"], rationale="BatchNorm momentum semantic risk.", failure_mode="bn_momentum_semantics"),
        l1_case("bench_v1.0.0/L1/risk/interpolate_align", "x = torch.randn(1, 3, 14, 14, device=device, dtype=dtype)", "F.interpolate(x, size=(28, 28), mode='bilinear', align_corners=False)", ["torch.nn.functional.interpolate"], rationale="Interpolate align_corners semantic risk.", failure_mode="interpolate_align_corners"),
        l1_case("bench_v1.0.0/L1/complex/polar", "amp = torch.ones(4, 64, device=device)\nphase = torch.randn(4, 64, device=device)", "torch.polar(amp, phase)", ["torch.polar"], rationale="Complex polar construction.", difficulty="advanced", failure_mode="complex_unsupported"),
        l1_case("bench_v1.0.0/L1/complex/complex_mul", "a = torch.polar(torch.ones(32, 64, device=device), torch.randn(32, 64, device=device))", "a * a.conj()", ["torch.polar", "complex:multiply"], rationale="Complex multiplication.", difficulty="advanced", failure_mode="complex_unsupported"),
        l1_case("bench_v1.0.0/L1/complex/view_as_real", "z = torch.randn(4, 64, device=device, dtype=torch.cfloat)", "torch.view_as_real(z)", ["torch.view_as_real"], rationale="Complex to real view.", difficulty="advanced", failure_mode="complex_unsupported"),
        l1_case("bench_v1.0.0/L1/complex/rope_rotation", "x = torch.randn(2, 16, 64, device=device, dtype=dtype)\ntheta = torch.randn(16, 32, device=device, dtype=dtype)\ndef rope(x, theta):\n    amp = torch.ones_like(theta)\n    z = torch.polar(amp, theta)\n    x2 = x.reshape(x.shape[0], x.shape[1], -1, 2).contiguous()\n    z = z.unsqueeze(0)\n    base = torch.view_as_complex(x2)\n    return torch.view_as_real(base * z).reshape_as(x)\n", "rope(x, theta)", ["torch.polar", "torch.view_as_complex", "torch.view_as_real"], rationale="End-to-end RoPE rotation.", difficulty="advanced", failure_mode="complex_unsupported"),
        l1_case("bench_v1.0.0/L1/complex/complex_abs_angle", "z = torch.randn(4, 64, device=device, dtype=torch.cfloat)", "{\"abs\": torch.abs(z), \"angle\": torch.angle(z)}", ["torch.abs", "torch.angle"], rationale="Complex abs and angle.", difficulty="advanced", failure_mode="complex_unsupported"),
        l1_case("bench_v1.0.0/L1/conv/conv2d_dilated", "conv = nn.Conv2d(32, 32, 3, 1, 2, dilation=2).to(device=device, dtype=dtype)\nx = torch.randn(2, 32, 64, 64, device=device, dtype=dtype)", "conv(x)", ["torch.nn.Conv2d"], rationale="Dilated convolution from existing bridge hot paths."),
        l1_case("bench_v1.0.0/L1/norm/batchnorm1d", "bn = nn.BatchNorm1d(128).to(device=device, dtype=dtype)\nbn.train()\nx = torch.randn(8, 128, device=device, dtype=dtype)", "bn(x)", ["torch.nn.BatchNorm1d"], rationale="BatchNorm1d variant."),
        l1_case("bench_v1.0.0/L1/act/relu6", "x = torch.randn(4, 64, 28, 28, device=device, dtype=dtype)", "nn.ReLU6()(x)", ["torch.nn.ReLU6"], rationale="ReLU6 used in MobileNetV2."),
        l1_case("bench_v1.0.0/L1/linear/linear_3d", "fc = nn.Linear(256, 512).to(device=device, dtype=dtype)\nx = torch.randn(4, 32, 256, device=device, dtype=dtype)", "fc(x)", ["torch.nn.Linear"], rationale="Linear projection over sequence batch."),
        l1_case("bench_v1.0.0/L1/pool/interpolate_nearest", "x = torch.randn(1, 3, 32, 32, device=device, dtype=dtype)", "F.interpolate(x, scale_factor=2.0, mode='nearest')", ["torch.nn.functional.interpolate"], rationale="Nearest-neighbor resize."),
        l1_case("bench_v1.0.0/L1/tensor/flatten", "x = torch.randn(4, 16, 8, 8, device=device, dtype=dtype)", "torch.flatten(x, 1)", ["torch.flatten"], rationale="Flatten before linear head."),
        l1_case("bench_v1.0.0/L1/reduce/std", "x = torch.randn(8, 64, device=device, dtype=dtype)", "torch.std(x, dim=-1)", ["torch.std"], rationale="Standard deviation reduction."),
        l1_case("bench_v1.0.0/L1/attn/masked_fill", "scores = torch.randn(4, 8, 32, 32, device=device, dtype=dtype)\nmask = torch.triu(torch.ones(32, 32, device=device, dtype=torch.bool), diagonal=1)", "scores.masked_fill(mask, float('-inf'))", ["Tensor.masked_fill"], rationale="Masked attention score path."),
        l1_case("bench_v1.0.0/L1/complex/view_as_complex", "x = torch.randn(4, 64, 2, device=device, dtype=dtype)", "torch.view_as_complex(x)", ["torch.view_as_complex"], rationale="Real-to-complex view.", difficulty="advanced", failure_mode="complex_unsupported"),
    ]
    return cases


def generic_l2_model_setup(model_code: str, model_ctor: str, input_setup: str, forward_expr: str) -> tuple[str, str]:
    setup = f"""
{model_code}
"""
    run_body = f"""
device = select_device()
precision = "FP32"
dtype = precision_dtype(precision, device)
model = {model_ctor}.to(device=device, dtype=dtype)
{textwrap.dedent(input_setup).strip()}
out = {forward_expr}
primary = out[0] if isinstance(out, tuple) else out
loss = primary.float().mean()
loss.backward()
globals()["ACTIVATIONS"] = {{
    "output_mean": tensor_mean(primary),
    "output_sample": tensor_sample(primary),
}}
globals()["GRADIENTS"] = {{
    "input_grad_norm": grad_norm(x.grad if 'x' in locals() else None),
    "param_grad_norm": grad_norm(first_grad(model)),
}}
return primary
"""
    return setup, run_body


def build_l2_cases() -> list[CaseSpec]:
    convbn_setup = """
class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))
"""
    residual_setup = """
class ResidualBlock(nn.Module):
    def __init__(self, c_in=64, c_out=64, stride=1, bottleneck=False, groups=1):
        super().__init__()
        self.relu = nn.ReLU()
        self.bottleneck = bottleneck
        if bottleneck:
            self.conv1 = nn.Conv2d(c_in, c_out // 4, 1, 1, 0, bias=False)
            self.bn1 = nn.BatchNorm2d(c_out // 4)
            self.conv2 = nn.Conv2d(c_out // 4, c_out // 4, 3, stride, 1, groups=groups, bias=False)
            self.bn2 = nn.BatchNorm2d(c_out // 4)
            self.conv3 = nn.Conv2d(c_out // 4, c_out, 1, 1, 0, bias=False)
            self.bn3 = nn.BatchNorm2d(c_out)
        else:
            self.conv1 = nn.Conv2d(c_in, c_out, 3, stride, 1, groups=groups, bias=False)
            self.bn1 = nn.BatchNorm2d(c_out)
            self.conv2 = nn.Conv2d(c_out, c_out, 3, 1, 1, groups=groups, bias=False)
            self.bn2 = nn.BatchNorm2d(c_out)
        if stride != 1 or c_in != c_out:
            self.down = nn.Sequential(nn.Conv2d(c_in, c_out, 1, stride, bias=False), nn.BatchNorm2d(c_out))
        else:
            self.down = nn.Identity()

    def forward(self, x):
        identity = self.down(x)
        if self.bottleneck:
            out = self.relu(self.bn1(self.conv1(x)))
            out = self.relu(self.bn2(self.conv2(out)))
            out = self.bn3(self.conv3(out))
        else:
            out = self.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
        return self.relu(out + identity)
"""
    se_setup = """
class SEBlock(nn.Module):
    def __init__(self, c=64, r=16):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(c, c // r)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(c // r, c)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.shape
        w = self.gap(x).view(b, c)
        w = self.sigmoid(self.fc2(self.relu(self.fc1(w)))).view(b, c, 1, 1)
        return x * w
"""
    mbconv_setup = MOBILENET_BLOCK
    attention_setup = """
class SelfAttention(nn.Module):
    def __init__(self, dim=256, heads=8, use_native_sdp=False, use_mask=False):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.use_native_sdp = use_native_sdp
        self.use_mask = use_mask
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        bsz, seq, dim = x.shape
        qkv = self.qkv(x).reshape(bsz, seq, 3, self.heads, dim // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if self.use_native_sdp:
            out = F.scaled_dot_product_attention(q, k, v)
        else:
            attn = (q @ k.transpose(-2, -1)) * (dim // self.heads) ** -0.5
            if self.use_mask:
                mask = torch.triu(torch.ones(seq, seq, device=x.device, dtype=torch.bool), diagonal=1)
                attn = attn.masked_fill(mask, float("-inf"))
            attn = attn.softmax(dim=-1)
            out = attn @ v
        out = out.transpose(1, 2).reshape(bsz, seq, dim)
        return self.proj(self.drop(out))
"""
    encoder_setup = VIT_BLOCK
    ffn_setup = """
class FFN(nn.Module):
    def __init__(self, dim=256, hidden=1024, activation="gelu"):
        super().__init__()
        act = nn.GELU() if activation == "gelu" else nn.ReLU()
        self.net = nn.Sequential(nn.Linear(dim, hidden), act, nn.Linear(hidden, dim), nn.Dropout(0.1))

    def forward(self, x):
        return self.net(x)

class SwiGLUFFN(nn.Module):
    def __init__(self, dim=256, hidden=1024):
        super().__init__()
        self.gate = nn.Linear(dim, hidden)
        self.up = nn.Linear(dim, hidden)
        self.down = nn.Linear(hidden, dim)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))
"""
    cnn_head_setup = """
class CNNHead(nn.Module):
    def __init__(self, in_c=512, num_cls=1000, dropout=0.0, hidden=None):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()
        if hidden is None:
            self.fc = nn.Linear(in_c, num_cls)
        else:
            self.fc = nn.Sequential(nn.Linear(in_c, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, num_cls))

    def forward(self, x):
        x = torch.flatten(self.pool(x), 1)
        x = self.dropout(x)
        return self.fc(x)
"""
    bnm_setup = """
class BNMomentumTest(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm2d(16, momentum=0.1)
        self.conv = nn.Conv2d(16, 16, 3, 1, 1)

    def forward(self, x):
        return self.bn(self.conv(x))
"""
    grad_clip_setup = """
class GradClipModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(256, 256)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.fc(x))
"""
    rmsnorm_setup = """
class RMSNormManual(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + 1e-6)
        return x * self.weight
"""
    swiglu_setup = ffn_setup

    cases: list[CaseSpec] = []
    setup, body = generic_l2_model_setup(convbn_setup, "ConvBNReLU(64, 64)", "x = torch.randn(4, 64, 56, 56, device=device, dtype=dtype, requires_grad=True)", "model(x)")
    cases.append(l2_case("bench_v1.0.0/L2/cnn/conv_bn_relu_64ch", setup, body, ["torch.nn.Conv2d", "torch.nn.BatchNorm2d", "torch.nn.ReLU"], rationale="SORT high-frequency Conv-BN-ReLU block."))
    setup, body = generic_l2_model_setup(convbn_setup, "ConvBNReLU(128, 256, stride=2)", "x = torch.randn(4, 128, 28, 28, device=device, dtype=dtype, requires_grad=True)", "model(x)")
    cases.append(l2_case("bench_v1.0.0/L2/cnn/conv_bn_relu_128ch_stride2", setup, body, ["torch.nn.Conv2d", "torch.nn.BatchNorm2d", "torch.nn.ReLU"], rationale="Stride-2 Conv-BN-ReLU block."))
    setup, body = generic_l2_model_setup(convbn_setup, "ConvBNReLU(256, 64, kernel_size=1, stride=1, padding=0)", "x = torch.randn(4, 256, 14, 14, device=device, dtype=dtype, requires_grad=True)", "model(x)")
    cases.append(l2_case("bench_v1.0.0/L2/cnn/conv_bn_relu_1x1_proj", setup, body, ["torch.nn.Conv2d", "torch.nn.BatchNorm2d", "torch.nn.ReLU"], rationale="Projection Conv-BN-ReLU block."))
    setup, body = generic_l2_model_setup(residual_setup, "ResidualBlock()", "x = torch.randn(4, 64, 32, 32, device=device, dtype=dtype, requires_grad=True)", "model(x)")
    cases.append(l2_case("bench_v1.0.0/L2/cnn/residual_block_basic", setup, body, ["torch.nn.Conv2d", "torch.nn.BatchNorm2d", "torch.nn.ReLU"], rationale="Basic residual block."))
    setup, body = generic_l2_model_setup(residual_setup, "ResidualBlock(64, 128, stride=2)", "x = torch.randn(4, 64, 56, 56, device=device, dtype=dtype, requires_grad=True)", "model(x)")
    cases.append(l2_case("bench_v1.0.0/L2/cnn/residual_block_downsample", setup, body, ["torch.nn.Conv2d", "torch.nn.BatchNorm2d", "torch.nn.ReLU"], rationale="Downsample residual block."))
    setup, body = generic_l2_model_setup(residual_setup, "ResidualBlock(256, 256, bottleneck=True)", "x = torch.randn(4, 256, 56, 56, device=device, dtype=dtype, requires_grad=True)", "model(x)")
    cases.append(l2_case("bench_v1.0.0/L2/cnn/residual_block_bottleneck", setup, body, ["torch.nn.Conv2d", "torch.nn.BatchNorm2d", "torch.nn.ReLU"], rationale="Bottleneck residual block."))
    setup, body = generic_l2_model_setup(se_setup, "SEBlock(64, 16)", "x = torch.randn(4, 64, 32, 32, device=device, dtype=dtype, requires_grad=True)", "model(x)")
    cases.append(l2_case("bench_v1.0.0/L2/cnn/se_block", setup, body, ["torch.nn.AdaptiveAvgPool2d", "torch.nn.Linear", "torch.nn.Sigmoid"], rationale="Squeeze-and-excitation block."))
    setup, body = generic_l2_model_setup(se_setup, "SEBlock(128, 16)", "x = torch.randn(4, 128, 28, 28, device=device, dtype=dtype, requires_grad=True)", "model(x)")
    cases.append(l2_case("bench_v1.0.0/L2/cnn/se_block_128ch", setup, body, ["torch.nn.AdaptiveAvgPool2d", "torch.nn.Linear", "torch.nn.Sigmoid"], rationale="SE block 128 channels."))
    setup, body = generic_l2_model_setup(mbconv_setup, "InvertedResidual(32, 32, 1, 6)", "x = torch.randn(4, 32, 112, 112, device=device, dtype=dtype, requires_grad=True)", "model(x)")
    cases.append(l2_case("bench_v1.0.0/L2/cnn/mbconv_inverted_residual", setup, body, ["torch.nn.Conv2d", "torch.nn.BatchNorm2d", "torch.nn.ReLU6"], rationale="MobileNetV2 inverted residual block."))
    setup, body = generic_l2_model_setup(attention_setup, "SelfAttention(256, 8)", "x = torch.randn(4, 197, 256, device=device, dtype=dtype, requires_grad=True)", "model(x)")
    cases.append(l2_case("bench_v1.0.0/L2/transformer/self_attention", setup, body, ["torch.nn.Linear", "torch.matmul", "torch.nn.Softmax"], rationale="Hand-written self-attention block."))
    setup, body = generic_l2_model_setup(attention_setup, "SelfAttention(768, 12, use_native_sdp=True)", "x = torch.randn(4, 197, 768, device=device, dtype=dtype, requires_grad=True)", "model(x)")
    cases.append(l2_case("bench_v1.0.0/L2/transformer/self_attention_scaled_dpa", setup, body, ["torch.nn.functional.scaled_dot_product_attention"], rationale="Self-attention via native SDPA."))
    setup = ""
    body = """
device = select_device()
precision = "FP32"
dtype = precision_dtype(precision, device)
attn = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True).to(device=device, dtype=dtype)
x = torch.randn(4, 100, 512, device=device, dtype=dtype, requires_grad=True)
out, weights = attn(x, x, x)
loss = out.float().mean()
loss.backward()
globals()["ACTIVATIONS"] = {"output_mean": tensor_mean(out), "weight_mean": tensor_mean(weights)}
globals()["GRADIENTS"] = {"input_grad_norm": grad_norm(x.grad), "param_grad_norm": grad_norm(first_grad(attn))}
return out
"""
    cases.append(l2_case("bench_v1.0.0/L2/transformer/multihead_attn", setup, body, ["torch.nn.MultiheadAttention"], rationale="nn.MultiheadAttention path."))
    setup, body = generic_l2_model_setup(encoder_setup, "AttentionBlock(256, 8, 1024, activation='gelu')", "x = torch.randn(4, 197, 256, device=device, dtype=dtype, requires_grad=True)", "model(x)")
    cases.append(l2_case("bench_v1.0.0/L2/transformer/encoder_layer", setup, body, ["torch.nn.LayerNorm", "torch.nn.MultiheadAttention", "torch.nn.GELU"], rationale="Transformer encoder layer."))
    setup, body = generic_l2_model_setup(encoder_setup, "AttentionBlock(768, 12, 3072, activation='silu')", "x = torch.randn(4, 197, 768, device=device, dtype=dtype, requires_grad=True)", "model(x)")
    cases.append(l2_case("bench_v1.0.0/L2/transformer/encoder_layer_preln", setup, body, ["torch.nn.LayerNorm", "torch.nn.MultiheadAttention", "torch.nn.SiLU"], rationale="Transformer encoder variant with SiLU FFN."))
    setup, body = generic_l2_model_setup(encoder_setup, "AttentionBlock(128, 4, 512, activation='gelu')", "x = torch.randn(4, 50, 128, device=device, dtype=dtype, requires_grad=True)", "model(x)")
    cases.append(l2_case("bench_v1.0.0/L2/transformer/encoder_layer_narrow", setup, body, ["torch.nn.LayerNorm", "torch.nn.MultiheadAttention", "torch.nn.GELU"], rationale="Narrow Transformer encoder."))
    setup, body = generic_l2_model_setup(ffn_setup, "FFN(256, 1024, activation='gelu')", "x = torch.randn(4, 197, 256, device=device, dtype=dtype, requires_grad=True)", "model(x)")
    cases.append(l2_case("bench_v1.0.0/L2/transformer/ffn_gelu", setup, body, ["torch.nn.Linear", "torch.nn.GELU"], rationale="Transformer FFN with GELU."))
    setup, body = generic_l2_model_setup(ffn_setup, "FFN(512, 2048, activation='relu')", "x = torch.randn(4, 100, 512, device=device, dtype=dtype, requires_grad=True)", "model(x)")
    cases.append(l2_case("bench_v1.0.0/L2/transformer/ffn_relu", setup, body, ["torch.nn.Linear", "torch.nn.ReLU"], rationale="Transformer FFN with ReLU."))
    setup, body = generic_l2_model_setup(ffn_setup, "SwiGLUFFN(256, 1024)", "x = torch.randn(4, 197, 256, device=device, dtype=dtype, requires_grad=True)", "model(x)")
    cases.append(l2_case("bench_v1.0.0/L2/transformer/ffn_swiglu", setup, body, ["torch.nn.Linear", "torch.nn.SiLU"], rationale="SwiGLU FFN subgraph."))
    setup, body = generic_l2_model_setup(cnn_head_setup, "CNNHead(512, 1000)", "x = torch.randn(4, 512, 7, 7, device=device, dtype=dtype, requires_grad=True)", "model(x)")
    cases.append(l2_case("bench_v1.0.0/L2/head/cnn_head_1000cls", setup, body, ["torch.nn.AdaptiveAvgPool2d", "torch.nn.Linear"], rationale="CNN head pool-flatten-linear."))
    setup, body = generic_l2_model_setup(cnn_head_setup, "CNNHead(2048, 1000, dropout=0.5)", "x = torch.randn(4, 2048, 7, 7, device=device, dtype=dtype, requires_grad=True)", "model(x)")
    cases.append(l2_case("bench_v1.0.0/L2/head/cnn_head_dropout", setup, body, ["torch.nn.AdaptiveAvgPool2d", "torch.nn.Dropout", "torch.nn.Linear"], rationale="CNN head with dropout."))
    setup, body = generic_l2_model_setup(cnn_head_setup, "CNNHead(512, 100, dropout=0.3, hidden=256)", "x = torch.randn(4, 512, 7, 7, device=device, dtype=dtype, requires_grad=True)", "model(x)")
    cases.append(l2_case("bench_v1.0.0/L2/head/cnn_head_2layer", setup, body, ["torch.nn.AdaptiveAvgPool2d", "torch.nn.Linear", "torch.nn.ReLU"], rationale="Two-layer CNN head."))
    cases.append(
        l2_case(
            "bench_v1.0.0/L2/seq/lstm_2layer",
            "",
            """
device = select_device()
dtype = precision_dtype("FP32", device)
model = nn.LSTM(input_size=256, hidden_size=512, num_layers=2, batch_first=True).to(device=device, dtype=dtype)
x = torch.randn(4, 50, 256, device=device, dtype=dtype, requires_grad=True)
out, (h, c) = model(x)
loss = out.float().mean()
loss.backward()
globals()["ACTIVATIONS"] = {"output_mean": tensor_mean(out), "hidden_mean": tensor_mean(h)}
globals()["GRADIENTS"] = {"input_grad_norm": grad_norm(x.grad), "param_grad_norm": grad_norm(first_grad(model))}
return out
""",
            ["torch.nn.LSTM"],
            rationale="Two-layer LSTM sequence block.",
        )
    )
    cases.append(
        l2_case(
            "bench_v1.0.0/L2/seq/gru_bidirectional",
            "",
            """
device = select_device()
dtype = precision_dtype("FP32", device)
model = nn.GRU(input_size=128, hidden_size=256, num_layers=1, bidirectional=True, batch_first=True).to(device=device, dtype=dtype)
x = torch.randn(4, 30, 128, device=device, dtype=dtype, requires_grad=True)
out, hidden = model(x)
loss = out.float().mean()
loss.backward()
globals()["ACTIVATIONS"] = {"output_mean": tensor_mean(out), "hidden_mean": tensor_mean(hidden)}
globals()["GRADIENTS"] = {"input_grad_norm": grad_norm(x.grad), "param_grad_norm": grad_norm(first_grad(model))}
return out
""",
            ["torch.nn.GRU"],
            rationale="Bidirectional GRU sequence block.",
        )
    )
    cases.append(
        l2_case(
            "bench_v1.0.0/L2/seq/lstm_cell_loop",
            "",
            """
device = select_device()
dtype = precision_dtype("FP32", device)
cell = nn.LSTMCell(128, 128).to(device=device, dtype=dtype)
x = torch.randn(4, 20, 128, device=device, dtype=dtype, requires_grad=True)
h = torch.zeros(4, 128, device=device, dtype=dtype)
c = torch.zeros(4, 128, device=device, dtype=dtype)
outs = []
for t in range(x.shape[1]):
    h, c = cell(x[:, t, :], (h, c))
    outs.append(h)
out = torch.stack(outs, dim=1)
loss = out.float().mean()
loss.backward()
globals()["ACTIVATIONS"] = {"output_mean": tensor_mean(out), "hidden_mean": tensor_mean(h)}
globals()["GRADIENTS"] = {"input_grad_norm": grad_norm(x.grad), "param_grad_norm": grad_norm(first_grad(cell))}
return out
""",
            ["torch.nn.LSTMCell"],
            rationale="Manual LSTMCell loop.",
        )
    )
    for name, expr, ops, rationale in [
        ("cross_entropy", "F.cross_entropy(logits, target)", ["torch.nn.functional.cross_entropy"], "Cross-entropy loss subgraph."),
        ("mse", "F.mse_loss(pred, target)", ["torch.nn.functional.mse_loss"], "MSE loss subgraph."),
        ("bce_with_logits", "F.binary_cross_entropy_with_logits(logits, target)", ["torch.nn.functional.binary_cross_entropy_with_logits"], "BCE-with-logits subgraph."),
    ]:
        if name == "cross_entropy":
            body = """
device = select_device()
dtype = precision_dtype("FP32", device)
logits = torch.randn(4, 1000, device=device, dtype=dtype, requires_grad=True)
target = torch.randint(0, 1000, (4,), device=device)
out = F.cross_entropy(logits, target)
out.backward()
globals()["ACTIVATIONS"] = {"loss": float(out.detach().cpu())}
globals()["GRADIENTS"] = {"logit_grad_norm": grad_norm(logits.grad)}
return out
"""
        elif name == "mse":
            body = """
device = select_device()
dtype = precision_dtype("FP32", device)
pred = torch.randn(4, 256, device=device, dtype=dtype, requires_grad=True)
target = torch.randn(4, 256, device=device, dtype=dtype)
out = F.mse_loss(pred, target)
out.backward()
globals()["ACTIVATIONS"] = {"loss": float(out.detach().cpu())}
globals()["GRADIENTS"] = {"pred_grad_norm": grad_norm(pred.grad)}
return out
"""
        else:
            body = """
device = select_device()
dtype = precision_dtype("FP32", device)
logits = torch.randn(4, 1, device=device, dtype=dtype, requires_grad=True)
target = torch.randint(0, 2, (4, 1), device=device).float()
out = F.binary_cross_entropy_with_logits(logits, target)
out.backward()
globals()["ACTIVATIONS"] = {"loss": float(out.detach().cpu())}
globals()["GRADIENTS"] = {"logit_grad_norm": grad_norm(logits.grad)}
return out
"""
        cases.append(l2_case(f"bench_v1.0.0/L2/loss/{name}", "", body, ops, rationale=rationale))
    cases.append(l2_case("bench_v1.0.0/L2/trap/dropout_keep_vs_p", "", """
device = select_device()
dtype = precision_dtype("FP32", device)
model = nn.Sequential(nn.Dropout(p=0.3), nn.Linear(512, 256)).to(device=device, dtype=dtype)
model.train()
x = torch.ones(8, 512, device=device, dtype=dtype, requires_grad=True)
out = model(x)
loss = out.float().mean()
loss.backward()
globals()["ACTIVATIONS"] = {"output_mean": tensor_mean(out)}
globals()["GRADIENTS"] = {"input_grad_norm": grad_norm(x.grad), "param_grad_norm": grad_norm(first_grad(model))}
return out
""", ["torch.nn.Dropout", "torch.nn.Linear"], rationale="Explicit dropout keep-vs-p semantic trap.", failure_mode="dropout_semantics"))
    cases.append(l2_case("bench_v1.0.0/L2/trap/bn_momentum_semantic", bnm_setup, """
device = select_device()
dtype = precision_dtype("FP32", device)
model = BNMomentumTest().to(device=device, dtype=dtype)
model.train()
last = None
for step in range(3):
    x = torch.randn(4, 16, 32, 32, device=device, dtype=dtype, requires_grad=True)
    last = model(x + step * 0.1)
loss = last.float().mean()
loss.backward()
globals()["ACTIVATIONS"] = {"output_mean": tensor_mean(last), "running_mean_norm": grad_norm(model.bn.running_mean)}
globals()["GRADIENTS"] = {"param_grad_norm": grad_norm(first_grad(model))}
return last
""", ["torch.nn.BatchNorm2d", "torch.nn.Conv2d"], rationale="BatchNorm momentum semantic trap.", failure_mode="bn_momentum_semantics"))
    cases.append(l2_case("bench_v1.0.0/L2/trap/gradient_clip_fp16", grad_clip_setup, """
device = select_device()
precision = "AMP_FP16"
dtype = precision_dtype(precision, device)
model = GradClipModel().to(device=device, dtype=dtype)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
x = torch.randn(8, 256, device=device, dtype=dtype, requires_grad=True)
with maybe_autocast(device, precision):
    out = model(x)
    loss = out.float().mean()
loss.backward()
pre_clip = grad_norm(first_grad(model))
nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
post_clip = grad_norm(first_grad(model))
optimizer.step()
globals()["ACTIVATIONS"] = {"loss": float(loss.detach().cpu())}
globals()["GRADIENTS"] = {"pre_clip": pre_clip, "post_clip": post_clip}
return out
""", ["torch.nn.Linear", "torch.nn.ReLU", "torch.nn.utils.clip_grad_norm_"], rationale="Gradient clip plus AMP-style path.", failure_mode="gradient_clip_amp"))
    cases.append(l2_case("bench_v1.0.0/L2/backward/rmsnorm_grad", rmsnorm_setup, """
device = select_device()
dtype = precision_dtype("FP32", device)
model = nn.RMSNorm(128).to(device=device, dtype=dtype) if hasattr(nn, "RMSNorm") else RMSNormManual(128).to(device=device, dtype=dtype)
x = torch.randn(4, 128, device=device, dtype=dtype, requires_grad=True)
out = model(x)
loss = out.float().pow(2).mean()
loss.backward()
globals()["ACTIVATIONS"] = {"output_mean": tensor_mean(out)}
globals()["GRADIENTS"] = {"input_grad_norm": grad_norm(x.grad), "param_grad_norm": grad_norm(first_grad(model))}
return out
""", ["torch.nn.RMSNorm"], rationale="RMSNorm forward/backward path.", failure_mode="rmsnorm_gradient"))
    cases.append(l2_case("bench_v1.0.0/L2/backward/grad_chain_pow_mean", "", """
device = select_device()
dtype = precision_dtype("FP32", device)
x = torch.randn(8, 16, device=device, dtype=dtype, requires_grad=True)
out = x.pow(2).mean()
out.backward()
analytic = 2 * x.detach() / x.numel()
globals()["ACTIVATIONS"] = {"loss": float(out.detach().cpu())}
globals()["GRADIENTS"] = {"grad_norm": grad_norm(x.grad), "analytic_error": float((x.grad - analytic).abs().max().cpu())}
return out
""", ["Tensor.pow", "torch.mean"], rationale="Elementary gradient chain from ascend-torch4ms.", failure_mode="autograd_chain"))
    cases.append(l2_case("bench_v1.0.0/L2/backward/grad_chain_norm_full", "", """
device = select_device()
dtype = precision_dtype("FP32", device)
x = torch.randn(4, 64, device=device, dtype=dtype, requires_grad=True)
scale = torch.nn.Parameter(torch.ones(64, device=device, dtype=dtype))
norm = (x.pow(2).mean(dim=-1, keepdim=True) + 1e-6).sqrt()
out = (x / norm) * scale
loss = out.float().pow(2).mean()
loss.backward()
globals()["ACTIVATIONS"] = {"loss": float(loss.detach().cpu())}
globals()["GRADIENTS"] = {"input_grad_norm": grad_norm(x.grad), "scale_grad_norm": grad_norm(scale.grad)}
return out
""", ["Tensor.pow", "torch.mean", "torch.sqrt"], rationale="RMSNorm-like full gradient chain.", failure_mode="autograd_chain"))
    cases.append(l2_case("bench_v1.0.0/L2/backward/frozen_param", "", """
device = select_device()
dtype = precision_dtype("FP32", device)
class FrozenParamModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 5)
        self.fc2.weight.requires_grad = False
    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))
model = FrozenParamModel().to(device=device, dtype=dtype)
x = torch.randn(8, 10, device=device, dtype=dtype, requires_grad=True)
out = model(x)
loss = out.float().mean()
loss.backward()
globals()["ACTIVATIONS"] = {"output_mean": tensor_mean(out)}
globals()["GRADIENTS"] = {"fc1_grad": grad_norm(model.fc1.weight.grad), "fc2_grad_is_none": model.fc2.weight.grad is None}
return out
""", ["torch.nn.Linear", "torch.nn.functional.relu"], rationale="Frozen parameter must stay grad-free.", failure_mode="frozen_param_grad"))
    cases.append(l2_case("bench_v1.0.0/L2/backward/no_grad_context", "", """
device = select_device()
dtype = precision_dtype("FP32", device)
model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 8)).to(device=device, dtype=dtype)
x = torch.randn(4, 16, device=device, dtype=dtype)
with torch.no_grad():
    out = model(x)
grad_flags = [param.grad is None for param in model.parameters()]
globals()["ACTIVATIONS"] = {"output_mean": tensor_mean(out)}
globals()["GRADIENTS"] = {"all_none": all(grad_flags)}
return out
""", ["torch.no_grad", "torch.nn.Linear"], rationale="torch.no_grad semantics.", failure_mode="no_grad_context"))
    cases.append(l2_case("bench_v1.0.0/L2/backward/embedding_grad", "", """
device = select_device()
dtype = precision_dtype("FP32", device)
emb = nn.Embedding(1000, 128).to(device=device)
idx = torch.randint(0, 1000, (8, 16), device=device)
out = emb(idx).to(dtype)
loss = out.float().mean()
loss.backward()
globals()["ACTIVATIONS"] = {"output_mean": tensor_mean(out)}
globals()["GRADIENTS"] = {"embed_grad_norm": grad_norm(emb.weight.grad)}
return out
""", ["torch.nn.Embedding"], rationale="Embedding backward path.", failure_mode="embedding_gradient"))
    setup, body = generic_l2_model_setup(swiglu_setup, "SwiGLUFFN(256, 1024)", "x = torch.randn(4, 197, 256, device=device, dtype=dtype, requires_grad=True)", "model(x)")
    cases.append(l2_case("bench_v1.0.0/L2/transformer/swiglu_ffn", setup, body, ["torch.nn.Linear", "torch.nn.SiLU"], rationale="LLaMA-style SwiGLU FFN."))
    cases.append(l2_case("bench_v1.0.0/L2/complex/rope_full", "", """
device = select_device()
dtype = precision_dtype("FP32", device)
x = torch.randn(2, 32, 64, device=device, dtype=dtype, requires_grad=True)
dim = x.shape[-1]
freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=device, dtype=dtype) / dim))
pos = torch.arange(x.shape[1], device=device, dtype=dtype)
theta = pos[:, None] * freq[None, :]
cos = theta.cos()[None, :, :]
sin = theta.sin()[None, :, :]
even = x[..., 0::2]
odd = x[..., 1::2]
rot_even = even * cos - odd * sin
rot_odd = even * sin + odd * cos
out = torch.empty_like(x)
out[..., 0::2] = rot_even
out[..., 1::2] = rot_odd
loss = out.float().mean()
loss.backward()
globals()["ACTIVATIONS"] = {"output_mean": tensor_mean(out)}
globals()["GRADIENTS"] = {"input_grad_norm": grad_norm(x.grad)}
return out
""", ["torch.cos", "torch.sin"], rationale="Real-valued RoPE implementation.", failure_mode="rope_precision"))
    cases.append(l2_case("bench_v1.0.0/L2/complex/rope_complex", "", """
device = select_device()
x = torch.randn(2, 32, 64, device=device, dtype=torch.float32, requires_grad=True)
theta = torch.randn(32, 32, device=device)
z = torch.polar(torch.ones_like(theta), theta)
base = torch.view_as_complex(x.reshape(2, 32, 32, 2).contiguous())
out = torch.view_as_real(base * z.unsqueeze(0)).reshape_as(x)
loss = out.float().mean()
loss.backward()
globals()["ACTIVATIONS"] = {"output_mean": tensor_mean(out)}
globals()["GRADIENTS"] = {"input_grad_norm": grad_norm(x.grad)}
return out
""", ["torch.polar", "torch.view_as_complex", "torch.view_as_real"], rationale="Complex-valued RoPE implementation.", failure_mode="complex_unsupported"))
    setup, body = generic_l2_model_setup(residual_setup, "ResidualBlock(64, 64, groups=4)", "x = torch.randn(4, 64, 32, 32, device=device, dtype=dtype, requires_grad=True)", "model(x)")
    cases.append(l2_case("bench_v1.0.0/L2/cnn/residual_block_grouped", setup, body, ["torch.nn.Conv2d", "torch.nn.BatchNorm2d"], rationale="Grouped residual block extension from operator coverage."))
    setup, body = generic_l2_model_setup(attention_setup, "SelfAttention(256, 8, use_native_sdp=False, use_mask=True)", "x = torch.randn(4, 128, 256, device=device, dtype=dtype, requires_grad=True)", "model(x)")
    cases.append(l2_case("bench_v1.0.0/L2/transformer/attention_with_mask", setup, body, ["torch.nn.Linear", "Tensor.masked_fill"], rationale="Masked self-attention extension."))
    cases.append(l2_case("bench_v1.0.0/L2/seq/embedding_lstm_classifier", "", """
device = select_device()
dtype = precision_dtype("FP32", device)
emb = nn.Embedding(512, 64).to(device=device)
lstm = nn.LSTM(64, 128, batch_first=True).to(device=device, dtype=dtype)
head = nn.Linear(128, 10).to(device=device, dtype=dtype)
tokens = torch.randint(0, 512, (4, 24), device=device)
x = emb(tokens).to(dtype)
out, _ = lstm(x)
logits = head(out[:, -1])
loss = F.cross_entropy(logits, torch.randint(0, 10, (4,), device=device))
loss.backward()
globals()["ACTIVATIONS"] = {"logit_mean": tensor_mean(logits)}
globals()["GRADIENTS"] = {"embed_grad_norm": grad_norm(emb.weight.grad), "head_grad_norm": grad_norm(head.weight.grad)}
return logits
""", ["torch.nn.Embedding", "torch.nn.LSTM", "torch.nn.Linear"], rationale="Embedding-plus-LSTM classifier extension."))
    return cases


def build_l3_cases() -> list[CaseSpec]:
    def model_case(identifier: str, model_setup: str, model_ctor: str, input_expr: str, expected_ops: list[str], *, precision: str = "FP32", rationale: str, failure_mode: str | None = None, extra: str = "") -> CaseSpec:
        run_body = f"""
device = select_device()
precision = "{precision}"
dtype = precision_dtype(precision, device)
model = {model_ctor}.to(device=device, dtype=dtype)
{extra}
x = {input_expr}
out = model(x)
loss = out.float().mean()
loss.backward()
debug = getattr(model, "debug", {{}})
globals()["ACTIVATIONS"] = {{k: tensor_mean(v) for k, v in debug.items()}}
globals()["GRADIENTS"] = {{
    "param_grad_norm": grad_norm(first_grad(model)),
    "input_grad_norm": grad_norm(x.grad),
}}
return out
"""
        return l3_case(identifier, model_setup, run_body, expected_ops, rationale=rationale, failure_mode=failure_mode)

    cases = [
        model_case("bench_v1.0.0/L3/resnet18/fp32_basic", RESNET_BLOCK, "ResNet18()", "torch.randn(1, 3, 224, 224, device=device, dtype=dtype, requires_grad=True)", ["torch.nn.Conv2d", "torch.nn.BatchNorm2d", "torch.nn.ReLU", "torch.nn.MaxPool2d", "torch.nn.AdaptiveAvgPool2d", "torch.nn.Linear"], rationale="Hand-written ResNet-18 FP32."),
        model_case("bench_v1.0.0/L3/resnet18/fp16_amp", RESNET_BLOCK, "ResNet18()", "torch.randn(1, 3, 224, 224, device=device, dtype=dtype, requires_grad=True)", ["torch.nn.Conv2d", "torch.nn.BatchNorm2d", "torch.nn.ReLU", "torch.nn.MaxPool2d", "torch.nn.AdaptiveAvgPool2d", "torch.nn.Linear"], precision="FP16", rationale="ResNet-18 FP16/AMP path."),
        model_case("bench_v1.0.0/L3/resnet18/bn_momentum_01", RESNET_BLOCK, "ResNet18(bn_momentum=0.1)", "torch.randn(1, 3, 224, 224, device=device, dtype=dtype, requires_grad=True)", ["torch.nn.Conv2d", "torch.nn.BatchNorm2d", "torch.nn.ReLU", "torch.nn.MaxPool2d", "torch.nn.AdaptiveAvgPool2d", "torch.nn.Linear"], rationale="ResNet-18 with BatchNorm momentum=0.1.", failure_mode="bn_momentum_semantics"),
        model_case("bench_v1.0.0/L3/resnet18/trap_modelmeta", RESNET_BLOCK, "ResNet18(trap_mode='smr1')", "torch.randn(1, 3, 224, 224, device=device, dtype=dtype, requires_grad=True)", ["torch.nn.Conv2d", "torch.nn.BatchNorm2d", "torch.nn.ReLU", "torch.nn.MaxPool2d", "torch.nn.AdaptiveAvgPool2d", "torch.nn.Linear"], rationale="ResNet-18 with ModelMeta SMR1 trap.", failure_mode="modelmeta_trap"),
        model_case("bench_v1.0.0/L3/vit_tiny/fp32_basic", VIT_BLOCK, "ViTTiny()", "torch.randn(1, 3, 224, 224, device=device, dtype=dtype, requires_grad=True)", ["torch.nn.Conv2d", "torch.nn.LayerNorm", "torch.nn.Linear", "torch.nn.GELU", "torch.nn.Softmax", "torch.nn.Dropout"], rationale="Hand-written ViT-Tiny FP32."),
        model_case("bench_v1.0.0/L3/vit_tiny/fp16", VIT_BLOCK, "ViTTiny()", "torch.randn(1, 3, 224, 224, device=device, dtype=dtype, requires_grad=True)", ["torch.nn.Conv2d", "torch.nn.LayerNorm", "torch.nn.Linear", "torch.nn.GELU", "torch.nn.Softmax", "torch.nn.Dropout"], precision="FP16", rationale="ViT-Tiny FP16 path."),
        model_case("bench_v1.0.0/L3/vit_tiny/gelu_vs_silu", VIT_BLOCK, "ViTTiny(activation='silu')", "torch.randn(1, 3, 224, 224, device=device, dtype=dtype, requires_grad=True)", ["torch.nn.Conv2d", "torch.nn.LayerNorm", "torch.nn.Linear", "torch.nn.SiLU", "torch.nn.Dropout"], rationale="ViT-Tiny with SiLU FFN."),
        model_case("bench_v1.0.0/L3/vit_tiny/trap_modelmeta", VIT_BLOCK, "ViTTiny(trap_mode='smr3')", "torch.randn(1, 3, 224, 224, device=device, dtype=dtype, requires_grad=True)", ["torch.nn.Conv2d", "torch.nn.LayerNorm", "torch.nn.Linear", "torch.nn.GELU"], rationale="ViT-Tiny with ModelMeta SMR3 trap.", failure_mode="modelmeta_trap"),
        model_case("bench_v1.0.0/L3/mobilenetv2/fp32_basic", MOBILENET_BLOCK, "MobileNetV2Small()", "torch.randn(1, 3, 224, 224, device=device, dtype=dtype, requires_grad=True)", ["torch.nn.Conv2d", "torch.nn.BatchNorm2d", "torch.nn.ReLU6", "torch.nn.AdaptiveAvgPool2d"], rationale="Hand-written MobileNetV2 FP32."),
        model_case("bench_v1.0.0/L3/mobilenetv2/fp16", MOBILENET_BLOCK, "MobileNetV2Small()", "torch.randn(1, 3, 224, 224, device=device, dtype=dtype, requires_grad=True)", ["torch.nn.Conv2d", "torch.nn.BatchNorm2d", "torch.nn.ReLU6", "torch.nn.AdaptiveAvgPool2d"], precision="FP16", rationale="MobileNetV2 FP16 path."),
        model_case("bench_v1.0.0/L3/mobilenetv2/narrow", MOBILENET_BLOCK, "MobileNetV2Small(width_mult=0.5)", "torch.randn(1, 3, 224, 224, device=device, dtype=dtype, requires_grad=True)", ["torch.nn.Conv2d", "torch.nn.BatchNorm2d", "torch.nn.ReLU6", "torch.nn.AdaptiveAvgPool2d"], rationale="Narrow MobileNetV2 width multiplier 0.5."),
        model_case("bench_v1.0.0/L3/mobilenetv2/trap_modelmeta", MOBILENET_BLOCK, "MobileNetV2Small(trap_mode='smr4')", "torch.randn(1, 3, 224, 224, device=device, dtype=dtype, requires_grad=True)", ["torch.nn.Conv2d", "torch.nn.BatchNorm2d", "torch.nn.ReLU6", "torch.nn.AdaptiveAvgPool2d"], rationale="MobileNetV2 with ModelMeta SMR4 trap.", failure_mode="modelmeta_trap"),
        model_case("bench_v1.0.0/L3/yolo_backbone/fp32_basic", YOLO_BLOCK, "YOLOv5nBackbone()", "torch.randn(1, 3, 640, 640, device=device, dtype=dtype, requires_grad=True)", ["torch.nn.Conv2d", "torch.nn.BatchNorm2d", "torch.nn.SiLU", "torch.nn.MaxPool2d", "torch.cat"], rationale="Simplified YOLOv5n backbone FP32."),
        model_case("bench_v1.0.0/L3/yolo_backbone/fp16_640", YOLO_BLOCK, "YOLOv5nBackbone()", "torch.randn(1, 3, 640, 640, device=device, dtype=dtype, requires_grad=True)", ["torch.nn.Conv2d", "torch.nn.BatchNorm2d", "torch.nn.SiLU", "torch.nn.MaxPool2d", "torch.cat"], precision="FP16", rationale="YOLOv5n backbone FP16."),
        model_case("bench_v1.0.0/L3/yolo_backbone/nms_deterministic", YOLO_BLOCK, "YOLOv5nBackbone(with_head=True)", "torch.randn(1, 3, 640, 640, device=device, dtype=dtype, requires_grad=True)", ["torch.nn.Conv2d", "torch.nn.BatchNorm2d", "torch.nn.SiLU", "torch.cat"], rationale="YOLOv5n backbone with deterministic NMS head.", extra="pass"),
        model_case("bench_v1.0.0/L3/yolo_backbone/trap", YOLO_BLOCK, "YOLOv5nBackbone(trap_mode='smr2')", "torch.randn(1, 3, 640, 640, device=device, dtype=dtype, requires_grad=True)", ["torch.nn.Conv2d", "torch.nn.BatchNorm2d", "torch.nn.SiLU", "torch.nn.MaxPool2d", "torch.cat"], rationale="YOLOv5n backbone with ModelMeta SMR2 trap.", failure_mode="modelmeta_trap"),
        model_case("bench_v1.0.0/L3/unet/fp32_basic", UNET_BLOCK, "UNetSmall()", "torch.randn(1, 3, 224, 224, device=device, dtype=dtype, requires_grad=True)", ["torch.nn.Conv2d", "torch.nn.ReLU", "torch.nn.MaxPool2d", "torch.nn.ConvTranspose2d", "torch.cat"], rationale="Hand-written UNet 2-level FP32."),
        model_case("bench_v1.0.0/L3/unet/fp32_with_bn", UNET_BLOCK, "UNetSmall(with_bn=True)", "torch.randn(1, 3, 224, 224, device=device, dtype=dtype, requires_grad=True)", ["torch.nn.Conv2d", "torch.nn.BatchNorm2d", "torch.nn.ReLU", "torch.nn.MaxPool2d", "torch.nn.ConvTranspose2d", "torch.cat"], rationale="UNet with BatchNorm."),
        model_case("bench_v1.0.0/L3/unet/3ch_output", UNET_BLOCK, "UNetSmall(out_ch=3, with_bn=True)", "torch.randn(1, 3, 224, 224, device=device, dtype=dtype, requires_grad=True)", ["torch.nn.Conv2d", "torch.nn.BatchNorm2d", "torch.nn.ReLU", "torch.nn.MaxPool2d", "torch.nn.ConvTranspose2d", "torch.cat"], rationale="UNet RGB output variant."),
        model_case("bench_v1.0.0/L3/unet/deep", UNET_BLOCK, "UNetSmall(depth=3, with_bn=True)", "torch.randn(1, 3, 224, 224, device=device, dtype=dtype, requires_grad=True)", ["torch.nn.Conv2d", "torch.nn.BatchNorm2d", "torch.nn.ReLU", "torch.nn.MaxPool2d", "torch.nn.ConvTranspose2d", "torch.cat"], rationale="Deep UNet 3-level variant."),
        model_case("bench_v1.0.0/L3/resnet18/frozen_bn_eval", RESNET_BLOCK, "ResNet18()", "torch.randn(1, 3, 224, 224, device=device, dtype=dtype, requires_grad=True)", ["torch.nn.Conv2d", "torch.nn.BatchNorm2d", "torch.nn.ReLU"], rationale="ResNet-18 with frozen BatchNorm eval semantics.", failure_mode="frozen_bn", extra="freeze_batch_norm(model)"),
        model_case("bench_v1.0.0/L3/vit_tiny/rope_attention", VIT_BLOCK, "ViTTiny(use_rope=True)", "torch.randn(1, 3, 224, 224, device=device, dtype=dtype, requires_grad=True)", ["torch.nn.Conv2d", "torch.nn.LayerNorm", "torch.nn.Linear", "torch.cos", "torch.sin"], rationale="ViT-Tiny with RoPE-style rotary embedding.", failure_mode="rope_precision"),
        model_case("bench_v1.0.0/L3/mobilenetv2/wide", MOBILENET_BLOCK, "MobileNetV2Small(width_mult=1.25)", "torch.randn(1, 3, 224, 224, device=device, dtype=dtype, requires_grad=True)", ["torch.nn.Conv2d", "torch.nn.BatchNorm2d", "torch.nn.ReLU6"], rationale="Wider MobileNetV2 extension."),
        model_case("bench_v1.0.0/L3/yolo_backbone/focus_slicing", YOLO_BLOCK, "YOLOv5nBackbone()", "torch.randn(1, 3, 320, 320, device=device, dtype=dtype, requires_grad=True)", ["torch.nn.Conv2d", "torch.cat"], rationale="YOLO Focus slicing path at smaller resolution."),
        model_case("bench_v1.0.0/L3/unet/bilinear_upsample", UNET_BLOCK, "UNetSmall(with_bn=True, bilinear=True)", "torch.randn(1, 3, 224, 224, device=device, dtype=dtype, requires_grad=True)", ["torch.nn.Conv2d", "torch.nn.BatchNorm2d", "torch.nn.ReLU", "torch.nn.functional.interpolate", "torch.cat"], rationale="UNet with bilinear upsampling extension."),
    ]
    return cases


def build_training_case_code(model_setup: str, model_ctor: str, *, family: str, expected_ops: list[str], identifier: str, optimizer_name: str, precision: str, steps: int = 2, epochs: int = 0, extra_training: str = "", scheduler_name: str | None = None, rationale: str, training_config: dict[str, Any], failure_mode: str | None = None) -> CaseSpec:
    if family == "unet":
        batch_code = """
x = torch.randn(batch_size, 3, 128, 128, device=device, dtype=dtype)
y = torch.randn(batch_size, out_ch, 128, 128, device=device, dtype=dtype)
criterion = nn.MSELoss()
"""
        target_expr = "criterion(out, y)"
    elif family == "yolo":
        batch_code = """
x = torch.randn(batch_size, 3, 320, 320, device=device, dtype=dtype)
y = torch.randn(batch_size, 24, 20, 20, device=device, dtype=dtype)
criterion = nn.MSELoss()
"""
        target_expr = "criterion(out, y)"
    elif family == "sequence":
        batch_code = """
x = torch.randn(batch_size, 50, 64, device=device, dtype=dtype)
y = torch.randint(0, num_classes, (batch_size,), device=device)
criterion = nn.CrossEntropyLoss()
"""
        target_expr = "criterion(out, y)"
    else:
        batch_code = """
x = torch.randn(batch_size, 3, 224, 224, device=device, dtype=dtype)
y = torch.randint(0, num_classes, (batch_size,), device=device)
criterion = nn.CrossEntropyLoss()
"""
        target_expr = "criterion(out, y)"
    opt_line = {
        "SGD": "optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)",
        "Adam": "optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=betas)",
        "AdamW": "optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)",
    }[optimizer_name]
    scheduler_block = {
        None: "scheduler = None",
        "CosineAnnealingLR": f"scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max={max(steps, epochs or 1)})",
        "OneCycleLR": f"scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max(lr, 1e-3), total_steps={(steps or (epochs * 5))})",
    }[scheduler_name]
    batch_block = indent(batch_code, "    ")
    extra_block = indent(extra_training.strip() or "pass", "    ")
    run_body = f"""
device = select_device()
precision = "{precision}"
dtype = precision_dtype(precision, device)
num_classes = 10
batch_size = 4
out_ch = 1 if "{family}" == "unet" else 24
model = {model_ctor}.to(device=device, dtype=dtype)
lr = {training_config.get("lr", 1e-3)}
momentum = {training_config.get("momentum", 0.9)}
weight_decay = {training_config.get("weight_decay", 0.0)}
betas = tuple({training_config.get("betas", [0.9, 0.999])})
{opt_line}
{scheduler_block}
scaler = torch.cuda.amp.GradScaler(enabled=("AMP" in precision and device.type != "cpu"))
loss_curve = []
grad_curve = []
nms_orders = []
for step in range({steps or max(epochs * 5, 1)}):
    optimizer.zero_grad(set_to_none=True)
{batch_block}
    with maybe_autocast(device, precision):
        out = model(x)
        loss = {target_expr}
    if scaler.is_enabled():
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
    else:
        loss.backward()
{extra_block}
    if scaler.is_enabled():
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    if scheduler is not None:
        scheduler.step()
    loss_curve.append(float(loss.detach().cpu()))
    grad_curve.append(grad_norm(first_grad(model)))
    if "{family}" == "yolo" and step % max({training_config.get("nms_eval_interval", 10)}, 1) == 0:
        with torch.no_grad():
            head = out.detach().float().reshape(out.shape[0], 6, -1)
            scores = head[:, 4, :].reshape(-1)
            coords = torch.sigmoid(head[:, :4, :].transpose(1, 2).reshape(-1, 4))
            boxes = torch.stack([
                coords[:, 0],
                coords[:, 1],
                coords[:, 0] + coords[:, 2],
                coords[:, 1] + coords[:, 3],
            ], dim=1)
            keep = nms_torch(boxes, scores, iou_threshold=0.5)
            nms_orders.append([int(i) for i in keep[:10].cpu()])
globals()["TASK_METRICS"] = {{
    "loss_curve": loss_curve,
    "grad_curve": grad_curve,
    "final_loss": loss_curve[-1],
    "final_grad_norm": grad_curve[-1],
    "accuracy_proxy": float(max(0.0, 1.0 / (1.0 + loss_curve[-1]))),
    "scheduler": "{scheduler_name or 'none'}",
    "nms_orders": nms_orders,
}}
return out
"""
    return l4_case(identifier, model_setup, run_body, expected_ops, training_config, rationale=rationale, failure_mode=failure_mode)


def build_l4_cases() -> list[CaseSpec]:
    families = {
        "resnet18": (RESNET_BLOCK, "ResNet18(num_classes=num_classes)", ["torch.nn.Conv2d", "torch.nn.BatchNorm2d", "torch.nn.ReLU", "torch.optim.SGD"]),
        "vit_tiny": (VIT_BLOCK, "ViTTiny(num_classes=num_classes)", ["torch.nn.Conv2d", "torch.nn.LayerNorm", "torch.nn.Linear", "torch.optim.AdamW"]),
        "mobilenet": (MOBILENET_BLOCK, "MobileNetV2Small(num_classes=num_classes)", ["torch.nn.Conv2d", "torch.nn.BatchNorm2d", "torch.nn.ReLU6", "torch.optim.Adam"]),
        "unet": (UNET_BLOCK, "UNetSmall(in_ch=3, out_ch=1, with_bn=True)", ["torch.nn.Conv2d", "torch.nn.ConvTranspose2d", "torch.optim.Adam"]),
        "yolo": (YOLO_BLOCK, "YOLOv5nBackbone(with_head=True)", ["torch.nn.Conv2d", "torch.nn.SiLU", "torch.optim.SGD"]),
        "sequence": (LSTM_TRAIN_BLOCK, "LSTMClassifier(num_classes=num_classes)", ["torch.nn.LSTM", "torch.nn.Linear", "torch.optim.Adam"]),
    }
    short_specs = [
        ("resnet18", "SGD", "FP32", {"lr": 0.01, "momentum": 0.9}, "bench_v1.0.0/L4/short/resnet18_sgd_2step"),
        ("vit_tiny", "AdamW", "FP32", {"lr": 1e-4, "weight_decay": 0.05}, "bench_v1.0.0/L4/short/vit_adamw_2step"),
        ("mobilenet", "Adam", "FP32", {"lr": 1e-3, "betas": [0.9, 0.999]}, "bench_v1.0.0/L4/short/mobilenet_adam_2step"),
        ("resnet18", "SGD", "AMP_FP16", {"lr": 0.01, "momentum": 0.9}, "bench_v1.0.0/L4/short/resnet18_sgd_amp_2step"),
        ("unet", "Adam", "FP32", {"lr": 1e-3}, "bench_v1.0.0/L4/short/unet_adam_2step"),
        ("yolo", "SGD", "FP32", {"lr": 0.01, "momentum": 0.937}, "bench_v1.0.0/L4/short/yolo_sgd_2step"),
        ("vit_tiny", "SGD", "FP32", {"lr": 0.01, "momentum": 0.9}, "bench_v1.0.0/L4/short/vit_sgd_2step"),
        ("resnet18", "AdamW", "AMP_FP16", {"lr": 1e-4, "weight_decay": 0.05}, "bench_v1.0.0/L4/short/resnet18_adamw_amp_2step"),
        ("mobilenet", "SGD", "AMP_FP16", {"lr": 0.01, "momentum": 0.9}, "bench_v1.0.0/L4/short/mobilenet_sgd_amp_2step"),
        ("vit_tiny", "Adam", "FP32", {"lr": 1e-3, "betas": [0.9, 0.999]}, "bench_v1.0.0/L4/short/vit_tiny_adam_2step"),
    ]
    for family in families:
        for optimizer in ("SGD", "Adam", "AdamW"):
            for precision in ("FP32", "AMP_FP16"):
                ident = f"bench_v1.0.0/L4/short/{family}_{optimizer.lower()}_{precision.lower()}_2step"
                if any(spec[-1] == ident for spec in short_specs):
                    continue
                params: dict[str, Any] = {"lr": 1e-3}
                if optimizer == "SGD":
                    params = {"lr": 0.01, "momentum": 0.9 if family != "yolo" else 0.937}
                elif optimizer == "Adam":
                    params = {"lr": 1e-3, "betas": [0.9, 0.999]}
                elif optimizer == "AdamW":
                    params = {"lr": 1e-4, "weight_decay": 0.05}
                short_specs.append((family, optimizer, precision, params, ident))
    short_specs = short_specs[:30]
    cases: list[CaseSpec] = []
    for family, optimizer_name, precision, params, identifier in short_specs:
        setup, ctor, ops = families[family]
        cfg = {"optimizer": optimizer_name, "steps": 2, "precision": precision, **params}
        cases.append(
            build_training_case_code(
                setup,
                ctor,
                family=family,
                expected_ops=ops,
                identifier=identifier,
                optimizer_name=optimizer_name,
                precision=precision,
                steps=2,
                rationale=f"Two-step training loop for {family} with {optimizer_name}/{precision}.",
                training_config=cfg,
                failure_mode="training_short",
            )
        )
    medium_specs = [
        ("bench_v1.0.0/L4/medium/resnet18_bn_amp_50step", "resnet18", "AdamW", "AMP_FP16", {"lr": 1e-4, "weight_decay": 0.05, "steps": 50}, "", "CosineAnnealingLR"),
        ("bench_v1.0.0/L4/medium/vit_gradient_clip_50step", "vit_tiny", "AdamW", "FP32", {"lr": 1e-4, "weight_decay": 0.05, "steps": 50, "clip": 1.0}, "nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)", "OneCycleLR"),
        ("bench_v1.0.0/L4/medium/lstm_gradient_50step", "sequence", "Adam", "FP32", {"lr": 1e-3, "steps": 50, "rnn_type": "LSTM", "seq_len": 50}, "", None),
        ("bench_v1.0.0/L4/medium/yolo_amp_50step", "yolo", "SGD", "AMP_FP16", {"lr": 0.01, "momentum": 0.937, "steps": 50, "nms_eval_interval": 10}, "", None),
        ("bench_v1.0.0/L4/medium/resnet18_frozen_bn_50step", "resnet18", "Adam", "FP32", {"lr": 1e-3, "steps": 50, "frozen_bn": True}, "freeze_batch_norm(model)", None),
        ("bench_v1.0.0/L4/medium/mobilenet_adamw_cosine_50step", "mobilenet", "AdamW", "FP32", {"lr": 1e-4, "weight_decay": 0.02, "steps": 50}, "", "CosineAnnealingLR"),
        ("bench_v1.0.0/L4/medium/unet_adam_50step", "unet", "Adam", "FP32", {"lr": 1e-3, "steps": 50}, "", None),
        ("bench_v1.0.0/L4/medium/vit_amp_50step", "vit_tiny", "AdamW", "AMP_FP16", {"lr": 1e-4, "weight_decay": 0.05, "steps": 50}, "", "OneCycleLR"),
    ]
    for identifier, family, optimizer_name, precision, cfg, extra, scheduler in medium_specs:
        setup, ctor, ops = families[family]
        cases.append(
            build_training_case_code(
                setup,
                ctor,
                family=family,
                expected_ops=ops,
                identifier=identifier,
                optimizer_name=optimizer_name,
                precision=precision,
                steps=50,
                extra_training=extra,
                scheduler_name=scheduler,
                rationale=f"50-step medium training loop for {family}.",
                training_config=cfg,
                failure_mode="training_medium",
            )
        )
    long_specs = [
        ("bench_v1.0.0/L4/long/resnet18_adam_cosine_30epoch", "resnet18", "Adam", "FP32", {"lr": 1e-3, "epochs": 30, "snapshots": 10}, "", "CosineAnnealingLR"),
        ("bench_v1.0.0/L4/long/vit_adamw_warmup_50epoch", "vit_tiny", "AdamW", "FP32", {"lr": 1e-3, "weight_decay": 0.05, "epochs": 50, "warmup_pct": 0.1, "snapshots": 10}, "", "OneCycleLR"),
        ("bench_v1.0.0/L4/long/mobilenet_adamw_20epoch", "mobilenet", "AdamW", "FP32", {"lr": 5e-4, "weight_decay": 0.02, "epochs": 20, "snapshots": 5}, "", "CosineAnnealingLR"),
    ]
    for identifier, family, optimizer_name, precision, cfg, extra, scheduler in long_specs:
        setup, ctor, ops = families[family]
        cases.append(
            build_training_case_code(
                setup,
                ctor,
                family=family,
                expected_ops=ops,
                identifier=identifier,
                optimizer_name=optimizer_name,
                precision=precision,
                steps=0,
                epochs=cfg["epochs"],
                extra_training=extra,
                scheduler_name=scheduler,
                rationale=f"Long training loop for {family} with {cfg['epochs']} epochs.",
                training_config=cfg,
                failure_mode="training_long",
            )
        )
    return cases


def case_path_for(identifier: str) -> Path:
    rel = identifier.removeprefix("bench_v1.0.0/")
    return CASES_ROOT / f"{rel}.json"


def write_case(case: CaseSpec) -> Path:
    path = case_path_for(case.id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(case.to_json(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def write_suite(name: str, paths: list[Path]) -> Path:
    suite = {
        "suite_id": name,
        "cases": [str(Path("..") / path.relative_to(BENCH_ROOT)) for path in paths],
        "adapters": [ADAPTER_REF],
    }
    out = SUITES_ROOT / f"{name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(suite, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out


def write_readme(total: int, by_level: dict[str, int]) -> None:
    content = f"""# Benchmark Library v1.0.0

This directory materializes the static TorchBridgeBench benchmark catalog
described in `ClaudeCodePluginDesign.md` Appendix A.

- Total cases: `{total}`
- L1 cases: `{by_level['L1']}`
- L2 cases: `{by_level['L2']}`
- L3 cases: `{by_level['L3']}`
- L4 cases: `{by_level['L4']}`

Generation command:

```bash
python scripts/generate_benchmark_library.py
```

The generator is build-time only. Runtime evaluation still consumes the static
JSON cases under `cases/`.

Bundled suites:

- `suites/dev_noop.json`: fast routine validation.
- `suites/smoke_noop.json`: cross-level benchmark smoke including L3/L4.
- `suites/all_noop.json`: full generated matrix.
"""
    (BENCH_ROOT / "README.md").write_text(content, encoding="utf-8")


def main() -> int:
    if BENCH_ROOT.exists():
        shutil.rmtree(BENCH_ROOT)
    cases = build_l1_cases() + build_l2_cases() + build_l3_cases() + build_l4_cases()
    paths: list[Path] = []
    by_level: defaultdict[str, list[Path]] = defaultdict(list)
    for case in cases:
        path = write_case(case)
        paths.append(path)
        by_level[case.level].append(path)
    write_suite("all_noop", paths)
    for level in ("L1", "L2", "L3", "L4"):
        write_suite(f"{level.lower()}_noop", by_level[level])
    dev = [
        case_path_for("bench_v1.0.0/L1/tensor/reshape"),
        case_path_for("bench_v1.0.0/L1/attn/matmul"),
    ]
    write_suite("dev_noop", dev)
    smoke = [
        case_path_for("bench_v1.0.0/L1/conv/conv2d_fp32"),
        case_path_for("bench_v1.0.0/L2/cnn/conv_bn_relu_64ch"),
        case_path_for("bench_v1.0.0/L3/mobilenetv2/narrow"),
        case_path_for("bench_v1.0.0/L4/short/unet_adam_2step"),
    ]
    write_suite("smoke_noop", smoke)
    counts = Counter(case.level for case in cases)
    manifest = {
        "schema_version": "tbbcc.benchmark.v0.1",
        "benchmark_id": "bench_v1.0.0",
        "generator": "scripts/generate_benchmark_library.py",
        "totals": {"total_cases": len(cases), "by_level": dict(counts)},
        "suites": {
            "all": "suites/all_noop.json",
            "dev": "suites/dev_noop.json",
            "L1": "suites/l1_noop.json",
            "L2": "suites/l2_noop.json",
            "L3": "suites/l3_noop.json",
            "L4": "suites/l4_noop.json",
            "smoke": "suites/smoke_noop.json",
        },
    }
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_readme(len(cases), {level: counts[level] for level in ("L1", "L2", "L3", "L4")})
    print(json.dumps(manifest["totals"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
