"""Detect CPU, GPU, NPU and framework runtimes; optional user confirmation."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any


def _run_cmd(cmd: list[str], timeout: float = 15.0) -> tuple[int, str, str]:
    try:
        p = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
        )
        return p.returncode, p.stdout or "", p.stderr or ""
    except FileNotFoundError:
        return -1, "", "command not found"
    except subprocess.TimeoutExpired:
        return -2, "", "timeout"


@dataclass
class HardwareReport:
    """Snapshot of the environment for reproducible reports."""

    hostname: str = ""
    os_name: str = ""
    os_version: str = ""
    python_version: str = ""
    python_executable: str = ""
    cpu_brand: str = ""
    cpu_count_logical: int = 0
    cuda_available: bool | None = None
    cuda_version: str | None = None
    torch_version: str | None = None
    mindspore_version: str | None = None
    jax_version: str | None = None
    nvidia_smi: str | None = None
    npu_smi: str | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = {
            "hostname": self.hostname,
            "os_name": self.os_name,
            "os_version": self.os_version,
            "python_version": self.python_version,
            "python_executable": self.python_executable,
            "cpu_brand": self.cpu_brand,
            "cpu_count_logical": self.cpu_count_logical,
            "cuda_available": self.cuda_available,
            "cuda_version": self.cuda_version,
            "torch_version": self.torch_version,
            "mindspore_version": self.mindspore_version,
            "jax_version": self.jax_version,
            "nvidia_smi": self.nvidia_smi,
            "npu_smi": self.npu_smi,
        }
        d.update(self.extras)
        return d


def _cpu_brand_windows() -> str:
    code, out, _ = _run_cmd(
        ["wmic", "cpu", "get", "Name"],
        timeout=10,
    )
    if code == 0 and out:
        lines = [ln.strip() for ln in out.splitlines() if ln.strip() and "Name" not in ln]
        return lines[0] if lines else ""
    return ""


def _cpu_brand_linux() -> str:
    try:
        with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return ""


def probe_hardware() -> HardwareReport:
    """Collect hardware and framework versions without requiring confirmation."""
    r = HardwareReport()
    r.hostname = platform.node()
    r.os_name = platform.system()
    r.os_version = platform.version()
    r.python_version = sys.version.split()[0]
    r.python_executable = sys.executable
    r.cpu_count_logical = os.cpu_count() or 0

    if sys.platform == "win32":
        r.cpu_brand = _cpu_brand_windows() or platform.processor() or ""
    else:
        r.cpu_brand = _cpu_brand_linux() or platform.processor() or ""

    # PyTorch
    try:
        import torch

        r.torch_version = torch.__version__
        r.cuda_available = torch.cuda.is_available()
        if r.cuda_available:
            r.cuda_version = getattr(torch.version, "cuda", None)
    except ImportError:
        r.torch_version = None
        r.cuda_available = None

    try:
        import mindspore as ms

        r.mindspore_version = getattr(ms, "__version__", str(ms))
    except ImportError:
        r.mindspore_version = None

    try:
        import jax

        r.jax_version = getattr(jax, "__version__", None)
    except ImportError:
        r.jax_version = None

    if shutil.which("nvidia-smi"):
        code, out, err = _run_cmd(["nvidia-smi", "-L"])
        r.nvidia_smi = (out + err).strip() if code == 0 else None

    # Huawei Ascend: npu-smi (common on CANN machines)
    if shutil.which("npu-smi"):
        code, out, err = _run_cmd(["npu-smi", "info"])
        r.npu_smi = (out + err).strip() if code == 0 else None
    elif shutil.which("npu-smi-info"):
        code, out, err = _run_cmd(["npu-smi-info"])
        r.npu_smi = (out + err).strip() if code == 0 else None

    return r


def format_hardware_summary(report: HardwareReport) -> str:
    lines = [
        f"Host: {report.hostname}",
        f"OS: {report.os_name} {report.os_version}",
        f"Python: {report.python_version} ({report.python_executable})",
        f"CPU: {report.cpu_brand or 'unknown'} ({report.cpu_count_logical} logical)",
        f"PyTorch: {report.torch_version or 'not installed'}",
        f"CUDA available (torch): {report.cuda_available}",
        f"MindSpore: {report.mindspore_version or 'not installed'}",
        f"JAX: {report.jax_version or 'not installed'}",
    ]
    if report.nvidia_smi:
        lines.append("nvidia-smi -L:\n" + report.nvidia_smi)
    if report.npu_smi:
        lines.append("NPU:\n" + report.npu_smi[:2000])
    return "\n".join(lines)


def user_confirms(prompt: str = "Proceed with evaluation on this machine?") -> bool:
    """Interactive yes/no (for Huawei-style explicit acknowledgement)."""
    try:
        ans = input(f"{prompt} [y/N]: ").strip().lower()
    except EOFError:
        return False
    return ans in ("y", "yes")
