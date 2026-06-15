"""Built-in report visualizations for eval-migration summaries."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any


FAILURE_PALETTE = {
    "DependencyMissing": "#8c8c8c",
    "EnvironmentFailure": "#bdbdbd",
    "ImportOrderError": "#969696",
    "OperatorNotFound": "#d73027",
    "TypeMismatch": "#fc8d59",
    "DeviceMismatch": "#fee08b",
    "ShapeMismatch": "#4575b4",
    "RuntimeCrash": "#313695",
    "NumericMismatch": "#f46d43",
    "AutogradFailure": "#7b3294",
    "TrainingDivergence": "#1a9850",
    "TranslationError": "#66c2a5",
    "Unknown": "#636363",
}


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def _label_from_path(path: Path, data: dict[str, Any]) -> str:
    suite_id = data.get("suite_id")
    if suite_id:
        return str(suite_id)
    adapter = data.get("adapter") or {}
    if isinstance(adapter, dict) and adapter.get("library_name"):
        return str(adapter["library_name"])
    parent = path.parent.name
    if parent in {"eval", "reports"} and path.parent.parent.name:
        return path.parent.parent.name
    return parent or path.stem


def _infer_failure_class(error: str | None) -> str:
    text = (error or "").lower()
    if not text:
        return "Unknown"
    if "no module named" in text or "modulenotfounderror" in text or "importerror" in text:
        return "DependencyMissing"
    if "shape" in text or "size mismatch" in text or "mat1 and mat2" in text:
        return "ShapeMismatch"
    if "device" in text or "cuda" in text or "npu" in text or "ascend" in text:
        return "DeviceMismatch"
    if "dtype" in text or "type" in text:
        return "TypeMismatch"
    if "grad" in text or "backward" in text or "autograd" in text:
        return "AutogradFailure"
    return "RuntimeCrash"


def _failure_classes_from_demo_payload(data: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    suites = data.get("suites") or {}
    if not isinstance(suites, dict):
        return counts
    diagnostics: dict[str, str] = {}
    agent = data.get("agent_report") or {}
    for item in agent.get("diagnostics") or []:
        if isinstance(item, dict) and item.get("case_id") and item.get("failure_class"):
            diagnostics[str(item["case_id"])] = str(item["failure_class"])
    for suite in suites.values():
        if not isinstance(suite, dict):
            continue
        for case in suite.get("cases") or []:
            if not isinstance(case, dict):
                continue
            if case.get("ok") or case.get("skipped"):
                continue
            cls = diagnostics.get(str(case.get("case_id"))) or _infer_failure_class(case.get("error"))
            counts[cls] = counts.get(cls, 0) + 1
    return counts


def read_summary(path: Path) -> dict[str, Any]:
    data = _load_json(path)
    failures = data.get("failure_classes") if isinstance(data.get("failure_classes"), dict) else None
    if failures is None:
        failures = _failure_classes_from_demo_payload(data)
    totals = data.get("totals") if isinstance(data.get("totals"), dict) else {}
    eval_summary = data.get("evaluation_summary") if isinstance(data.get("evaluation_summary"), dict) else {}
    compatibility_rate = totals.get("compatibility_rate", eval_summary.get("adaptation_rate_overall"))
    raw_pass_rate = totals.get("raw_pass_rate")
    if raw_pass_rate is None and isinstance(eval_summary.get("counts"), dict):
        counts = eval_summary["counts"]
        total = counts.get("total") or 0
        raw_pass_rate = (counts.get("passed") or 0) / total if total else None
    total_runs = totals.get("total")
    if total_runs is None and isinstance(eval_summary.get("counts"), dict):
        total_runs = eval_summary["counts"].get("total")
    return {
        "path": str(path),
        "label": _label_from_path(path, data),
        "failure_classes": {str(k): int(v) for k, v in (failures or {}).items()},
        "compatibility_rate": compatibility_rate,
        "raw_pass_rate": raw_pass_rate,
        "total": total_runs,
    }


def load_summaries(paths: list[Path]) -> list[dict[str, Any]]:
    summaries = [read_summary(path) for path in paths]
    seen: dict[str, int] = {}
    for item in summaries:
        base = item["label"]
        seen[base] = seen.get(base, 0) + 1
        if seen[base] > 1:
            item["label"] = f"{base}_{seen[base]}"
    return summaries


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - depends on optional package
        raise SystemExit(
            "Matplotlib is required for report plotting. Install matplotlib or run evaluation without plot-reports."
        ) from exc
    plt.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
        }
    )
    return plt


def _save_figure(fig: Any, out_base: Path) -> list[Path]:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    pdf = out_base.with_suffix(".pdf")
    png = out_base.with_suffix(".png")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=220, bbox_inches="tight")
    try:
        import matplotlib.pyplot as plt

        plt.close(fig)
    except Exception:
        pass
    return [pdf, png]


def plot_failure_taxonomy(summaries: list[dict[str, Any]], out_base: Path) -> list[Path]:
    plt = _require_matplotlib()
    classes = sorted({name for item in summaries for name in item["failure_classes"]})
    if not classes:
        classes = ["NoFailure"]
    labels = [item["label"] for item in summaries]
    width = max(4.2, min(9.0, 0.55 * len(labels) + 2.6))
    fig, ax = plt.subplots(figsize=(width, 2.6))
    bottoms = [0] * len(labels)
    for cls in classes:
        values = [item["failure_classes"].get(cls, 0) for item in summaries]
        if not any(values):
            continue
        ax.bar(labels, values, bottom=bottoms, label=cls, color=FAILURE_PALETTE.get(cls, "#9e9e9e"), linewidth=0)
        bottoms = [a + b for a, b in zip(bottoms, values)]
    ax.set_ylabel("Failed runs")
    ax.set_title("Failure taxonomy across evaluation reports")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.7)
    max_total = max(bottoms) if bottoms else 0
    ax.set_ylim(0, max(1, math.ceil(max_total * 1.18)))
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    fig.tight_layout()
    return _save_figure(fig, out_base)


def plot_compatibility_overview(summaries: list[dict[str, Any]], out_base: Path) -> list[Path]:
    plt = _require_matplotlib()
    labels = [item["label"] for item in summaries]
    compatibility = [item.get("compatibility_rate") for item in summaries]
    raw = [item.get("raw_pass_rate") for item in summaries]
    x = list(range(len(labels)))
    width = max(4.2, min(9.0, 0.55 * len(labels) + 2.4))
    fig, ax = plt.subplots(figsize=(width, 2.4))
    bar_width = 0.38
    ax.bar([i - bar_width / 2 for i in x], [0 if v is None else float(v) for v in compatibility], width=bar_width, label="Compatibility", color="#2b8cbe")
    ax.bar([i + bar_width / 2 for i in x], [0 if v is None else float(v) for v in raw], width=bar_width, label="Raw pass", color="#a6bddb")
    for i, value in enumerate(compatibility):
        if value is None:
            ax.text(i - bar_width / 2, 0.03, "n/a", ha="center", va="bottom", fontsize=7, rotation=90)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Rate")
    ax.set_title("Compatibility vs raw pass rate")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.7)
    ax.legend(frameon=False)
    fig.tight_layout()
    return _save_figure(fig, out_base)


def write_plot_manifest(out_dir: Path, summaries: list[dict[str, Any]], figures: list[Path]) -> Path:
    manifest = {
        "schema_version": "tbbcc.report_plots.v0.1",
        "summaries": summaries,
        "figures": [str(path) for path in figures],
    }
    path = out_dir / "plot_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def generate_report_plots(summary_paths: list[Path], out_dir: Path, kinds: list[str] | None = None) -> dict[str, Any]:
    if not summary_paths:
        raise ValueError("At least one summary JSON path is required")
    kinds = kinds or ["failure-taxonomy", "compatibility-overview"]
    summaries = load_summaries(summary_paths)
    figures: list[Path] = []
    if "failure-taxonomy" in kinds:
        figures.extend(plot_failure_taxonomy(summaries, out_dir / "failure_taxonomy"))
    if "compatibility-overview" in kinds:
        figures.extend(plot_compatibility_overview(summaries, out_dir / "compatibility_overview"))
    manifest = write_plot_manifest(out_dir, summaries, figures)
    return {
        "out_dir": str(out_dir),
        "manifest": str(manifest),
        "figures": [str(path) for path in figures],
    }
