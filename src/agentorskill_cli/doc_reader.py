"""Load documentation from local paths, URLs, or short text for LLM extraction."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

USER_AGENT = "agentorskill-cli/0.1 (doc fetch; +https://github.com)"


def is_url(s: str) -> bool:
    try:
        u = urlparse(s)
        return u.scheme in ("http", "https") and bool(u.netloc)
    except Exception:
        return False


def fetch_url(url: str, timeout: float = 30.0) -> str:
    """GET URL and return text (utf-8)."""
    headers = {"User-Agent": USER_AGENT}
    with httpx.Client(follow_redirects=True, timeout=timeout) as client:
        r = client.get(url, headers=headers)
        r.raise_for_status()
        return r.text


def read_local_path(path: str | Path) -> str:
    p = Path(path)
    if p.is_file():
        return p.read_text(encoding="utf-8", errors="replace")
    if p.is_dir():
        parts: list[str] = []
        for name in ("README.md", "README.rst", "README", "readme.md"):
            f = p / name
            if f.is_file():
                parts.append(f"## File: {f.name}\n\n")
                parts.append(f.read_text(encoding="utf-8", errors="replace"))
                break
        return "\n".join(parts) if parts else ""
    raise FileNotFoundError(str(p))


def collect_docs(
    *,
    readme_url: str | None = None,
    doc_urls: list[str] | None = None,
    local_path: str | None = None,
    extra_text: str | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """
    Aggregate documentation into one blob for the LLM.
    Returns (combined_markdown, sources_metadata).
    """
    chunks: list[str] = []
    sources: list[dict[str, Any]] = []

    if local_path:
        text = read_local_path(local_path)
        sources.append({"type": "path", "ref": local_path, "chars": len(text)})
        chunks.append(f"## Source: local {local_path}\n\n{text}")

    if readme_url:
        text = fetch_url(readme_url) if is_url(readme_url) else read_local_path(readme_url)
        sources.append({"type": "url" if is_url(readme_url) else "file", "ref": readme_url, "chars": len(text)})
        chunks.append(f"## Source: README {readme_url}\n\n{text}")

    for u in doc_urls or []:
        text = fetch_url(u) if is_url(u) else read_local_path(u)
        sources.append({"type": "url" if is_url(u) else "file", "ref": u, "chars": len(text)})
        chunks.append(f"## Source: {u}\n\n{text}")

    if extra_text:
        sources.append({"type": "inline", "ref": "extra_text", "chars": len(extra_text)})
        chunks.append(f"## Source: user extra\n\n{extra_text}")

    combined = "\n\n".join(chunks)
    # Trim extremely large inputs for LLM (keep head + tail)
    max_chars = 120_000
    if len(combined) > max_chars:
        head = combined[: max_chars // 2]
        tail = combined[-max_chars // 2 :]
        combined = head + "\n\n...[truncated]...\n\n" + tail
    return combined, sources


def snippet_for_evidence(text: str, phrase: str, context: int = 120) -> str:
    """Extract a short quote around phrase for evidence."""
    if not phrase:
        return ""
    idx = text.find(phrase)
    if idx < 0:
        return phrase[:200]
    start = max(0, idx - context)
    end = min(len(text), idx + len(phrase) + context)
    return text[start:end].replace("\n", " ")
