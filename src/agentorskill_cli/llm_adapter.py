"""Use an LLM to turn README/docs text into a validated AdapterSpec."""

from __future__ import annotations

import json
import os
from typing import Any

from openai import OpenAI

from agentorskill_cli.adapter_schema import AdapterSpec, validate_adapter_dict


SYSTEM_PROMPT = """You are an expert at PyTorch migration libraries (TorchAX, MindTorch, torch_xla, etc.).
Read the user's documentation and output ONE JSON object matching this schema exactly:

{
  "library_name": "string (required)",
  "version_hint": "string or null",
  "install": {
    "commands": ["shell commands if stated"],
    "pip_packages": ["package names if stated"],
    "notes": "string"
  },
  "enable": {
    "preamble": "Python code: imports + enable calls that MUST run before other torch imports when applicable",
    "import_order_note": "string"
  },
  "device": {
    "target_device": "e.g. cpu, cuda:0, jax",
    "tensor_device_code": "optional hint string",
    "model_to_device": "optional hint string"
  },
  "constraints": [{"api_or_area": "string", "reason": "string"}],
  "examples": [{"title": "string", "code": "string", "source_ref": "string"}],
  "confidence": {
    "overall": 0.0,
    "by_field": {},
    "evidence": [{"field": "string", "quote": "short quote from doc", "source": "which section"}]
  },
  "extra": {}
}

Rules:
- enable.preamble must be non-empty valid Python.
- If docs say 'import torchax' and 'torchax.enable_globally()', include both in preamble.
- For MindTorch style: preamble must import mstorch_enable BEFORE torch if docs require it.
- Prefer documented minimal setup over guessing.
- Put uncertain items in constraints or notes, lower confidence.
Output ONLY valid JSON, no markdown fences.
"""


def extract_adapter_with_llm(
    doc_text: str,
    *,
    model: str | None = None,
    base_url: str | None = None,
) -> AdapterSpec:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Provide --adapter-file with a JSON AdapterSpec, "
            "or set OPENAI_API_KEY for LLM extraction."
        )

    client = OpenAI(api_key=api_key, base_url=base_url or None)
    m = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

    user = f"Documentation follows.\n\n---\n{doc_text}\n---\n\nRespond with the JSON object only."
    resp = client.chat.completions.create(
        model=m,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        temperature=0.1,
        response_format={"type": "json_object"},
    )
    raw = resp.choices[0].message.content or "{}"
    data: dict[str, Any] = json.loads(raw)
    return validate_adapter_dict(data)


def extract_adapter_json_only(raw_json: str) -> AdapterSpec:
    data = json.loads(raw_json)
    return validate_adapter_dict(data)
