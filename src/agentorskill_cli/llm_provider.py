"""Minimal LLM provider abstraction for agent workflows."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

from openai import OpenAI


@dataclass
class LLMJsonResponse:
    """Structured response plus raw accounting useful for agent audit."""

    data: dict[str, Any]
    raw_text: str
    model: str
    provider: str
    protocol: str
    prompt_chars: int
    completion_chars: int


class OpenAIChatProvider:
    """OpenAI-compatible Chat Completions provider."""

    provider = "openai-compatible"
    protocol = "chat_completions"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
    ) -> None:
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is required for agent mode")
        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        self.client = OpenAI(api_key=key, base_url=base_url or os.environ.get("OPENAI_BASE_URL") or None)

    def complete_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
    ) -> LLMJsonResponse:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content or "{}"
        return LLMJsonResponse(
            data=json.loads(raw),
            raw_text=raw,
            model=self.model,
            provider=self.provider,
            protocol=self.protocol,
            prompt_chars=len(system_prompt) + len(user_prompt),
            completion_chars=len(raw),
        )
