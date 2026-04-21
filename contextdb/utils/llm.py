"""LLM providers.

Minimal async contract over an LLM chat/completion, used by extraction,
compression, causal inference, and RL-as-policy pathways. Structured output is
approximated by asking the model for JSON and validating at the call site.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Any

from contextdb.core.exceptions import ConfigError


class LLMProvider(ABC):
    """Common interface for text-in / text-out LLMs."""

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.0,
        max_tokens: int = 1000,
        response_format: type | None = None,
    ) -> str: ...


class OpenAILLM(LLMProvider):
    """OpenAI chat-completions wrapper with retry."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        max_retries: int = 3,
    ) -> None:
        from openai import AsyncOpenAI

        self.model = model
        self.max_retries = max_retries
        self._client = AsyncOpenAI(api_key=api_key)

    async def generate(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.0,
        max_tokens: int = 1000,
        response_format: type | None = None,
    ) -> str:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        extra: dict[str, Any] = {}
        if response_format is not None:
            # Ask for JSON; callers validate against their Pydantic model.
            extra["response_format"] = {"type": "json_object"}

        delay = 1.0
        for attempt in range(self.max_retries):
            try:
                response = await self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,  # type: ignore[arg-type]
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **extra,
                )
                content = response.choices[0].message.content or ""
                return content
            except Exception:  # noqa: BLE001
                if attempt == self.max_retries - 1:
                    raise
                await asyncio.sleep(delay)
                delay *= 2
        return ""  # pragma: no cover


class MockLLM(LLMProvider):
    """Scriptable LLM for tests.

    ``responses`` maps substring keys to the string that should be returned
    when that substring appears anywhere in the prompt. All calls are logged
    on :attr:`calls` so tests can assert on prompt content.
    """

    def __init__(
        self,
        responses: dict[str, str] | None = None,
        default: str = '{"facts": [], "entities": []}',
    ) -> None:
        self.responses = responses or {}
        self.default = default
        self.calls: list[dict[str, Any]] = []

    async def generate(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.0,
        max_tokens: int = 1000,
        response_format: type | None = None,
    ) -> str:
        self.calls.append(
            {
                "prompt": prompt,
                "system": system,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        for key, response in self.responses.items():
            if key in prompt:
                return response
        return self.default


def get_llm_provider(
    model: str,
    api_key: str | None = None,
    **kwargs: Any,
) -> LLMProvider:
    """Route a model name to its provider.

    ``mock``/``test`` → :class:`MockLLM`; ``gpt-*`` or ``o1-*`` →
    :class:`OpenAILLM`. Anything else raises :class:`ConfigError`.
    """
    if model in {"mock", "test"}:
        return MockLLM(**kwargs)
    if model.startswith("gpt-") or model.startswith("o1-") or model.startswith("openai:"):
        resolved = model.replace("openai:", "", 1) if model.startswith("openai:") else model
        return OpenAILLM(model=resolved, api_key=api_key)
    raise ConfigError(
        f"Unknown LLM model '{model}'. Use 'mock' for tests or 'gpt-*'/'o1-*' for OpenAI."
    )
