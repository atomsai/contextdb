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
    """OpenAI chat-completions wrapper with retry.

    Also covers any OpenAI-compatible endpoint (Groq, Ollama, vLLM,
    Together, ...) when ``base_url`` is supplied. The OpenAI SDK insists on
    a non-empty ``api_key`` even for keyless local servers, so a placeholder
    is substituted in that case.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        max_retries: int = 3,
        base_url: str | None = None,
    ) -> None:
        from openai import AsyncOpenAI

        self.model = model
        self.max_retries = max_retries
        self.base_url = base_url
        self._client = AsyncOpenAI(
            api_key=api_key or "contextdb-keyless-local",
            base_url=base_url,
        )

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


class LazyLLM(LLMProvider):
    """Defer provider construction until the first ``generate()`` call.

    The realtime write path (``factual.add_fast``) must work — and stay
    off the LLM — even when no API key is configured. Constructing the
    provider lazily means a missing key surfaces as a :class:`ConfigError`
    at the first call that genuinely needs a model, instead of breaking
    unrelated LLM-free operations at init time.
    """

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.kwargs = kwargs
        self._inner: LLMProvider | None = None

    @property
    def materialized(self) -> bool:
        """Whether the underlying provider has been constructed yet."""
        return self._inner is not None

    def materialize(self) -> LLMProvider:
        if self._inner is None:
            self._inner = get_llm_provider(
                self.model,
                self.api_key,
                base_url=self.base_url,
                **self.kwargs,
            )
        return self._inner

    async def generate(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.0,
        max_tokens: int = 1000,
        response_format: type | None = None,
    ) -> str:
        return await self.materialize().generate(
            prompt,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
        )


def get_llm_provider(
    model: str,
    api_key: str | None = None,
    base_url: str | None = None,
    **kwargs: Any,
) -> LLMProvider:
    """Route a model name to its provider.

    ``mock``/``test`` → :class:`MockLLM`. When ``base_url`` is set, ANY
    model name is accepted and sent to that OpenAI-compatible endpoint
    (Groq, Ollama, vLLM, Together, ...). Without ``base_url``, ``gpt-*``,
    ``o1-*`` and ``openai:*`` route to OpenAI and require an API key.
    Anything else raises :class:`ConfigError`.
    """
    if model in {"mock", "test"}:
        return MockLLM(**kwargs)
    if base_url is not None:
        resolved = model.replace("openai:", "", 1) if model.startswith("openai:") else model
        return OpenAILLM(model=resolved, api_key=api_key, base_url=base_url)
    if model.startswith("gpt-") or model.startswith("o1-") or model.startswith("openai:"):
        resolved = model.replace("openai:", "", 1) if model.startswith("openai:") else model
        if not api_key:
            raise ConfigError(
                f"LLM model '{model}' routes to the OpenAI default endpoint but no "
                "API key is configured. Set CONTEXTDB_LLM_API_KEY or OPENAI_API_KEY, "
                "or pass llm_base_url to use an OpenAI-compatible endpoint "
                "(Groq, Ollama, vLLM, Together, ...)."
            )
        return OpenAILLM(model=resolved, api_key=api_key)
    raise ConfigError(
        f"Unknown LLM model '{model}'. Use 'mock' for tests, 'gpt-*'/'o1-*' for "
        "OpenAI, or set llm_base_url to route any model name to an "
        "OpenAI-compatible endpoint."
    )
