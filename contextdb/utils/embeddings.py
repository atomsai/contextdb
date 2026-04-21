"""Embedding providers.

ContextDB speaks a minimal :class:`EmbeddingProvider` protocol so swapping
backends (OpenAI ↔ local sentence-transformers ↔ deterministic mock) is a
one-line change. :func:`get_embedding_provider` is the factory most callers
want.
"""

from __future__ import annotations

import asyncio
import hashlib
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from contextdb.core.exceptions import ConfigError

# Known OpenAI embedding dimensions; update as new models ship.
_OPENAI_DIMS: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class EmbeddingProvider(ABC):
    """Async embedding contract. Implementations must be batch-safe."""

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]: ...

    @abstractmethod
    def dimension(self) -> int: ...


class OpenAIEmbedding(EmbeddingProvider):
    """OpenAI embedding API wrapper with exponential-backoff retry."""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
        max_retries: int = 3,
    ) -> None:
        from openai import AsyncOpenAI

        self.model = model
        self.max_retries = max_retries
        self._client = AsyncOpenAI(api_key=api_key)
        self._dim = _OPENAI_DIMS.get(model, 1536)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        # OpenAI's per-call limit is 2048 inputs; chunk defensively.
        chunks = [texts[i : i + 2048] for i in range(0, len(texts), 2048)]
        out: list[list[float]] = []
        for chunk in chunks:
            out.extend(await self._embed_with_retry(chunk))
        return out

    async def _embed_with_retry(self, texts: list[str]) -> list[list[float]]:
        delay = 1.0
        for attempt in range(self.max_retries):
            try:
                response = await self._client.embeddings.create(
                    model=self.model, input=texts
                )
                return [d.embedding for d in response.data]
            except Exception:  # noqa: BLE001
                if attempt == self.max_retries - 1:
                    raise
                await asyncio.sleep(delay)
                delay *= 2
        return []  # pragma: no cover

    def dimension(self) -> int:
        return self._dim


class SentenceTransformerEmbedding(EmbeddingProvider):
    """Local-model embeddings via the optional ``sentence-transformers`` dep."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - optional dep
            raise RuntimeError(
                "sentence-transformers is not installed. "
                "Install with `pip install contextdb[local]`."
            ) from exc
        self._model = SentenceTransformer(model_name)
        self._dim = int(self._model.get_sentence_embedding_dimension())

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        loop = asyncio.get_running_loop()

        def _run() -> list[list[float]]:
            return [list(map(float, v)) for v in self._model.encode(texts)]

        return await loop.run_in_executor(None, _run)

    def dimension(self) -> int:
        return self._dim


class MockEmbedding(EmbeddingProvider):
    """Deterministic pseudo-random embeddings for tests; no network calls."""

    def __init__(self, dimension: int = 384) -> None:
        self._dim = dimension

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._text_to_vector(t) for t in texts]

    def _text_to_vector(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:4], "big")
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(self._dim).astype(np.float32)
        # Encourage discrimination by mixing in per-word contribution.
        for word in text.lower().split():
            word_digest = hashlib.md5(word.encode("utf-8"), usedforsecurity=False).digest()
            w_seed = int.from_bytes(word_digest[:4], "big")
            word_rng = np.random.default_rng(w_seed)
            vec += 0.5 * word_rng.standard_normal(self._dim).astype(np.float32)
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec = vec / norm
        return [float(x) for x in vec]

    def dimension(self) -> int:
        return self._dim


def get_embedding_provider(
    model: str,
    api_key: str | None = None,
    **kwargs: Any,
) -> EmbeddingProvider:
    """Pick an embedding provider from a short model string.

    ``mock``/``test`` → :class:`MockEmbedding`; ``text-embedding-*`` or
    ``openai:*`` → :class:`OpenAIEmbedding`; anything else is routed to
    :class:`SentenceTransformerEmbedding`.
    """
    if model in {"mock", "test"}:
        return MockEmbedding(dimension=kwargs.get("dimension", 384))
    if model.startswith("text-embedding-") or model.startswith("openai:"):
        resolved = model.replace("openai:", "", 1) if model.startswith("openai:") else model
        return OpenAIEmbedding(model=resolved, api_key=api_key)
    try:
        return SentenceTransformerEmbedding(model_name=model)
    except RuntimeError as exc:
        raise ConfigError(str(exc)) from exc
