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
from collections import OrderedDict
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

    async def embed_query(self, text: str) -> list[float]:
        """Embed one retrieval query."""
        return (await self.embed([text]))[0]

    async def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """Embed stored memory/document text."""
        return await self.embed(texts)


class OpenAIEmbedding(EmbeddingProvider):
    """OpenAI embedding API wrapper with exponential-backoff retry.

    Also covers any OpenAI-compatible embeddings endpoint when ``base_url``
    is supplied. For unknown models the declared dimension falls back to
    ``dim_override`` (typically ``ContextDBConfig.embedding_dim``) so the
    store schema matches what the endpoint actually returns.
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
        max_retries: int = 3,
        base_url: str | None = None,
        dim_override: int | None = None,
    ) -> None:
        from openai import AsyncOpenAI

        self.model = model
        self.max_retries = max_retries
        self.base_url = base_url
        self._client = AsyncOpenAI(
            api_key=api_key or "contextdb-keyless-local",
            base_url=base_url,
        )
        self._dim = _OPENAI_DIMS.get(model) or dim_override or 1536

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return await self._embed_role(texts, role=None)

    async def embed_query(self, text: str) -> list[float]:
        return (
            await self._embed_role([text], role="query")
        )[0]

    async def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        return await self._embed_role(texts, role="document")

    async def _embed_role(
        self,
        texts: list[str],
        *,
        role: str | None,
    ) -> list[list[float]]:
        if not texts:
            return []
        # OpenAI's per-call limit is 2048 inputs; chunk defensively.
        chunks = [texts[i : i + 2048] for i in range(0, len(texts), 2048)]
        out: list[list[float]] = []
        for chunk in chunks:
            out.extend(
                await self._embed_with_retry(
                    chunk,
                    role=role,
                )
            )
        return out

    async def _embed_with_retry(
        self,
        texts: list[str],
        *,
        role: str | None,
    ) -> list[list[float]]:
        delay = 1.0
        for attempt in range(self.max_retries):
            try:
                response = await self._client.embeddings.create(
                    model=self.model,
                    input=texts,
                    extra_headers=(
                        {"X-ContextDB-Embedding-Role": role}
                        if role is not None
                        else None
                    ),
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
                "Install with `pip install pycontextdb[local]`."
            ) from exc
        self._model = SentenceTransformer(model_name)
        self._model_name = model_name
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

    async def embed_query(self, text: str) -> list[float]:
        if "e5" in self._model_name.lower():
            text = f"query: {text}"
        return (await self.embed([text]))[0]

    async def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        if "e5" in self._model_name.lower():
            texts = [f"passage: {text}" for text in texts]
        return await self.embed(texts)


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


class CachedEmbeddingProvider(EmbeddingProvider):
    """LRU cache over exact input strings. Query embeddings repeat a lot."""

    def __init__(self, inner: EmbeddingProvider, maxsize: int = 2048) -> None:
        self._inner = inner
        self._maxsize = maxsize
        self._cache: OrderedDict[
            tuple[str, str],
            list[float],
        ] = OrderedDict()

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return await self._embed_kind("generic", texts)

    async def embed_query(self, text: str) -> list[float]:
        return (
            await self._embed_kind("query", [text])
        )[0]

    async def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        return await self._embed_kind("document", texts)

    async def _embed_kind(
        self,
        kind: str,
        texts: list[str],
    ) -> list[list[float]]:
        if not texts:
            return []
        if self._maxsize <= 0:
            if kind == "query":
                return [await self._inner.embed_query(texts[0])]
            if kind == "document":
                return await self._inner.embed_documents(texts)
            return await self._inner.embed(texts)
        missing: list[str] = []
        seen: set[str] = set()
        for text in texts:
            key = (kind, text)
            if key in self._cache:
                self._cache.move_to_end(key)
            elif text not in seen:
                missing.append(text)
                seen.add(text)
        if missing:
            if kind == "query":
                vectors = list(
                    await asyncio.gather(
                        *[
                            self._inner.embed_query(text)
                            for text in missing
                        ]
                    )
                )
            elif kind == "document":
                vectors = await self._inner.embed_documents(missing)
            else:
                vectors = await self._inner.embed(missing)
            for text, vector in zip(missing, vectors, strict=True):
                self._cache[(kind, text)] = vector
                if len(self._cache) > self._maxsize:
                    self._cache.popitem(last=False)
        return [self._cache[(kind, text)] for text in texts]

    def dimension(self) -> int:
        return self._inner.dimension()


class TimeoutEmbeddingProvider(EmbeddingProvider):
    """Fail an embedding call after ``timeout_seconds`` so voice turns can degrade."""

    def __init__(self, inner: EmbeddingProvider, timeout_seconds: float) -> None:
        self._inner = inner
        self._timeout = timeout_seconds

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return await asyncio.wait_for(self._inner.embed(texts), timeout=self._timeout)

    async def embed_query(self, text: str) -> list[float]:
        return await asyncio.wait_for(
            self._inner.embed_query(text),
            timeout=self._timeout,
        )

    async def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        return await asyncio.wait_for(
            self._inner.embed_documents(texts),
            timeout=self._timeout,
        )

    def dimension(self) -> int:
        return self._inner.dimension()


def wrap_embedder(
    provider: EmbeddingProvider,
    *,
    cache_size: int = 2048,
    timeout_seconds: float | None = None,
) -> EmbeddingProvider:
    wrapped: EmbeddingProvider = provider
    if timeout_seconds is not None and timeout_seconds > 0:
        wrapped = TimeoutEmbeddingProvider(wrapped, timeout_seconds)
    if cache_size > 0:
        wrapped = CachedEmbeddingProvider(wrapped, maxsize=cache_size)
    return wrapped


def get_embedding_provider(
    model: str,
    api_key: str | None = None,
    base_url: str | None = None,
    **kwargs: Any,
) -> EmbeddingProvider:
    """Pick an embedding provider from a short model string.

    ``mock``/``test`` → :class:`MockEmbedding`. When ``base_url`` is set,
    any model name is sent to that OpenAI-compatible endpoint. Otherwise
    ``text-embedding-*`` or ``openai:*`` → :class:`OpenAIEmbedding`;
    anything else is routed to :class:`SentenceTransformerEmbedding`.
    """
    if model in {"mock", "test"}:
        return MockEmbedding(dimension=kwargs.get("dimension", 384))
    if base_url is not None:
        resolved = model.replace("openai:", "", 1) if model.startswith("openai:") else model
        return OpenAIEmbedding(
            model=resolved,
            api_key=api_key,
            base_url=base_url,
            dim_override=kwargs.get("dimension"),
        )
    if model.startswith("text-embedding-") or model.startswith("openai:"):
        resolved = model.replace("openai:", "", 1) if model.startswith("openai:") else model
        return OpenAIEmbedding(model=resolved, api_key=api_key)
    try:
        return SentenceTransformerEmbedding(model_name=model)
    except RuntimeError as exc:
        raise ConfigError(str(exc)) from exc
