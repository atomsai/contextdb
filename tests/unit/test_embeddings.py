"""Tests for embedding providers."""

from __future__ import annotations

import pytest

from contextdb.core.exceptions import ConfigError
from contextdb.utils.embeddings import (
    CachedEmbeddingProvider,
    EmbeddingProvider,
    MockEmbedding,
    get_embedding_provider,
)


@pytest.mark.asyncio
async def test_mock_embedding_deterministic() -> None:
    provider = MockEmbedding(dimension=16)
    v1 = (await provider.embed(["hello world"]))[0]
    v2 = (await provider.embed(["hello world"]))[0]
    assert v1 == v2


@pytest.mark.asyncio
async def test_mock_embedding_distinct() -> None:
    provider = MockEmbedding(dimension=16)
    v_hello = (await provider.embed(["hello"]))[0]
    v_bye = (await provider.embed(["goodbye"]))[0]
    assert v_hello != v_bye


def test_factory_routes_mock() -> None:
    provider = get_embedding_provider("mock", dimension=8)
    assert isinstance(provider, MockEmbedding)
    assert provider.dimension() == 8


def test_factory_unknown_model_raises() -> None:
    with pytest.raises((ConfigError, RuntimeError)):
        get_embedding_provider("a-random-name-that-does-not-exist")


@pytest.mark.asyncio
async def test_cache_separates_query_and_document_roles() -> None:
    class RoleProvider(EmbeddingProvider):
        async def embed(self, texts: list[str]) -> list[list[float]]:
            return [[0.0] for _text in texts]

        async def embed_query(self, text: str) -> list[float]:
            return [1.0]

        async def embed_documents(
            self,
            texts: list[str],
        ) -> list[list[float]]:
            return [[2.0] for _text in texts]

        def dimension(self) -> int:
            return 1

    cached = CachedEmbeddingProvider(RoleProvider())
    assert await cached.embed_query("same text") == [1.0]
    assert await cached.embed_documents(["same text"]) == [[2.0]]
    assert await cached.embed_query("same text") == [1.0]
