"""Tests for the top-level ``contextdb`` package."""

from __future__ import annotations

import pytest

import contextdb


def test_version_exported() -> None:
    assert contextdb.__version__
    assert "." in contextdb.__version__


def test_init_returns_client() -> None:
    client = contextdb.init(user_id="u1", llm_api_key="x", storage_url="sqlite:///:memory:")
    assert isinstance(client, contextdb.ContextDB)
    assert client.user_id == "u1"


def test_public_exports_present() -> None:
    expected = {
        "ConfigError",
        "ContextDB",
        "ContextDBConfig",
        "ContextDBError",
        "MemoryNotFoundError",
        "PrivacyError",
        "StorageError",
        "__version__",
        "init",
    }
    assert expected.issubset(set(contextdb.__all__))


@pytest.mark.asyncio
async def test_hosts_can_share_one_embedding_provider() -> None:
    class SharedProvider(contextdb.EmbeddingProvider):
        def __init__(self) -> None:
            self.calls = 0
            self.closed = 0

        async def embed(self, texts: list[str]) -> list[list[float]]:
            self.calls += 1
            return [[1.0, 0.0] for _text in texts]

        def dimension(self) -> int:
            return 2

        async def close(self) -> None:
            self.closed += 1

    provider = SharedProvider()
    first = contextdb.init(
        user_id="first",
        storage_url="sqlite:///:memory:",
        embedding_model="shared-test",
        embedding_dim=2,
        llm_model="mock",
        llm_api_key="mock",
        embedding_provider=provider,
    )
    second = contextdb.init(
        user_id="second",
        storage_url="sqlite:///:memory:",
        embedding_model="shared-test",
        embedding_dim=2,
        llm_model="mock",
        llm_api_key="mock",
        embedding_provider=provider,
    )
    try:
        await first.factual.add("first", source="user_stated")
        await second.factual.add("second", source="user_stated")
    finally:
        await first.close()
        await second.close()
    assert provider.calls == 2
    assert provider.closed == 0
    await provider.close()
    assert provider.closed == 1
