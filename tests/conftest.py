"""Shared test fixtures for ContextDB."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import TYPE_CHECKING

import pytest_asyncio

from contextdb import ContextDB, ContextDBConfig
from contextdb.store.sqlite_store import SQLiteStore
from contextdb.utils.embeddings import MockEmbedding
from contextdb.utils.llm import MockLLM

if TYPE_CHECKING:
    import pytest


@pytest_asyncio.fixture
async def mock_embedder() -> MockEmbedding:
    return MockEmbedding(dimension=32)


@pytest_asyncio.fixture
async def mock_llm() -> MockLLM:
    return MockLLM()


@pytest_asyncio.fixture
async def tmp_store(tmp_path: Path) -> AsyncIterator[SQLiteStore]:
    store = SQLiteStore(
        storage_url=f"sqlite:///{tmp_path}/test.db",
        embedding_dim=32,
    )
    await store.initialize()
    try:
        yield store
    finally:
        await store.close()


@pytest_asyncio.fixture
async def client(tmp_path: Path) -> AsyncIterator[ContextDB]:
    config = ContextDBConfig(
        storage_url=f"sqlite:///{tmp_path}/client.db",
        embedding_model="mock",
        embedding_dim=32,
        llm_model="mock",
        llm_api_key="mock",
        enable_entity_graph=False,
        enable_multi_graph=False,
        enable_rl_manager=False,
        enable_audit=True,
        enable_auto_link=True,
        pii_action="redact",
    )
    db = ContextDB(config)
    try:
        yield db
    finally:
        await db.close()


@pytest_asyncio.fixture
async def client_full(tmp_path: Path) -> AsyncIterator[ContextDB]:
    config = ContextDBConfig(
        storage_url=f"sqlite:///{tmp_path}/full.db",
        embedding_model="mock",
        embedding_dim=32,
        llm_model="mock",
        llm_api_key="mock",
        enable_entity_graph=True,
        enable_multi_graph=True,
        enable_rl_manager=False,
        enable_audit=True,
        enable_auto_link=True,
        pii_action="redact",
    )
    db = ContextDB(config)
    try:
        yield db
    finally:
        await db.close()


def _noop(_: pytest.Config) -> None:
    pass
