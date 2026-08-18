"""Foreign memory IDs must be rejected identically on every store backend.

The ownership checks for ID-based confirm/forget live in the client and
trust layers above the store, so the same tests run against SQLite always,
and against Postgres when ``CONTEXTDB_TEST_POSTGRES_URL`` points at a test
database (CI without Postgres skips those parametrizations).
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import pytest

import contextdb
from contextdb.core.config import ContextDBConfig
from contextdb.core.exceptions import MemoryNotFoundError
from contextdb.store.factory import open_store

_PG_URL = os.environ.get("CONTEXTDB_TEST_POSTGRES_URL")

_BACKENDS = ["sqlite", "postgres"]


def _storage_url(backend: str, tmp_path: Path) -> str:
    if backend == "postgres":
        if not _PG_URL:
            pytest.skip("set CONTEXTDB_TEST_POSTGRES_URL to run Postgres store tests")
        return _PG_URL
    return f"sqlite:///{tmp_path}/scoped-{uuid.uuid4().hex}.db"


def _cfg(backend: str, tmp_path: Path, storage_url: str | None = None) -> ContextDBConfig:
    return ContextDBConfig(
        storage_url=storage_url or _storage_url(backend, tmp_path),
        embedding_model="mock",
        embedding_dim=32,
        llm_model="mock",
        llm_api_key="mock",
        enable_entity_graph=False,
        enable_multi_graph=False,
        enable_auto_link=False,
        enable_audit=True,
    )


def _user(role: str) -> str:
    """Unique user ids keep a shared Postgres database isolated per test."""
    return f"{role}-{uuid.uuid4().hex}"


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", _BACKENDS)
async def test_confirm_rejects_foreign_memory_id_on_store(
    backend: str, tmp_path: Path
) -> None:
    alice, bob = _user("alice"), _user("bob")
    db = contextdb.init(config=_cfg(backend, tmp_path))
    try:
        item = await db.factual.add(
            "Alice PIN is 7788",
            source="user_stated",
            confidence=0.5,
            action_relevant=True,
            entity="account",
            attribute="pin",
            user_id=alice,
        )
        with pytest.raises(MemoryNotFoundError):
            await db.factual.confirm(item.id, user_id=bob)
        still = await db.get(item.id)
        assert still is not None
        assert still.confirmed is False

        confirmed = await db.factual.confirm(item.id, user_id=alice)
        assert confirmed.confirmed is True
    finally:
        await db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", _BACKENDS)
async def test_forget_rejects_foreign_memory_id_on_store(
    backend: str, tmp_path: Path
) -> None:
    alice, bob = _user("alice"), _user("bob")
    db = contextdb.init(config=_cfg(backend, tmp_path))
    try:
        item = await db.factual.add(
            "Alice PIN is 7788", source="user_stated", user_id=alice
        )
        with pytest.raises(MemoryNotFoundError):
            await db.forget(user_id=bob, memory_id=item.id)
        assert await db.get(item.id) is not None

        assert await db.forget(user_id=bob, memory_id="no-such-id") == 0
        assert await db.forget(user_id=alice, memory_id=item.id) == 1
        assert await db.get(item.id) is None
    finally:
        await db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", _BACKENDS)
async def test_scoped_store_hides_foreign_ids(backend: str, tmp_path: Path) -> None:
    """Store-level guarantee under the client checks: a scoped store cannot
    see or delete another user's row; an unscoped store can — which is what
    lets the client reject a known foreign id instead of deleting it."""
    alice, bob = _user("alice"), _user("bob")
    url = _storage_url(backend, tmp_path)
    db = contextdb.init(config=_cfg(backend, tmp_path, storage_url=url))
    try:
        item = await db.factual.add("bob secret", source="user_stated", user_id=bob)
    finally:
        await db.close()

    scoped = open_store(url, user_id=alice, embedding_dim=32)
    await scoped.initialize()
    try:
        assert await scoped.get(item.id) is None
        assert await scoped.get_raw(item.id) is None
        assert await scoped.delete(item.id, hard=True) is False
    finally:
        await scoped.close()

    unscoped = open_store(url, embedding_dim=32)
    await unscoped.initialize()
    try:
        seen = await unscoped.get_raw(item.id)
        assert seen is not None
        assert seen.user_id == bob
    finally:
        await unscoped.close()
