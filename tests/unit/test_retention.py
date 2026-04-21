"""Tests for retention enforcement."""

from __future__ import annotations

from datetime import timedelta

import pytest

from contextdb.core.models import MemoryItem, MemoryStatus, MemoryType, RetentionPolicy
from contextdb.privacy.retention import RetentionManager
from contextdb.store.sqlite_store import SQLiteStore


@pytest.mark.asyncio
async def test_enforce_archives_expired(tmp_store: SQLiteStore) -> None:
    policy = RetentionPolicy(
        default_ttl=timedelta(seconds=0),
        factual_ttl=timedelta(seconds=0),
        experiential_ttl=timedelta(seconds=0),
        working_ttl=timedelta(seconds=0),
    )
    mgr = RetentionManager(tmp_store, None, policy)
    item = await tmp_store.add(
        MemoryItem(content="old", embedding=[0.0] * 32, memory_type=MemoryType.FACTUAL)
    )
    affected = await mgr.enforce()
    assert affected >= 1
    raw = await tmp_store.get_raw(item.id)
    assert raw is not None
    assert raw.status == MemoryStatus.ARCHIVED


@pytest.mark.asyncio
async def test_erase_user_hard_deletes(tmp_store: SQLiteStore) -> None:
    tmp_store._user_id = "u1"
    item = await tmp_store.add(
        MemoryItem(content="x", embedding=[0.0] * 32)
    )
    mgr = RetentionManager(tmp_store, None, RetentionPolicy())
    deleted = await mgr.erase_user("u1")
    assert deleted == 1
    assert await tmp_store.get_raw(item.id) is None
