"""Tests for :class:`SQLiteStore`."""

from __future__ import annotations

import pytest

from contextdb.core.exceptions import MemoryNotFoundError
from contextdb.core.models import MemoryItem, MemoryStatus, MemoryType
from contextdb.store.sqlite_store import SQLiteStore


@pytest.mark.asyncio
async def test_add_and_get_roundtrip(tmp_store: SQLiteStore) -> None:
    item = MemoryItem(
        content="hello world",
        embedding=[0.1] * 32,
        memory_type=MemoryType.FACTUAL,
    )
    stored = await tmp_store.add(item)
    fetched = await tmp_store.get(stored.id)
    assert fetched is not None
    assert fetched.content == "hello world"
    assert fetched.memory_type == MemoryType.FACTUAL
    # Access counter is bumped in the DB; refetch to observe the new value.
    refetched = await tmp_store.get_raw(stored.id)
    assert refetched is not None
    assert refetched.access_count >= 1


@pytest.mark.asyncio
async def test_update_fields(tmp_store: SQLiteStore) -> None:
    item = MemoryItem(content="x", embedding=[0.0] * 32)
    stored = await tmp_store.add(item)
    updated = await tmp_store.update(stored.id, content="y", metadata={"k": 1})
    assert updated.content == "y"
    assert updated.metadata == {"k": 1}


@pytest.mark.asyncio
async def test_update_missing_raises(tmp_store: SQLiteStore) -> None:
    with pytest.raises(MemoryNotFoundError):
        await tmp_store.update("nope", content="x")


@pytest.mark.asyncio
async def test_soft_and_hard_delete(tmp_store: SQLiteStore) -> None:
    item = await tmp_store.add(MemoryItem(content="soft", embedding=[0.0] * 32))
    await tmp_store.delete(item.id)  # soft
    raw = await tmp_store.get_raw(item.id)
    assert raw is not None and raw.status == MemoryStatus.DELETED
    await tmp_store.delete(item.id, hard=True)
    assert await tmp_store.get_raw(item.id) is None


@pytest.mark.asyncio
async def test_list_and_count(tmp_store: SQLiteStore) -> None:
    for i in range(3):
        await tmp_store.add(
            MemoryItem(content=f"m{i}", embedding=[float(i)] * 32)
        )
    assert await tmp_store.count() == 3
    items = await tmp_store.list_memories()
    assert len(items) == 3


@pytest.mark.asyncio
async def test_search_by_embedding(tmp_store: SQLiteStore) -> None:
    a = await tmp_store.add(MemoryItem(content="a", embedding=[1.0] + [0.0] * 31))
    b = await tmp_store.add(MemoryItem(content="b", embedding=[0.0] + [1.0] + [0.0] * 30))
    query = [1.0] + [0.0] * 31
    hits = await tmp_store.search_by_embedding(query, top_k=1)
    assert hits
    assert hits[0].id == a.id
    assert b.id  # unused but validated above
