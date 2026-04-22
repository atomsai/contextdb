"""Tests for :class:`SQLiteStore`."""

from __future__ import annotations

import asyncio

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


@pytest.mark.asyncio
async def test_wal_mode(tmp_store: SQLiteStore) -> None:
    """Initialize() must enable WAL journaling for concurrent reads."""
    conn = tmp_store._require_conn()
    cursor = await conn.execute("PRAGMA journal_mode")
    row = await cursor.fetchone()
    assert row is not None
    assert str(row[0]).lower() == "wal"


@pytest.mark.asyncio
async def test_delete_timezone(tmp_store: SQLiteStore) -> None:
    """Soft delete must write a timezone-aware ``updated_at``."""
    item = await tmp_store.add(MemoryItem(content="tz", embedding=[0.0] * 32))
    await tmp_store.delete(item.id, hard=False)
    raw = await tmp_store.get_raw(item.id)
    assert raw is not None
    assert raw.status == MemoryStatus.DELETED
    assert raw.updated_at.tzinfo is not None


@pytest.mark.asyncio
async def test_concurrent_reads(tmp_store: SQLiteStore) -> None:
    """Ten parallel searches must all complete without serialization errors."""
    seeded = await tmp_store.add(
        MemoryItem(content="seed", embedding=[1.0] + [0.0] * 31)
    )
    query = [1.0] + [0.0] * 31
    results = await asyncio.gather(
        *(tmp_store.search_by_embedding(query, top_k=3) for _ in range(10))
    )
    for hits in results:
        assert hits and hits[0].id == seeded.id


@pytest.mark.asyncio
async def test_count_by_type(tmp_store: SQLiteStore) -> None:
    """count_by_type should aggregate without loading rows."""
    await tmp_store.add(MemoryItem(content="f", embedding=[0.0] * 32))
    await tmp_store.add(
        MemoryItem(
            content="e",
            embedding=[0.0] * 32,
            memory_type=MemoryType.EXPERIENTIAL,
        )
    )
    counts = await tmp_store.count_by_type()
    assert counts[MemoryType.FACTUAL.value] == 1
    assert counts[MemoryType.EXPERIENTIAL.value] == 1
    assert counts[MemoryType.WORKING.value] == 0


@pytest.mark.asyncio
async def test_iter_memories_paginates(tmp_store: SQLiteStore) -> None:
    """iter_memories must stream the full set in fixed-size pages."""
    for i in range(7):
        await tmp_store.add(MemoryItem(content=f"m{i}", embedding=[float(i)] * 32))
    seen = [m async for m in tmp_store.iter_memories(batch_size=3)]
    assert len(seen) == 7


@pytest.mark.asyncio
async def test_delete_older_than(tmp_store: SQLiteStore) -> None:
    """delete_older_than issues a single SQL delete."""
    from datetime import datetime, timezone

    item = await tmp_store.add(MemoryItem(content="x", embedding=[0.0] * 32))
    future = datetime.now(tz=timezone.utc).replace(year=2099).isoformat()
    removed = await tmp_store.delete_older_than(future, hard=True)
    assert removed >= 1
    assert await tmp_store.get_raw(item.id) is None
