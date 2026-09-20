from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from contextdb.core.exceptions import ConfigError
from contextdb.core.models import MemoryItem
from contextdb.store.factory import (
    is_postgres_url,
    normalize_postgres_url,
    open_store,
)
from contextdb.store.pg_sql import qmark_to_dollar, split_script, translate_sqlite_sql
from contextdb.store.postgres_store import PostgresStore, _PgAdapter


def test_qmark_to_dollar() -> None:
    assert qmark_to_dollar("SELECT * FROM t WHERE a = ? AND b = ?") == (
        "SELECT * FROM t WHERE a = $1 AND b = $2"
    )


def test_insert_or_replace_semantic() -> None:
    sql = translate_sqlite_sql(
        "INSERT OR REPLACE INTO semantic_edges "
        "(source_id, target_id, weight, metadata, created_at) VALUES (?,?,?,?,?)"
    )
    assert "ON CONFLICT (source_id, target_id) DO UPDATE" in sql
    assert "$5" in sql


def test_insert_or_ignore() -> None:
    sql = translate_sqlite_sql("INSERT OR IGNORE INTO memory_entity_edges VALUES (?,?,?,?)")
    assert "ON CONFLICT DO NOTHING" in sql
    assert "$4" in sql


def test_split_script() -> None:
    assert split_script("A;\nB;\n") == ["A", "B"]


def test_postgres_url() -> None:
    assert is_postgres_url("postgresql://localhost/db")
    assert is_postgres_url("postgres://localhost/db")
    assert not is_postgres_url("sqlite:///x.db")
    assert normalize_postgres_url("postgresql+asyncpg://h/db") == "postgresql://h/db"


def test_external_postgres_pool_rejects_sqlite() -> None:
    with pytest.raises(ConfigError, match="requires a PostgreSQL"):
        open_store("sqlite:///:memory:", postgres_pool=object())


@pytest.mark.asyncio
async def test_postgres_store_does_not_close_external_pool() -> None:
    class FakePool:
        def __init__(self) -> None:
            self.closed = 0

        async def close(self) -> None:
            self.closed += 1

    pool = FakePool()
    store = PostgresStore("postgresql://example/contextdb", pool=pool)
    await store.close()
    assert pool.closed == 0


@pytest.mark.asyncio
async def test_list_by_entities_filters_and_copies_only_the_requested_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = PostgresStore(
        "postgresql://example/contextdb",
        tenant_id="tenant-1",
        agent_id="agent-1",
        pool=object(),
    )
    now = datetime.now(tz=timezone.utc)
    entity_b = [
        MemoryItem(
            content=f"b-{index}",
            entity_key="entity-b",
            user_id="user-1",
            tenant_id="tenant-1",
            agent_id="agent-1",
        )
        for index in range(10)
    ]
    entity_a = [
        MemoryItem(
            content=f"a-{index}",
            entity_key="entity-a",
            user_id="user-1",
            tenant_id="tenant-1",
            agent_id="agent-1",
            metadata={"nested": [index]},
        )
        for index in range(20)
    ]
    future = MemoryItem(
        content="future",
        entity_key="entity-a",
        user_id="user-1",
        tenant_id="tenant-1",
        agent_id="agent-1",
        valid_from=now + timedelta(days=1),
    )
    indexed = [*entity_b, future, *entity_a]
    store._index_items = {item.id: item for item in indexed}
    store._index_loaded = True

    original_model_copy = MemoryItem.model_copy
    copies = 0

    def counted_model_copy(
        self: MemoryItem,
        *,
        update: dict[str, Any] | None = None,
        deep: bool = False,
    ) -> MemoryItem:
        nonlocal copies
        copies += 1
        return original_model_copy(self, update=update, deep=deep)

    monkeypatch.setattr(MemoryItem, "model_copy", counted_model_copy)
    results = await store.list_by_entities(
        ["entity-a", "entity-b"],
        user_id="user-1",
        exclude_ids={entity_a[0].id},
        valid_at=now,
        limit=5,
    )

    assert len(results) == 5
    assert copies == 5
    assert all(item.entity_key == "entity-a" for item in results)
    assert entity_a[0].id not in {item.id for item in results}
    results[0].metadata["nested"].append(99)
    source = store._index_items[results[0].id]
    assert source.metadata["nested"] != results[0].metadata["nested"]


@pytest.mark.asyncio
async def test_postgres_embedding_search_can_borrow_read_only_index_items(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = PostgresStore(
        "postgresql://example/contextdb",
        tenant_id="tenant-1",
        agent_id="agent-1",
        pool=object(),
    )
    source = MemoryItem(
        content="Thursday is confirmed",
        embedding=[1.0, 0.0],
        user_id="user-1",
        tenant_id="tenant-1",
        agent_id="agent-1",
        metadata={"nested": ["source"]},
    )
    store._index_items = {source.id: source}

    class FakeIndex:
        def search(
            self,
            _query: object,
            *,
            top_k: int,
            include_ids: set[str],
        ) -> list[tuple[str, float]]:
            assert top_k == 1
            assert include_ids == {source.id}
            return [(source.id, 1.0)]

    async def ensure_index() -> Any:
        return FakeIndex()

    monkeypatch.setattr(store, "_ensure_index", ensure_index)
    copied = await store.search_by_embedding(
        [1.0, 0.0],
        top_k=1,
        user_id="user-1",
    )
    borrowed = await store.search_by_embedding(
        [1.0, 0.0],
        top_k=1,
        user_id="user-1",
        copy_items=False,
    )

    assert copied == [source]
    assert copied[0] is not source
    assert borrowed[0] is source
    copied[0].metadata["nested"].append("caller")
    assert source.metadata == {"nested": ["source"]}


@pytest.mark.asyncio
async def test_bound_adapter_reuses_advisory_transaction_connection() -> None:
    class NoAcquirePool:
        def acquire(self) -> None:
            raise AssertionError("bound execution must not acquire another connection")

    class FakeConnection:
        def __init__(self) -> None:
            self.statements: list[str] = []

        async def fetch(self, sql: str, *params: object) -> list[dict[str, int]]:
            self.statements.append(sql)
            return [{"value": 1}]

        async def execute(self, sql: str, *params: object) -> str:
            self.statements.append(sql)
            return "UPDATE 1"

    adapter = _PgAdapter(NoAcquirePool())
    connection = FakeConnection()
    token = adapter.bind(connection)
    try:
        selected = await adapter.execute("SELECT value FROM example")
        updated = await adapter.execute("UPDATE example SET value = ?", (2,))
    finally:
        adapter.reset(token)

    assert await selected.fetchall() == [{"value": 1}]
    assert updated.rowcount == 1
    assert len(connection.statements) == 2
