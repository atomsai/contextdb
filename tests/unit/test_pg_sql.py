import pytest

from contextdb.core.exceptions import ConfigError
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
