import pytest

from contextdb.core.exceptions import ConfigError
from contextdb.store.factory import (
    is_postgres_url,
    normalize_postgres_url,
    open_store,
)
from contextdb.store.pg_sql import qmark_to_dollar, split_script, translate_sqlite_sql
from contextdb.store.postgres_store import PostgresStore


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
