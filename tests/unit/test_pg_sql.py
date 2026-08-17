from contextdb.store.factory import is_postgres_url, normalize_postgres_url
from contextdb.store.pg_sql import qmark_to_dollar, split_script, translate_sqlite_sql


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
