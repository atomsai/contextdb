"""Pick a store implementation from ``storage_url``."""

from __future__ import annotations

from typing import Any

from contextdb.core.exceptions import ConfigError
from contextdb.store.base import BaseStore
from contextdb.store.sqlite_store import SQLiteStore


def is_postgres_url(url: str) -> bool:
    lowered = url.lower()
    return lowered.startswith(("postgres://", "postgresql://", "postgresql+asyncpg://"))


def normalize_postgres_url(url: str) -> str:
    if url.lower().startswith("postgresql+asyncpg://"):
        return "postgresql://" + url.split("://", 1)[1]
    return url


def open_store(
    storage_url: str,
    *,
    user_id: str | None = None,
    tenant_id: str | None = None,
    agent_id: str | None = None,
    embedding_dim: int = 1536,
    postgres_pool: Any | None = None,
) -> BaseStore:
    """Return a SQLite or Postgres store. Postgres needs ``pycontextdb[postgres]``."""
    if is_postgres_url(storage_url):
        try:
            from contextdb.store.postgres_store import PostgresStore
        except ImportError as exc:
            raise ConfigError(
                "Postgres URLs require asyncpg. Install with "
                "`pip install 'pycontextdb[postgres]'`."
            ) from exc
        return PostgresStore(
            storage_url=normalize_postgres_url(storage_url),
            user_id=user_id,
            tenant_id=tenant_id,
            agent_id=agent_id,
            embedding_dim=embedding_dim,
            pool=postgres_pool,
        )
    if postgres_pool is not None:
        raise ConfigError("postgres_pool requires a PostgreSQL storage_url")
    return SQLiteStore(
        storage_url=storage_url,
        user_id=user_id,
        tenant_id=tenant_id,
        agent_id=agent_id,
        embedding_dim=embedding_dim,
    )


def store_kwargs_from_client(client: Any) -> dict[str, Any]:
    return {
        "user_id": client.user_id,
        "tenant_id": client.tenant_id,
        "agent_id": client.agent_id,
    }
