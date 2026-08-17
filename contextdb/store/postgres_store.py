"""Postgres implementation of the memory store.

Same schema and visibility rules as :class:`SQLiteStore`. Vectors stay in
the row (``BYTEA``) and are mirrored into the in-process
:class:`VectorIndex` — ``pgvector`` is optional later, not required to
self-host. Graph and audit tables are created through a small SQLite→Postgres
SQL adapter so existing graph code keeps working.

Install: ``pip install 'pycontextdb[postgres]'``.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping
from datetime import datetime, timezone
from typing import Any

import numpy as np
from numpy.typing import NDArray

from contextdb.core.exceptions import MemoryNotFoundError, StorageError
from contextdb.core.models import MemoryItem, MemoryStatus, MemoryType
from contextdb.store.base import BaseStore
from contextdb.store.pg_sql import split_script, translate_sqlite_sql
from contextdb.store.sqlite_store import (
    _TRUST_COLUMNS,
    SCHEMA,
    _embedding_to_blob,
    _passes_filters,
    _row_to_item,
)
from contextdb.store.vector_index import VectorIndex, get_vector_index

_PG_SCHEMA = SCHEMA.replace("BLOB", "BYTEA")


class _PgResult:
    def __init__(self, rows: list[Mapping[str, Any]], rowcount: int = 0) -> None:
        self._rows = rows
        self.rowcount = rowcount

    async def fetchone(self) -> Mapping[str, Any] | None:
        return self._rows[0] if self._rows else None

    async def fetchall(self) -> list[Mapping[str, Any]]:
        return list(self._rows)


class _PgAdapter:
    """Connection-shaped object the graph and audit modules already call."""

    def __init__(self, pool: Any) -> None:
        self._pool = pool

    async def execute(self, sql: str, params: tuple[Any, ...] | list[Any] = ()) -> _PgResult:
        translated = translate_sqlite_sql(sql)
        async with self._pool.acquire() as conn:
            stripped = translated.lstrip().upper()
            if stripped.startswith("SELECT"):
                rows = await conn.fetch(translated, *params)
                return _PgResult([dict(row) for row in rows], rowcount=len(rows))
            status = await conn.execute(translated, *params)
            count = 0
            if isinstance(status, str) and status.split()[-1].isdigit():
                count = int(status.split()[-1])
            return _PgResult([], rowcount=count)

    async def executescript(self, script: str) -> None:
        async with self._pool.acquire() as conn:
            for stmt in split_script(script):
                await conn.execute(translate_sqlite_sql(stmt))

    async def commit(self) -> None:
        return None


class PostgresStore(BaseStore):
    """Async Postgres store with the same scope rules as SQLite."""

    def __init__(
        self,
        storage_url: str,
        user_id: str | None = None,
        tenant_id: str | None = None,
        agent_id: str | None = None,
        vector_index: VectorIndex | None = None,
        embedding_dim: int = 1536,
    ) -> None:
        self._url = storage_url
        self._user_id = user_id
        self._tenant_id = tenant_id
        self._agent_id = agent_id
        self._pool: Any = None
        self._adapter: _PgAdapter | None = None
        self._index: VectorIndex | None = vector_index
        self._embedding_dim = embedding_dim
        self._index_loaded = False
        self._write_lock = asyncio.Lock()
        self._slot_locks: dict[tuple[str, ...], asyncio.Lock] = {}
        self._slot_locks_guard = asyncio.Lock()

    async def initialize(self) -> None:
        if self._pool is not None:
            return
        try:
            import asyncpg
        except ImportError as exc:  # pragma: no cover - optional extra
            raise StorageError(
                "asyncpg is not installed. `pip install 'pycontextdb[postgres]'`."
            ) from exc
        self._pool = await asyncpg.create_pool(self._url, min_size=1, max_size=10)
        self._adapter = _PgAdapter(self._pool)
        async with self._pool.acquire() as conn:
            for stmt in split_script(_PG_SCHEMA):
                await conn.execute(stmt)
            cols = {
                row["column_name"]
                for row in await conn.fetch(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_name = 'memories'"
                )
            }
            for column, ddl in _TRUST_COLUMNS:
                if column not in cols:
                    await conn.execute(
                        f"ALTER TABLE memories ADD COLUMN IF NOT EXISTS {column} {ddl}"
                    )
            if "user_id" not in cols:
                await conn.execute(
                    "ALTER TABLE memories ADD COLUMN IF NOT EXISTS user_id TEXT"
                )

    def _require_conn(self) -> _PgAdapter:
        if self._adapter is None:
            raise StorageError("PostgresStore is not initialized. Call initialize() first.")
        return self._adapter

    def _require_pool(self) -> Any:
        if self._pool is None:
            raise StorageError("PostgresStore is not initialized. Call initialize() first.")
        return self._pool

    def _resolve_scope(self, user_id: str | None) -> str | None:
        if self._user_id is not None:
            return self._user_id
        return user_id

    def _scope_allows(self, row: Mapping[str, Any]) -> bool:
        return (
            (self._user_id is None or row["user_id"] == self._user_id)
            and (self._tenant_id is None or row["tenant_id"] == self._tenant_id)
            and (self._agent_id is None or row["agent_id"] == self._agent_id)
        )

    def _scope_sql(self, user_id: str | None = None) -> tuple[str, list[Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        scope = self._resolve_scope(user_id)
        if scope is not None:
            clauses.append("user_id = ?")
            params.append(scope)
        if self._tenant_id is not None:
            clauses.append("tenant_id = ?")
            params.append(self._tenant_id)
        if self._agent_id is not None:
            clauses.append("agent_id = ?")
            params.append(self._agent_id)
        return " AND ".join(clauses), params

    async def slot_lock(
        self,
        entity_key: str,
        attribute_key: str,
        user_id: str | None = None,
    ) -> asyncio.Lock:
        scope = self._resolve_scope(user_id)
        key = (scope or "", self._tenant_id or "", entity_key, attribute_key)
        async with self._slot_locks_guard:
            lock = self._slot_locks.get(key)
            if lock is None:
                lock = asyncio.Lock()
                self._slot_locks[key] = lock
            return lock

    async def _fetch(
        self, sql: str, params: list[Any] | tuple[Any, ...] = ()
    ) -> list[dict[str, Any]]:
        result = await self._require_conn().execute(sql, list(params))
        return [dict(row) for row in await result.fetchall()]

    async def _execute(self, sql: str, params: list[Any] | tuple[Any, ...] = ()) -> int:
        result = await self._require_conn().execute(sql, list(params))
        return int(result.rowcount)

    async def _ensure_index(self) -> VectorIndex:
        if self._index is None:
            self._index = get_vector_index(self._embedding_dim)
        if not self._index_loaded:
            rows = await self._fetch(
                "SELECT id, embedding, embedding_dim FROM memories "
                "WHERE embedding IS NOT NULL AND status = 'ACTIVE'"
            )
            if rows:
                ids = [row["id"] for row in rows]
                vectors = np.stack(
                    [_pg_embedding(row["embedding"]) for row in rows],
                    axis=0,
                )
                self._index.add(ids, vectors)
            self._index_loaded = True
        return self._index

    async def add(self, item: MemoryItem) -> MemoryItem:
        uid = item.user_id or self._user_id
        if self._user_id is not None and uid is not None and uid != self._user_id:
            raise StorageError("cannot write another user's memory into a scoped store")
        item.user_id = uid
        blob, dim = _embedding_to_blob(item.embedding)
        async with self._write_lock:
            await self._execute(
                """
                INSERT INTO memories (
                    id, content, embedding, embedding_dim, memory_type, source,
                    metadata, user_id, event_time, ingestion_time, pii_annotations,
                    retention_policy, created_at, updated_at, access_count, last_accessed,
                    confidence, status, entity_mentions, tags,
                    epistemic_source, corroboration_count, action_relevant,
                    entity_key, attribute_key, valid_from, valid_until,
                    superseded_by, pending_consolidation, injection_suspect,
                    corroborated_by, confirmed, confirmed_at, write_generation,
                    slot_class, slot_value, negated, tenant_id, agent_id,
                    session_id, pii_shadow, contested
                ) VALUES (
                    ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,
                    ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,
                    ?,?
                )
                """,
                (
                    item.id,
                    item.content,
                    blob,
                    dim,
                    item.memory_type.value,
                    item.source,
                    _json(item.metadata),
                    uid,
                    _iso(item.event_time),
                    item.ingestion_time.isoformat(),
                    _json([a.model_dump(mode="json") for a in item.pii_annotations]),
                    item.retention_policy.model_dump_json() if item.retention_policy else None,
                    item.created_at.isoformat(),
                    item.updated_at.isoformat(),
                    item.access_count,
                    _iso(item.last_accessed),
                    item.confidence,
                    item.status.value,
                    _json(item.entity_mentions),
                    _json(item.tags),
                    item.epistemic_source,
                    item.corroboration_count,
                    int(item.action_relevant),
                    item.entity_key,
                    item.attribute_key,
                    _iso(item.valid_from),
                    _iso(item.valid_until),
                    item.superseded_by,
                    int(item.pending_consolidation),
                    int(item.injection_suspect),
                    _json(item.corroborated_by),
                    int(item.confirmed),
                    _iso(item.confirmed_at),
                    item.write_generation,
                    item.slot_class,
                    item.slot_value,
                    int(item.negated),
                    item.tenant_id or self._tenant_id,
                    item.agent_id or self._agent_id,
                    item.session_id,
                    item.pii_shadow,
                    int(item.contested),
                ),
            )
        if item.embedding is not None:
            index = await self._ensure_index()
            index.add([item.id], np.asarray([item.embedding], dtype=np.float32))
        return item

    async def get(self, memory_id: str) -> MemoryItem | None:
        rows = await self._fetch("SELECT * FROM memories WHERE id = ?", (memory_id,))
        if not rows or not self._scope_allows(rows[0]):
            return None
        item = _row_to_item(_normalize_row(rows[0]))
        now_iso = datetime.now(tz=timezone.utc).isoformat()
        await self._execute(
            "UPDATE memories SET access_count = access_count + 1, last_accessed = ? WHERE id = ?",
            (now_iso, memory_id),
        )
        return item

    async def get_raw(self, memory_id: str) -> MemoryItem | None:
        rows = await self._fetch("SELECT * FROM memories WHERE id = ?", (memory_id,))
        if not rows or not self._scope_allows(rows[0]):
            return None
        return _row_to_item(_normalize_row(rows[0]))

    async def update(self, memory_id: str, **kwargs: object) -> MemoryItem:
        current = await self.get_raw(memory_id)
        if current is None:
            raise MemoryNotFoundError(memory_id)
        sets: list[str] = []
        params: list[Any] = []
        allowed = {
            "content",
            "embedding",
            "metadata",
            "status",
            "source",
            "confidence",
            "pii_annotations",
            "entity_mentions",
            "tags",
            "event_time",
            "memory_type",
            "access_count",
            "last_accessed",
            "epistemic_source",
            "corroboration_count",
            "action_relevant",
            "entity_key",
            "attribute_key",
            "valid_from",
            "valid_until",
            "superseded_by",
            "pending_consolidation",
            "injection_suspect",
            "corroborated_by",
            "confirmed",
            "confirmed_at",
            "write_generation",
            "slot_class",
            "slot_value",
            "negated",
            "tenant_id",
            "agent_id",
            "session_id",
            "pii_shadow",
            "contested",
        }
        unknown = set(kwargs) - allowed
        if unknown:
            raise ValueError(f"Unknown update fields: {unknown}")
        for k, v in kwargs.items():
            if k == "embedding":
                blob, dim = _embedding_to_blob(v)  # type: ignore[arg-type]
                sets.append("embedding = ?")
                sets.append("embedding_dim = ?")
                params.extend([blob, dim])
            elif k in {"metadata", "entity_mentions", "tags", "corroborated_by"}:
                sets.append(f"{k} = ?")
                params.append(_json(v))
            elif k == "pii_annotations":
                assert isinstance(v, list)
                sets.append("pii_annotations = ?")
                params.append(_json([a.model_dump(mode="json") for a in v]))
            elif k == "memory_type":
                sets.append("memory_type = ?")
                params.append(v.value if isinstance(v, MemoryType) else str(v))
            elif k == "status":
                sets.append("status = ?")
                params.append(v.value if isinstance(v, MemoryStatus) else str(v))
            elif k in {"event_time", "last_accessed", "valid_from", "valid_until", "confirmed_at"}:
                sets.append(f"{k} = ?")
                params.append(v.isoformat() if isinstance(v, datetime) else v)
            elif k in {
                "action_relevant",
                "pending_consolidation",
                "injection_suspect",
                "confirmed",
                "negated",
                "contested",
            }:
                sets.append(f"{k} = ?")
                params.append(int(bool(v)))
            else:
                sets.append(f"{k} = ?")
                params.append(v)
        sets.append("updated_at = ?")
        params.append(datetime.now(tz=timezone.utc).isoformat())
        params.append(memory_id)
        async with self._write_lock:
            await self._execute(
                f"UPDATE memories SET {', '.join(sets)} WHERE id = ?", params
            )
        if "embedding" in kwargs:
            index = await self._ensure_index()
            index.remove([memory_id])
            if kwargs["embedding"] is not None:
                index.add(
                    [memory_id],
                    np.asarray([kwargs["embedding"]], dtype=np.float32),
                )
        refreshed = await self.get_raw(memory_id)
        assert refreshed is not None
        return refreshed

    async def delete(self, memory_id: str, hard: bool = False) -> bool:
        rows = await self._fetch(
            "SELECT user_id, tenant_id, agent_id FROM memories WHERE id = ?",
            (memory_id,),
        )
        if not rows or not self._scope_allows(rows[0]):
            return False
        async with self._write_lock:
            if hard:
                await self._execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            else:
                await self._execute(
                    "UPDATE memories SET status = ?, updated_at = ? WHERE id = ?",
                    (
                        MemoryStatus.DELETED.value,
                        datetime.now(tz=timezone.utc).isoformat(),
                        memory_id,
                    ),
                )
        if self._index is not None and self._index_loaded:
            self._index.remove([memory_id])
        return True

    async def search_by_embedding(
        self,
        embedding: list[float],
        top_k: int = 10,
        filters: dict[str, object] | None = None,
        user_id: str | None = None,
    ) -> list[MemoryItem]:
        index = await self._ensure_index()
        query = np.asarray(embedding, dtype=np.float32)
        scope_sql, scope_params = self._scope_sql(user_id)
        raw = index.search(query, top_k=top_k * 3 if (filters or scope_sql) else top_k)
        if not raw:
            return []
        ids = [mid for mid, _ in raw]
        placeholders = ",".join(["?"] * len(ids))
        sql = f"SELECT * FROM memories WHERE id IN ({placeholders})"
        params: list[Any] = list(ids)
        if scope_sql:
            sql += f" AND {scope_sql}"
            params.extend(scope_params)
        rows = await self._fetch(sql, params)
        items_by_id = {row["id"]: _row_to_item(_normalize_row(row)) for row in rows}
        results: list[MemoryItem] = []
        for mid, _ in raw:
            item = items_by_id.get(mid)
            if item is None or item.status != MemoryStatus.ACTIVE:
                continue
            if filters and not _passes_filters(item, filters):
                continue
            results.append(item)
            if len(results) >= top_k:
                break
        return results

    async def list_memories(
        self,
        user_id: str | None = None,
        memory_type: MemoryType | None = None,
        status: MemoryStatus | None = MemoryStatus.ACTIVE,
        limit: int = 100,
        offset: int = 0,
    ) -> list[MemoryItem]:
        clauses: list[str] = []
        params: list[Any] = []
        if status is not None:
            clauses.append("status = ?")
            params.append(status.value)
        scope_sql, scope_params = self._scope_sql(user_id)
        if scope_sql:
            clauses.append(scope_sql)
            params.extend(scope_params)
        if memory_type is not None:
            clauses.append("memory_type = ?")
            params.append(memory_type.value)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.extend([limit, offset])
        rows = await self._fetch(
            f"SELECT * FROM memories {where} ORDER BY created_at DESC LIMIT ? OFFSET ?",
            params,
        )
        return [_row_to_item(_normalize_row(row)) for row in rows]

    async def list_by_slot(
        self,
        entity_key: str,
        attribute_key: str,
        status: MemoryStatus | None = MemoryStatus.ACTIVE,
        user_id: str | None = None,
    ) -> list[MemoryItem]:
        clauses = ["entity_key = ?", "attribute_key = ?"]
        params: list[Any] = [entity_key, attribute_key]
        if status is not None:
            clauses.append("status = ?")
            params.append(status.value)
        scope_sql, scope_params = self._scope_sql(user_id)
        if scope_sql:
            clauses.append(scope_sql)
            params.extend(scope_params)
        rows = await self._fetch(
            f"SELECT * FROM memories WHERE {' AND '.join(clauses)}", params
        )
        return [_row_to_item(_normalize_row(row)) for row in rows]

    async def list_by_entity(
        self,
        entity_key: str,
        status: MemoryStatus | None = MemoryStatus.ACTIVE,
        user_id: str | None = None,
    ) -> list[MemoryItem]:
        clauses = ["entity_key = ?"]
        params: list[Any] = [entity_key]
        if status is not None:
            clauses.append("status = ?")
            params.append(status.value)
        scope_sql, scope_params = self._scope_sql(user_id)
        if scope_sql:
            clauses.append(scope_sql)
            params.extend(scope_params)
        rows = await self._fetch(
            f"SELECT * FROM memories WHERE {' AND '.join(clauses)}", params
        )
        return [_row_to_item(_normalize_row(row)) for row in rows]

    async def list_pending_consolidation(self, limit: int = 100) -> list[MemoryItem]:
        sql = (
            "SELECT * FROM memories WHERE pending_consolidation = 1 "
            "AND status = 'ACTIVE'"
        )
        params: list[Any] = []
        scope_sql, scope_params = self._scope_sql(None)
        if scope_sql:
            sql += f" AND {scope_sql}"
            params.extend(scope_params)
        sql += " ORDER BY created_at ASC LIMIT ?"
        params.append(limit)
        rows = await self._fetch(sql, params)
        return [_row_to_item(_normalize_row(row)) for row in rows]

    async def count(self, user_id: str | None = None) -> int:
        sql = "SELECT COUNT(*) AS n FROM memories WHERE status = 'ACTIVE'"
        params: list[Any] = []
        scope_sql, scope_params = self._scope_sql(user_id)
        if scope_sql:
            sql += f" AND {scope_sql}"
            params.extend(scope_params)
        rows = await self._fetch(sql, params)
        return int(rows[0]["n"]) if rows else 0

    async def count_any_status(self, user_id: str) -> int:
        sql = "SELECT COUNT(*) AS n FROM memories"
        params: list[Any] = []
        scope_sql, scope_params = self._scope_sql(user_id)
        if scope_sql:
            sql += f" WHERE {scope_sql}"
            params.extend(scope_params)
        rows = await self._fetch(sql, params)
        return int(rows[0]["n"]) if rows else 0

    async def index_ids(self) -> set[str]:
        index = await self._ensure_index()
        return set(index.ids())

    async def count_by_type(self, user_id: str | None = None) -> dict[str, int]:
        params: list[Any] = []
        sql = "SELECT memory_type, COUNT(*) AS n FROM memories WHERE status = 'ACTIVE'"
        scope_sql, scope_params = self._scope_sql(user_id)
        if scope_sql:
            sql += f" AND {scope_sql}"
            params.extend(scope_params)
        sql += " GROUP BY memory_type"
        rows = await self._fetch(sql, params)
        counts: dict[str, int] = {mt.value: 0 for mt in MemoryType}
        for row in rows:
            counts[str(row["memory_type"])] = int(row["n"])
        return counts

    async def iter_memories(
        self,
        user_id: str | None = None,
        memory_type: MemoryType | None = None,
        status: MemoryStatus | None = MemoryStatus.ACTIVE,
        batch_size: int = 500,
    ) -> AsyncIterator[MemoryItem]:
        offset = 0
        while True:
            page = await self.list_memories(
                user_id=user_id,
                memory_type=memory_type,
                status=status,
                limit=batch_size,
                offset=offset,
            )
            if not page:
                return
            for item in page:
                yield item
            if len(page) < batch_size:
                return
            offset += batch_size

    async def delete_older_than(
        self,
        iso_cutoff: str,
        user_id: str | None = None,
        hard: bool = True,
    ) -> int:
        params: list[Any] = [iso_cutoff]
        where = "created_at < ?"
        scope_sql, scope_params = self._scope_sql(user_id)
        if scope_sql:
            where += f" AND {scope_sql}"
            params.extend(scope_params)
        async with self._write_lock:
            if hard:
                deleted = await self._execute(f"DELETE FROM memories WHERE {where}", params)
            else:
                deleted = await self._execute(
                    f"UPDATE memories SET status = 'DELETED', updated_at = ? WHERE {where}",
                    [datetime.now(tz=timezone.utc).isoformat(), *params],
                )
        if self._index is not None and self._index_loaded:
            self._index_loaded = False
            self._index = None
        return deleted

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
            self._adapter = None


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _json(value: object) -> str:
    import json

    return json.dumps(value)


def _pg_embedding(blob: Any) -> NDArray[np.float32]:
    if blob is None:
        return np.zeros(0, dtype=np.float32)
    if isinstance(blob, memoryview):
        blob = blob.tobytes()
    return np.frombuffer(blob, dtype=np.float32)


def _normalize_row(row: Mapping[str, Any]) -> dict[str, Any]:
    data = dict(row)
    embedding = data.get("embedding")
    if isinstance(embedding, memoryview):
        data["embedding"] = embedding.tobytes()
    elif isinstance(embedding, bytearray):
        data["embedding"] = bytes(embedding)
    return data
