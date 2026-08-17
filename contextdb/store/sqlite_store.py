"""SQLite-backed implementation of :class:`BaseStore`.

Uses ``aiosqlite`` for async SQLite access. Vectors are stored as float32
``BLOB`` columns on the memory row, and mirrored into a :class:`VectorIndex`
for fast similarity search. The index is rebuilt on first use if missing.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import aiosqlite
import numpy as np
from numpy.typing import NDArray

from contextdb.core.exceptions import MemoryNotFoundError, StorageError
from contextdb.core.models import (
    MemoryItem,
    MemoryStatus,
    MemoryType,
    PIIAnnotation,
    RetentionPolicy,
)
from contextdb.store.base import BaseStore
from contextdb.store.vector_index import VectorIndex, get_vector_index

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Mapping

SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    embedding BLOB,
    embedding_dim INTEGER,
    memory_type TEXT NOT NULL DEFAULT 'FACTUAL',
    source TEXT DEFAULT '',
    metadata TEXT DEFAULT '{}',
    user_id TEXT,
    event_time TEXT,
    ingestion_time TEXT NOT NULL,
    pii_annotations TEXT DEFAULT '[]',
    retention_policy TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    access_count INTEGER DEFAULT 0,
    last_accessed TEXT,
    confidence REAL DEFAULT 1.0,
    status TEXT DEFAULT 'ACTIVE',
    entity_mentions TEXT DEFAULT '[]',
    tags TEXT DEFAULT '[]',
    epistemic_source TEXT NOT NULL DEFAULT 'user_stated',
    corroboration_count INTEGER NOT NULL DEFAULT 1,
    action_relevant INTEGER NOT NULL DEFAULT 0,
    entity_key TEXT,
    attribute_key TEXT,
    valid_from TEXT,
    valid_until TEXT,
    superseded_by TEXT,
    pending_consolidation INTEGER NOT NULL DEFAULT 0,
    injection_suspect INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_memories_user_id ON memories(user_id);
CREATE INDEX IF NOT EXISTS idx_memories_status ON memories(status);
CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type);
CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at);
CREATE INDEX IF NOT EXISTS idx_memories_slot ON memories(entity_key, attribute_key);
CREATE INDEX IF NOT EXISTS idx_memories_pending ON memories(pending_consolidation);
"""

# Trust-model columns added after v0.1. Each entry: (column, DDL fragment).
# Applied idempotently by _migrate() for databases created before the trust
# model shipped. Postgres equivalent (run once per environment):
#   ALTER TABLE memories
#     ADD COLUMN IF NOT EXISTS epistemic_source TEXT NOT NULL DEFAULT 'user_stated',
#     ADD COLUMN IF NOT EXISTS corroboration_count INTEGER NOT NULL DEFAULT 1,
#     ADD COLUMN IF NOT EXISTS action_relevant BOOLEAN NOT NULL DEFAULT FALSE,
#     ADD COLUMN IF NOT EXISTS entity_key TEXT,
#     ADD COLUMN IF NOT EXISTS attribute_key TEXT,
#     ADD COLUMN IF NOT EXISTS valid_from TIMESTAMPTZ,
#     ADD COLUMN IF NOT EXISTS valid_until TIMESTAMPTZ,
#     ADD COLUMN IF NOT EXISTS superseded_by TEXT,
#     ADD COLUMN IF NOT EXISTS pending_consolidation BOOLEAN NOT NULL DEFAULT FALSE,
#     ADD COLUMN IF NOT EXISTS injection_suspect BOOLEAN NOT NULL DEFAULT FALSE;
#   CREATE INDEX IF NOT EXISTS idx_memories_slot
#     ON memories(entity_key, attribute_key);
#   CREATE INDEX IF NOT EXISTS idx_memories_pending
#     ON memories(pending_consolidation);
_TRUST_COLUMNS: list[tuple[str, str]] = [
    ("epistemic_source", "TEXT NOT NULL DEFAULT 'user_stated'"),
    ("corroboration_count", "INTEGER NOT NULL DEFAULT 1"),
    ("action_relevant", "INTEGER NOT NULL DEFAULT 0"),
    ("entity_key", "TEXT"),
    ("attribute_key", "TEXT"),
    ("valid_from", "TEXT"),
    ("valid_until", "TEXT"),
    ("superseded_by", "TEXT"),
    ("pending_consolidation", "INTEGER NOT NULL DEFAULT 0"),
    ("injection_suspect", "INTEGER NOT NULL DEFAULT 0"),
]


def _parse_storage_url(url: str) -> str:
    """Extract a filesystem path from a sqlite://[/...] URL."""
    prefix = "sqlite:///"
    if url.startswith(prefix):
        return url[len(prefix) :] or ":memory:"
    if url == "sqlite://:memory:" or url == "sqlite://":
        return ":memory:"
    return url


def _embedding_to_blob(embedding: list[float] | None) -> tuple[bytes | None, int | None]:
    if embedding is None:
        return None, None
    arr = np.asarray(embedding, dtype=np.float32)
    return arr.tobytes(), len(embedding)


def _blob_to_embedding(blob: bytes | None) -> list[float] | None:
    if blob is None:
        return None
    arr: NDArray[np.float32] = np.frombuffer(blob, dtype=np.float32)
    return [float(x) for x in arr]


def _opt_datetime(raw: Any) -> datetime | None:
    return datetime.fromisoformat(raw) if raw else None


def _row_to_item(row: Mapping[str, Any]) -> MemoryItem:
    retention_raw = row["retention_policy"]
    retention = (
        RetentionPolicy.model_validate_json(retention_raw) if retention_raw else None
    )
    pii = [PIIAnnotation.model_validate(a) for a in json.loads(row["pii_annotations"] or "[]")]
    return MemoryItem(
        id=row["id"],
        content=row["content"],
        embedding=_blob_to_embedding(row["embedding"]),
        memory_type=MemoryType(row["memory_type"]),
        source=row["source"] or "",
        metadata=json.loads(row["metadata"] or "{}"),
        event_time=_opt_datetime(row["event_time"]),
        ingestion_time=datetime.fromisoformat(row["ingestion_time"]),
        pii_annotations=pii,
        retention_policy=retention,
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
        access_count=int(row["access_count"]),
        last_accessed=_opt_datetime(row["last_accessed"]),
        confidence=float(row["confidence"]),
        status=MemoryStatus(row["status"]),
        entity_mentions=json.loads(row["entity_mentions"] or "[]"),
        tags=json.loads(row["tags"] or "[]"),
        # Trust-model columns. ``row.get`` guards rows read from a connection
        # whose migration has not run yet (defensive; initialize() migrates).
        epistemic_source=row.get("epistemic_source") or "user_stated",
        corroboration_count=int(row.get("corroboration_count") or 1),
        action_relevant=bool(row.get("action_relevant") or 0),
        entity_key=row.get("entity_key"),
        attribute_key=row.get("attribute_key"),
        valid_from=_opt_datetime(row.get("valid_from")),
        valid_until=_opt_datetime(row.get("valid_until")),
        superseded_by=row.get("superseded_by"),
        pending_consolidation=bool(row.get("pending_consolidation") or 0),
        injection_suspect=bool(row.get("injection_suspect") or 0),
    )


class SQLiteStore(BaseStore):
    """Durable, async SQLite store with in-memory vector index cache.

    Single-process concurrency: writes are serialized through an
    :class:`asyncio.Lock` while reads run freely. WAL journaling is enabled
    so concurrent readers never block a writer. For multi-process access
    use the PostgreSQL backend instead — SQLite's file-level locking does
    not guarantee safety across processes even with WAL.
    """

    def __init__(
        self,
        storage_url: str = "sqlite:///contextdb.db",
        user_id: str | None = None,
        vector_index: VectorIndex | None = None,
        embedding_dim: int = 1536,
    ) -> None:
        self._path = _parse_storage_url(storage_url)
        self._user_id = user_id
        self._conn: aiosqlite.Connection | None = None
        self._index: VectorIndex | None = vector_index
        self._embedding_dim = embedding_dim
        self._index_loaded = False
        self._write_lock: asyncio.Lock = asyncio.Lock()

    async def initialize(self) -> None:
        if self._conn is not None:
            return
        self._conn = await aiosqlite.connect(self._path)
        self._conn.row_factory = aiosqlite.Row
        # WAL lets readers proceed while a writer holds the reserved lock; the
        # busy timeout absorbs short contention windows instead of raising
        # SQLITE_BUSY. Both are safe to re-execute on reconnect.
        await self._conn.execute("PRAGMA journal_mode=WAL")
        await self._conn.execute("PRAGMA busy_timeout=5000")
        await self._conn.executescript(SCHEMA)
        await self._migrate(self._conn)
        await self._conn.commit()

    @staticmethod
    async def _migrate(conn: aiosqlite.Connection) -> None:
        """Idempotently add trust-model columns to pre-existing databases.

        SQLite has no ``ADD COLUMN IF NOT EXISTS``, so introspect first.
        Every column carries a default, so old rows read back exactly as
        the trust model expects (user_stated / corroboration 1 / current).
        """
        cursor = await conn.execute("PRAGMA table_info(memories)")
        existing = {row["name"] for row in await cursor.fetchall()}
        for column, ddl in _TRUST_COLUMNS:
            if column not in existing:
                await conn.execute(f"ALTER TABLE memories ADD COLUMN {column} {ddl}")

    def _require_conn(self) -> aiosqlite.Connection:
        if self._conn is None:
            raise StorageError("SQLiteStore is not initialized. Call initialize() first.")
        return self._conn

    async def _ensure_index(self) -> VectorIndex:
        if self._index is None:
            self._index = get_vector_index(self._embedding_dim)
        if not self._index_loaded:
            conn = self._require_conn()
            cursor = await conn.execute(
                "SELECT id, embedding, embedding_dim FROM memories "
                "WHERE embedding IS NOT NULL AND status = 'ACTIVE'"
            )
            rows = await cursor.fetchall()
            if rows:
                ids = [row["id"] for row in rows]
                vectors = np.stack(
                    [np.frombuffer(row["embedding"], dtype=np.float32) for row in rows],
                    axis=0,
                )
                self._index.add(ids, vectors)
            self._index_loaded = True
        return self._index

    async def add(self, item: MemoryItem) -> MemoryItem:
        conn = self._require_conn()
        blob, dim = _embedding_to_blob(item.embedding)
        async with self._write_lock:
            await conn.execute(
                """
                INSERT INTO memories (
                    id, content, embedding, embedding_dim, memory_type, source,
                    metadata, user_id, event_time, ingestion_time, pii_annotations,
                    retention_policy, created_at, updated_at, access_count, last_accessed,
                    confidence, status, entity_mentions, tags,
                    epistemic_source, corroboration_count, action_relevant,
                    entity_key, attribute_key, valid_from, valid_until,
                    superseded_by, pending_consolidation, injection_suspect
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    item.id,
                    item.content,
                    blob,
                    dim,
                    item.memory_type.value,
                    item.source,
                    json.dumps(item.metadata),
                    self._user_id,
                    item.event_time.isoformat() if item.event_time else None,
                    item.ingestion_time.isoformat(),
                    json.dumps([a.model_dump(mode="json") for a in item.pii_annotations]),
                    item.retention_policy.model_dump_json() if item.retention_policy else None,
                    item.created_at.isoformat(),
                    item.updated_at.isoformat(),
                    item.access_count,
                    item.last_accessed.isoformat() if item.last_accessed else None,
                    item.confidence,
                    item.status.value,
                    json.dumps(item.entity_mentions),
                    json.dumps(item.tags),
                    item.epistemic_source,
                    item.corroboration_count,
                    int(item.action_relevant),
                    item.entity_key,
                    item.attribute_key,
                    item.valid_from.isoformat() if item.valid_from else None,
                    item.valid_until.isoformat() if item.valid_until else None,
                    item.superseded_by,
                    int(item.pending_consolidation),
                    int(item.injection_suspect),
                ),
            )
            await conn.commit()
        if item.embedding is not None:
            index = await self._ensure_index()
            index.add([item.id], np.asarray([item.embedding], dtype=np.float32))
        return item

    async def get(self, memory_id: str) -> MemoryItem | None:
        conn = self._require_conn()
        cursor = await conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,))
        row = await cursor.fetchone()
        if row is None:
            return None
        item = _row_to_item(dict(row))
        now_iso = datetime.now(tz=item.ingestion_time.tzinfo).isoformat()
        await conn.execute(
            "UPDATE memories SET access_count = access_count + 1, last_accessed = ? WHERE id = ?",
            (now_iso, memory_id),
        )
        await conn.commit()
        return item

    async def update(self, memory_id: str, **kwargs: object) -> MemoryItem:
        conn = self._require_conn()
        current = await self.get_raw(memory_id)
        if current is None:
            raise MemoryNotFoundError(memory_id)

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
        }
        unknown = set(kwargs) - allowed
        if unknown:
            raise ValueError(f"Unknown update fields: {unknown}")

        sets: list[str] = []
        params: list[Any] = []
        for k, v in kwargs.items():
            if k == "embedding":
                blob, dim = _embedding_to_blob(v)  # type: ignore[arg-type]
                sets.append("embedding = ?")
                sets.append("embedding_dim = ?")
                params.extend([blob, dim])
            elif k == "metadata":
                sets.append("metadata = ?")
                params.append(json.dumps(v))
            elif k == "pii_annotations":
                sets.append("pii_annotations = ?")
                assert isinstance(v, list)
                params.append(
                    json.dumps([a.model_dump(mode="json") for a in v])
                )
            elif k == "entity_mentions" or k == "tags":
                sets.append(f"{k} = ?")
                params.append(json.dumps(v))
            elif k == "memory_type":
                sets.append("memory_type = ?")
                params.append(v.value if isinstance(v, MemoryType) else str(v))
            elif k == "status":
                sets.append("status = ?")
                params.append(v.value if isinstance(v, MemoryStatus) else str(v))
            elif k in {"event_time", "last_accessed", "valid_from", "valid_until"}:
                sets.append(f"{k} = ?")
                params.append(v.isoformat() if isinstance(v, datetime) else v)
            elif k in {"action_relevant", "pending_consolidation", "injection_suspect"}:
                sets.append(f"{k} = ?")
                params.append(int(bool(v)))
            else:
                sets.append(f"{k} = ?")
                params.append(v)

        sets.append("updated_at = ?")
        now = datetime.now(tz=current.updated_at.tzinfo)
        params.append(now.isoformat())
        params.append(memory_id)

        async with self._write_lock:
            await conn.execute(
                f"UPDATE memories SET {', '.join(sets)} WHERE id = ?", params
            )
            await conn.commit()

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

    async def get_raw(self, memory_id: str) -> MemoryItem | None:
        """Fetch without side effects (no access counter bump)."""
        conn = self._require_conn()
        cursor = await conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,))
        row = await cursor.fetchone()
        return _row_to_item(dict(row)) if row else None

    async def delete(self, memory_id: str, hard: bool = False) -> None:
        conn = self._require_conn()
        async with self._write_lock:
            if hard:
                await conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            else:
                now = datetime.now(tz=timezone.utc).isoformat()
                await conn.execute(
                    "UPDATE memories SET status = ?, updated_at = ? WHERE id = ?",
                    (MemoryStatus.DELETED.value, now, memory_id),
                )
            await conn.commit()
        if self._index is not None and self._index_loaded:
            self._index.remove([memory_id])

    async def search_by_embedding(
        self,
        embedding: list[float],
        top_k: int = 10,
        filters: dict[str, object] | None = None,
    ) -> list[MemoryItem]:
        conn = self._require_conn()
        index = await self._ensure_index()
        query = np.asarray(embedding, dtype=np.float32)
        # Fetch extra to allow for filter culling.
        raw = index.search(query, top_k=top_k * 3 if filters else top_k)
        if not raw:
            return []

        ids = [mid for mid, _ in raw]
        placeholders = ",".join(["?"] * len(ids))
        cursor = await conn.execute(
            f"SELECT * FROM memories WHERE id IN ({placeholders})", ids
        )
        rows = await cursor.fetchall()
        items_by_id = {row["id"]: _row_to_item(dict(row)) for row in rows}

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
        """List memories with optional filters.

        ``status=None`` disables the status filter (all lifecycle states) —
        used by verifiable forgetting, which must reach archived/derived rows.
        """
        conn = self._require_conn()
        clauses: list[str] = []
        params: list[Any] = []
        if status is not None:
            clauses.append("status = ?")
            params.append(status.value)
        if user_id is not None:
            clauses.append("user_id = ?")
            params.append(user_id)
        if memory_type is not None:
            clauses.append("memory_type = ?")
            params.append(memory_type.value)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.extend([limit, offset])
        cursor = await conn.execute(
            f"SELECT * FROM memories {where} "
            "ORDER BY created_at DESC LIMIT ? OFFSET ?",
            params,
        )
        rows = await cursor.fetchall()
        return [_row_to_item(dict(row)) for row in rows]

    async def list_by_slot(
        self,
        entity_key: str,
        attribute_key: str,
        status: MemoryStatus | None = MemoryStatus.ACTIVE,
    ) -> list[MemoryItem]:
        """All memories occupying the same entity+attribute slot.

        Slot identity — not string similarity — is what dedupe and
        contradiction detection key on.
        """
        conn = self._require_conn()
        clauses = ["entity_key = ?", "attribute_key = ?"]
        params: list[Any] = [entity_key, attribute_key]
        if status is not None:
            clauses.append("status = ?")
            params.append(status.value)
        cursor = await conn.execute(
            f"SELECT * FROM memories WHERE {' AND '.join(clauses)}",
            params,
        )
        rows = await cursor.fetchall()
        return [_row_to_item(dict(row)) for row in rows]

    async def list_pending_consolidation(self, limit: int = 100) -> list[MemoryItem]:
        """Active memories written via ``add_fast`` awaiting consolidation."""
        conn = self._require_conn()
        cursor = await conn.execute(
            "SELECT * FROM memories WHERE pending_consolidation = 1 "
            "AND status = 'ACTIVE' ORDER BY created_at ASC LIMIT ?",
            (limit,),
        )
        rows = await cursor.fetchall()
        return [_row_to_item(dict(row)) for row in rows]

    async def count(self, user_id: str | None = None) -> int:
        conn = self._require_conn()
        if user_id is None:
            cursor = await conn.execute(
                "SELECT COUNT(*) FROM memories WHERE status = 'ACTIVE'"
            )
        else:
            cursor = await conn.execute(
                "SELECT COUNT(*) FROM memories WHERE status = 'ACTIVE' AND user_id = ?",
                (user_id,),
            )
        row = await cursor.fetchone()
        return int(row[0]) if row else 0

    async def count_by_type(self, user_id: str | None = None) -> dict[str, int]:
        """Return active-memory counts bucketed by :class:`MemoryType`.

        Single aggregate SQL query — does not load rows into memory.
        """
        conn = self._require_conn()
        params: list[Any] = []
        sql = "SELECT memory_type, COUNT(*) FROM memories WHERE status = 'ACTIVE'"
        if user_id is not None:
            sql += " AND user_id = ?"
            params.append(user_id)
        sql += " GROUP BY memory_type"
        cursor = await conn.execute(sql, params)
        rows = await cursor.fetchall()
        counts: dict[str, int] = {mt.value: 0 for mt in MemoryType}
        for row in rows:
            counts[str(row[0])] = int(row[1])
        return counts

    async def iter_memories(
        self,
        user_id: str | None = None,
        memory_type: MemoryType | None = None,
        status: MemoryStatus | None = MemoryStatus.ACTIVE,
        batch_size: int = 500,
    ) -> AsyncIterator[MemoryItem]:
        """Stream memories in fixed-size pages to bound peak memory.

        ``status=None`` streams every lifecycle state (active, archived,
        deleted) — required by verifiable forgetting.
        """
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
        """Bulk delete memories with ``created_at < iso_cutoff``.

        Returns the number of affected rows. Runs as a single SQL statement
        rather than load-then-delete, so it stays O(1) in Python memory.
        """
        conn = self._require_conn()
        params: list[Any] = [iso_cutoff]
        where = "created_at < ?"
        if user_id is not None:
            where += " AND user_id = ?"
            params.append(user_id)
        async with self._write_lock:
            if hard:
                cursor = await conn.execute(
                    f"DELETE FROM memories WHERE {where}", params
                )
            else:
                cursor = await conn.execute(
                    f"UPDATE memories SET status = 'DELETED', updated_at = ? "
                    f"WHERE {where}",
                    [datetime.now(tz=timezone.utc).isoformat(), *params],
                )
            await conn.commit()
        # Drop removed ids from the index lazily on next rebuild.
        if self._index is not None and self._index_loaded:
            self._index_loaded = False
            self._index = None
        return int(cursor.rowcount or 0)

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    def _vectors_from_rows(self, rows: list[Mapping[str, Any]]) -> NDArray[np.float32]:
        return np.stack(
            [np.frombuffer(row["embedding"], dtype=np.float32) for row in rows], axis=0
        )


def _passes_filters(item: MemoryItem, filters: dict[str, object]) -> bool:
    for key, value in filters.items():
        if value is None:
            continue
        if key == "memory_type" and item.memory_type.value != value:
            return False
        if key == "status" and item.status.value != value:
            return False
        if key == "source" and item.source != value:
            return False
    return True
