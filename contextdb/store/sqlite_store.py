"""SQLite-backed implementation of :class:`BaseStore`.

Uses ``aiosqlite`` for async SQLite access. Vectors are stored as float32
``BLOB`` columns on the memory row, and mirrored into a :class:`VectorIndex`
for fast similarity search. The index is rebuilt on first use if missing.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import AsyncIterator, Callable, Mapping
from datetime import datetime, timezone
from typing import Any

import aiosqlite
import numpy as np
from numpy.typing import NDArray

from contextdb.core.exceptions import MemoryNotFoundError, StorageError
from contextdb.core.models import (
    MemoryConsistencyToken,
    MemoryItem,
    MemoryStatus,
    MemoryType,
    PIIAnnotation,
    RetentionPolicy,
)
from contextdb.store.base import BaseStore
from contextdb.store.vector_index import VectorIndex, get_vector_index

SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    embedding BLOB,
    embedding_dim INTEGER,
    embedding_model_id TEXT,
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
    injection_suspect INTEGER NOT NULL DEFAULT 0,
    corroborated_by TEXT DEFAULT '[]',
    confirmed INTEGER NOT NULL DEFAULT 0,
    confirmed_at TEXT,
    write_generation INTEGER NOT NULL DEFAULT 0,
    slot_class TEXT,
    slot_value TEXT,
    negated INTEGER NOT NULL DEFAULT 0,
    tenant_id TEXT,
    agent_id TEXT,
    session_id TEXT,
    pii_shadow TEXT,
    contested INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_memories_user_id ON memories(user_id);
CREATE INDEX IF NOT EXISTS idx_memories_status ON memories(status);
CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type);
CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at);
CREATE INDEX IF NOT EXISTS idx_memories_slot ON memories(entity_key, attribute_key);
CREATE INDEX IF NOT EXISTS idx_memories_pending ON memories(pending_consolidation);
CREATE INDEX IF NOT EXISTS idx_memories_tenant ON memories(tenant_id);
CREATE INDEX IF NOT EXISTS idx_memories_session ON memories(session_id);

CREATE TABLE IF NOT EXISTS contextdb_meta (
    key TEXT PRIMARY KEY,
    value TEXT
);
"""

# The process-local vector index mirrors rows owned by the database. Every
# mutating statement bumps this revision; an index whose loaded revision
# differs from the store's is rebuilt before it serves a search. That is
# what keeps two clients over one database (two instances in a process, or
# many processes on Postgres) from serving each other's writes as stale.
_REVISION_KEY = "index_revision"


def scoped_revision_key(
    tenant_id: str | None,
    agent_id: str | None,
) -> str:
    if tenant_id is None and agent_id is None:
        return _REVISION_KEY
    payload = json.dumps(
        [tenant_id, agent_id],
        separators=(",", ":"),
    ).encode()
    return (
        f"{_REVISION_KEY}:project:"
        f"{hashlib.sha256(payload).hexdigest()[:32]}"
    )


_READ_REVISION_SQL = "SELECT value FROM contextdb_meta WHERE key = ?"
_BUMP_REVISION_SQL = (
    "INSERT INTO contextdb_meta (key, value) VALUES (?, '1') "
    "ON CONFLICT(key) DO UPDATE "
    "SET value = CAST(CAST(contextdb_meta.value AS INTEGER) + 1 AS TEXT) "
    "RETURNING contextdb_meta.value"
)
_BUMP_SCOPED_REVISION_SQL = (
    "INSERT INTO contextdb_meta (key, value) VALUES (?, '1'), (?, '1') "
    "ON CONFLICT(key) DO UPDATE "
    "SET value = CAST(CAST(contextdb_meta.value AS INTEGER) + 1 AS TEXT) "
    "RETURNING contextdb_meta.key, contextdb_meta.value"
)

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
    ("corroborated_by", "TEXT DEFAULT '[]'"),
    ("confirmed", "INTEGER NOT NULL DEFAULT 0"),
    ("confirmed_at", "TEXT"),
    ("write_generation", "INTEGER NOT NULL DEFAULT 0"),
    ("slot_class", "TEXT"),
    ("slot_value", "TEXT"),
    ("negated", "INTEGER NOT NULL DEFAULT 0"),
    ("tenant_id", "TEXT"),
    ("agent_id", "TEXT"),
    ("session_id", "TEXT"),
    ("pii_shadow", "TEXT"),
    ("contested", "INTEGER NOT NULL DEFAULT 0"),
    ("embedding_model_id", "TEXT"),
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


async def _fetchone(
    conn: aiosqlite.Connection, sql: str, params: tuple[Any, ...] | list[Any] = ()
) -> Any:
    """fetchone that fully steps the cursor.

    An aiosqlite cursor that has returned its last row but has not been
    stepped to completion keeps the statement "in progress", and a
    concurrent ``commit()`` on the shared connection then fails with
    ``OperationalError: cannot commit transaction``. Exhausting the
    cursor avoids that.
    """
    cursor = await conn.execute(sql, params)
    rows = await cursor.fetchall()
    return next(iter(rows), None)


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
        embedding_model_id=row["embedding_model_id"],
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
        corroborated_by=json.loads(row.get("corroborated_by") or "[]"),
        confirmed=bool(row.get("confirmed") or 0),
        confirmed_at=_opt_datetime(row.get("confirmed_at")),
        write_generation=int(row.get("write_generation") or 0),
        slot_class=row.get("slot_class"),
        slot_value=row.get("slot_value"),
        negated=bool(row.get("negated") or 0),
        tenant_id=row.get("tenant_id"),
        agent_id=row.get("agent_id"),
        session_id=row.get("session_id"),
        pii_shadow=row.get("pii_shadow"),
        contested=bool(row.get("contested") or 0),
        user_id=row.get("user_id"),
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
        tenant_id: str | None = None,
        agent_id: str | None = None,
        vector_index: VectorIndex | None = None,
        embedding_dim: int = 1536,
        embedding_model_id: str | None = None,
    ) -> None:
        self._path = _parse_storage_url(storage_url)
        self._user_id = user_id
        self._tenant_id = tenant_id
        self._agent_id = agent_id
        self._revision_key = scoped_revision_key(tenant_id, agent_id)
        self._conn: aiosqlite.Connection | None = None
        self._index: VectorIndex | None = vector_index
        self._embedding_dim = embedding_dim
        self._embedding_model_id = embedding_model_id
        self._index_loaded = False
        self._loaded_revision: int | None = None
        self._write_lock: asyncio.Lock = asyncio.Lock()
        # Per-slot locks: two concurrent corrections into the same slot
        # must not both read the old occupant and both write.
        self._slot_locks: dict[tuple[str, ...], asyncio.Lock] = {}
        self._slot_locks_guard = asyncio.Lock()
        # Serializes hash-chain appends from the audit log (in-process;
        # SQLite is documented as a single-process store).
        self._audit_append_lock: asyncio.Lock = asyncio.Lock()

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
        if self._embedding_model_id is not None:
            await self._conn.execute(
                "UPDATE memories SET embedding_model_id = ? "
                "WHERE embedding_model_id IS NULL AND embedding_dim = ?",
                (
                    self._embedding_model_id,
                    self._embedding_dim,
                ),
            )
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

    def _resolve_scope(self, user_id: str | None) -> str | None:
        """Tenant confinement: a scoped store can ONLY see its own user.

        Scope is fixed at construction (``ContextDB(user_id=...)``) and
        cannot be widened by per-call parameters — that is what makes
        cross-tenant bleed structurally impossible rather than
        conventionally avoided. An unscoped store (``user_id=None``) is an
        admin/global context and may filter by an explicit ``user_id``.
        """
        if self._user_id is not None:
            return self._user_id
        return user_id

    def _scope_allows(self, row: Any) -> bool:
        return (
            (self._user_id is None or row["user_id"] == self._user_id)
            and (self._tenant_id is None or row["tenant_id"] == self._tenant_id)
            and (self._agent_id is None or row["agent_id"] == self._agent_id)
        )

    def _scope_sql(self, user_id: str | None = None) -> tuple[str, list[Any]]:
        """AND-joined visibility predicates + bind params."""
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
        """Lock keyed on (user, tenant, entity, attribute) — not the row.

        Two concurrent "actually 4pm" / "actually 5pm" writes into the
        same slot must serialize. SQLite serializes via this in-process
        lock plus the global write lock; it is documented as a
        single-process store (multi-process hosts use the Postgres
        backend, whose slot lock is a cross-process advisory lock).
        """
        scope = self._resolve_scope(user_id)
        key = (scope or "", self._tenant_id or "", entity_key, attribute_key)
        async with self._slot_locks_guard:
            lock = self._slot_locks.get(key)
            if lock is None:
                lock = asyncio.Lock()
                self._slot_locks[key] = lock
            return lock

    def audit_lock(self) -> asyncio.Lock:
        """In-process serialization for hash-chain appends."""
        return self._audit_append_lock

    async def _read_revision(self) -> int:
        row = await _fetchone(
            self._require_conn(),
            _READ_REVISION_SQL,
            (self._revision_key,),
        )
        return int(row["value"]) if row else 0

    async def consistency_token(self) -> MemoryConsistencyToken:
        return MemoryConsistencyToken(
            memory_version=await self._read_revision()
        )

    async def _bump_revision(self, conn: aiosqlite.Connection) -> int:
        """Atomically increment the global store revision and return it.

        Called inside the write lock, in the same commit as the mutation,
        so a revision the index has not seen always means "someone wrote".

        ``execute_fetchall`` runs execute+drain as one worker-thread call:
        a RETURNING statement that stays active across an await boundary
        makes a concurrent ``commit()`` on this shared connection fail
        with "cannot commit transaction - SQL statements in progress".
        """
        if self._revision_key != _REVISION_KEY:
            rows = await conn.execute_fetchall(
                _BUMP_SCOPED_REVISION_SQL,
                (self._revision_key, _REVISION_KEY),
            )
            row = next(
                (
                    candidate
                    for candidate in rows
                    if candidate["key"] == self._revision_key
                ),
                None,
            )
        else:
            rows = await conn.execute_fetchall(
                _BUMP_REVISION_SQL,
                (self._revision_key,),
            )
            row = next(iter(rows), None)
        assert row is not None
        return int(row["value"])

    async def _ensure_index(self) -> VectorIndex:
        if self._index is None:
            self._index = get_vector_index(self._embedding_dim)
        revision = await self._read_revision()
        if self._index_loaded and self._loaded_revision == revision:
            return self._index
        async with self._write_lock:
            # Re-check under the lock: a writer may have synced meanwhile.
            revision = await self._read_revision()
            if self._index_loaded and self._loaded_revision == revision:
                return self._index
            await self._rebuild_index(revision)
        return self._index

    async def _rebuild_index(self, revision: int) -> None:
        """Reload the process-local index from the authoritative rows.

        Caller must hold ``_write_lock``. The snapshot reflects every
        write up to and including ``revision`` — including writes made by
        other instances or processes sharing this database.
        """
        assert self._index is not None
        conn = self._require_conn()
        self._index.remove(self._index.ids())
        sql = (
            "SELECT id, embedding, embedding_dim FROM memories "
            "WHERE embedding IS NOT NULL AND status = 'ACTIVE' "
            "AND embedding_dim = ?"
        )
        params: list[Any] = [self._embedding_dim]
        if self._embedding_model_id is not None:
            sql += " AND embedding_model_id = ?"
            params.append(self._embedding_model_id)
        cursor = await conn.execute(
            sql,
            params,
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
        self._loaded_revision = revision

    def _sync_index_after_write(
        self,
        revision: int,
        delta: Callable[[VectorIndex], None] | None = None,
    ) -> None:
        """Reconcile the process-local index with a just-committed write.

        Caller must hold ``_write_lock``; the write must already have
        bumped the store revision to ``revision``. A warm index that was
        coherent at ``revision - 1`` takes the incremental ``delta``.
        Anything else — a cold index, a bulk mutation with no cheap delta
        (``delta=None``), or an interleaved write from another instance or
        process — invalidates the snapshot so the next read rebuilds from
        the authoritative rows (which include this write).
        """
        if self._index is None:
            self._index = get_vector_index(self._embedding_dim)
        if not self._index_loaded:
            return
        if delta is not None and self._loaded_revision == revision - 1:
            delta(self._index)
            self._loaded_revision = revision
            return
        self._index_loaded = False
        self._loaded_revision = None

    async def add(self, item: MemoryItem) -> MemoryItem:
        conn = self._require_conn()
        uid = item.user_id or self._user_id
        if self._user_id is not None and uid is not None and uid != self._user_id:
            raise StorageError("cannot write another user's memory into a scoped store")
        item.user_id = uid
        if item.embedding is not None and self._embedding_model_id is not None:
            if (
                item.embedding_model_id is not None
                and item.embedding_model_id != self._embedding_model_id
            ):
                raise StorageError("memory embedding model does not match store")
            item.embedding_model_id = self._embedding_model_id
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
                    superseded_by, pending_consolidation, injection_suspect,
                    corroborated_by, confirmed, confirmed_at, write_generation,
                    slot_class, slot_value, negated, tenant_id, agent_id,
                    session_id, pii_shadow, contested, embedding_model_id
                ) VALUES (
                    ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,
                    ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,
                    ?,?,?
                )
                """,
                (
                    item.id,
                    item.content,
                    blob,
                    dim,
                    item.memory_type.value,
                    item.source,
                    json.dumps(item.metadata),
                    uid,
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
                    json.dumps(item.corroborated_by),
                    int(item.confirmed),
                    item.confirmed_at.isoformat() if item.confirmed_at else None,
                    item.write_generation,
                    item.slot_class,
                    item.slot_value,
                    int(item.negated),
                    item.tenant_id or self._tenant_id,
                    item.agent_id or self._agent_id,
                    item.session_id,
                    item.pii_shadow,
                    int(item.contested),
                    item.embedding_model_id,
                ),
            )
            revision = await self._bump_revision(conn)
            await conn.commit()
            if item.embedding is not None:
                embedding = item.embedding

                def _delta(index: VectorIndex) -> None:
                    index.add([item.id], np.asarray([embedding], dtype=np.float32))

                self._sync_index_after_write(revision, _delta)
            else:
                self._sync_index_after_write(revision)
        return item

    async def get(self, memory_id: str) -> MemoryItem | None:
        conn = self._require_conn()
        row = await _fetchone(conn, "SELECT * FROM memories WHERE id = ?", (memory_id,))
        if row is None or not self._scope_allows(row):
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
        if "embedding" in kwargs and self._embedding_model_id is not None:
            kwargs["embedding_model_id"] = self._embedding_model_id

        allowed = {
            "content",
            "embedding",
            "embedding_model_id",
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
            elif k == "corroborated_by":
                sets.append("corroborated_by = ?")
                params.append(json.dumps(v))
            elif k == "confirmed_at":
                sets.append("confirmed_at = ?")
                params.append(v.isoformat() if isinstance(v, datetime) else v)
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
            revision = await self._bump_revision(conn)
            await conn.commit()
            if "embedding" in kwargs:
                new_embedding = kwargs["embedding"]

                def _delta(index: VectorIndex) -> None:
                    index.remove([memory_id])
                    if new_embedding is not None:
                        index.add(
                            [memory_id],
                            np.asarray([new_embedding], dtype=np.float32),
                        )

                self._sync_index_after_write(revision, _delta)
            else:
                self._sync_index_after_write(revision)

        refreshed = await self.get_raw(memory_id)
        assert refreshed is not None
        return refreshed

    async def get_raw(self, memory_id: str) -> MemoryItem | None:
        """Fetch without side effects (no access counter bump)."""
        row = await _fetchone(
            self._require_conn(), "SELECT * FROM memories WHERE id = ?", (memory_id,)
        )
        if row is None or not self._scope_allows(row):
            return None
        return _row_to_item(dict(row))

    async def delete(self, memory_id: str, hard: bool = False) -> bool:
        """Delete a memory. Returns False for missing or out-of-scope ids —
        callers must not audit-log a deletion that did not happen."""
        conn = self._require_conn()
        row = await _fetchone(
            conn,
            "SELECT user_id, tenant_id, agent_id FROM memories WHERE id = ?",
            (memory_id,),
        )
        if row is None or not self._scope_allows(row):
            return False
        async with self._write_lock:
            if hard:
                await conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            else:
                now = datetime.now(tz=timezone.utc).isoformat()
                await conn.execute(
                    "UPDATE memories SET status = ?, updated_at = ? WHERE id = ?",
                    (MemoryStatus.DELETED.value, now, memory_id),
                )
            revision = await self._bump_revision(conn)
            await conn.commit()

            def _delta(index: VectorIndex) -> None:
                index.remove([memory_id])

            self._sync_index_after_write(revision, _delta)
        return True

    async def search_by_embedding(
        self,
        embedding: list[float],
        top_k: int = 10,
        filters: dict[str, object] | None = None,
        user_id: str | None = None,
    ) -> list[MemoryItem]:
        conn = self._require_conn()
        index = await self._ensure_index()
        query = np.asarray(embedding, dtype=np.float32)
        scope_sql, scope_params = self._scope_sql(user_id)
        # Scope candidates BEFORE ranking: the allowlist comes from the
        # authoritative SQL scope, so out-of-scope vectors never compete
        # for the candidate budget (and a large foreign tenant cannot
        # starve an in-scope hit).
        allow_sql = (
            "SELECT id FROM memories WHERE status = 'ACTIVE' AND embedding IS NOT NULL"
        )
        if scope_sql:
            allow_sql += f" AND {scope_sql}"
        cursor = await conn.execute(allow_sql, scope_params)
        candidate_ids = {row["id"] for row in await cursor.fetchall()}
        if not candidate_ids:
            return []
        # Fetch extra only to allow for post-filter culling.
        raw = index.search(
            query,
            top_k=top_k * 3 if filters else top_k,
            include_ids=candidate_ids,
        )
        if not raw:
            return []

        ids = [mid for mid, _ in raw]
        placeholders = ",".join(["?"] * len(ids))
        sql = f"SELECT * FROM memories WHERE id IN ({placeholders})"
        params: list[Any] = list(ids)
        if scope_sql:
            sql += f" AND {scope_sql}"
            params.extend(scope_params)
        cursor = await conn.execute(sql, params)
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
        scope_sql, scope_params = self._scope_sql(user_id)
        if scope_sql:
            clauses.append(scope_sql)
            params.extend(scope_params)
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
        user_id: str | None = None,
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
        scope_sql, scope_params = self._scope_sql(user_id)
        if scope_sql:
            clauses.append(scope_sql)
            params.extend(scope_params)
        cursor = await conn.execute(
            f"SELECT * FROM memories WHERE {' AND '.join(clauses)}",
            params,
        )
        rows = await cursor.fetchall()
        return [_row_to_item(dict(row)) for row in rows]

    async def list_by_entity(
        self,
        entity_key: str,
        status: MemoryStatus | None = MemoryStatus.ACTIVE,
        user_id: str | None = None,
    ) -> list[MemoryItem]:
        """Current-family slots for one entity — the compositional hop."""
        conn = self._require_conn()
        clauses = ["entity_key = ?"]
        params: list[Any] = [entity_key]
        if status is not None:
            clauses.append("status = ?")
            params.append(status.value)
        scope_sql, scope_params = self._scope_sql(user_id)
        if scope_sql:
            clauses.append(scope_sql)
            params.extend(scope_params)
        cursor = await conn.execute(
            f"SELECT * FROM memories WHERE {' AND '.join(clauses)}",
            params,
        )
        rows = await cursor.fetchall()
        return [_row_to_item(dict(row)) for row in rows]

    async def list_pending_consolidation(self, limit: int = 100) -> list[MemoryItem]:
        """Active memories written via ``add_fast`` awaiting consolidation."""
        conn = self._require_conn()
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
        cursor = await conn.execute(sql, params)
        rows = await cursor.fetchall()
        return [_row_to_item(dict(row)) for row in rows]

    async def count(self, user_id: str | None = None) -> int:
        conn = self._require_conn()
        sql = "SELECT COUNT(*) FROM memories WHERE status = 'ACTIVE'"
        params: list[Any] = []
        scope_sql, scope_params = self._scope_sql(user_id)
        if scope_sql:
            sql += f" AND {scope_sql}"
            params.extend(scope_params)
        row = await _fetchone(conn, sql, params)
        return int(row[0]) if row else 0

    async def count_any_status(self, user_id: str) -> int:
        """Rows for a user across ALL lifecycle states — forgetting residue."""
        conn = self._require_conn()
        sql = "SELECT COUNT(*) FROM memories"
        params: list[Any] = []
        scope_sql, scope_params = self._scope_sql(user_id)
        if scope_sql:
            sql += f" WHERE {scope_sql}"
            params.extend(scope_params)
        row = await _fetchone(conn, sql, params)
        return int(row[0]) if row else 0

    async def index_ids(self) -> set[str]:
        """Live ids in the vector index — for deletion residue checks."""
        index = await self._ensure_index()
        return set(index.ids())

    async def count_by_type(self, user_id: str | None = None) -> dict[str, int]:
        """Return active-memory counts bucketed by :class:`MemoryType`.

        Single aggregate SQL query — does not load rows into memory.
        """
        conn = self._require_conn()
        params: list[Any] = []
        sql = "SELECT memory_type, COUNT(*) FROM memories WHERE status = 'ACTIVE'"
        scope_sql, scope_params = self._scope_sql(user_id)
        if scope_sql:
            sql += f" AND {scope_sql}"
            params.extend(scope_params)
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
        scope_sql, scope_params = self._scope_sql(user_id)
        if scope_sql:
            where += f" AND {scope_sql}"
            params.extend(scope_params)
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
            revision = await self._bump_revision(conn)
            await conn.commit()
            # Bulk mutation has no cheap delta: the snapshot is invalidated
            # and the next read rebuilds at the current revision.
            self._sync_index_after_write(revision)
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
