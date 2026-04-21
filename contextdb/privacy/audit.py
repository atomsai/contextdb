"""Hash-chained audit trail for every memory-touching operation.

Each :class:`AuditEntry` carries the SHA-256 hash of the previous entry, so
tampering with any record invalidates the chain downstream. This gives us
tamper-evidence without a signing authority — enough for internal compliance
and most external audits short of regulated industries (which should add a
signing key on top).

Verifying the chain is O(n) in entry count but reads are sequential and
bounded by storage. For production workloads, consider periodic anchoring
to an external log (e.g., a ledger) — not in scope for v0.1.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from contextdb.store.sqlite_store import SQLiteStore

_SCHEMA = """
CREATE TABLE IF NOT EXISTS audit_log (
    id TEXT PRIMARY KEY,
    sequence INTEGER NOT NULL,
    operation TEXT NOT NULL,
    memory_id TEXT,
    user_id TEXT,
    details TEXT DEFAULT '{}',
    previous_hash TEXT NOT NULL,
    entry_hash TEXT NOT NULL,
    timestamp TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_audit_sequence ON audit_log(sequence);
CREATE INDEX IF NOT EXISTS idx_audit_memory ON audit_log(memory_id);
CREATE INDEX IF NOT EXISTS idx_audit_user ON audit_log(user_id);
"""

_GENESIS_HASH = "0" * 64


class AuditEntry(BaseModel):
    """A single record in the hash-chained audit log."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    sequence: int
    operation: str
    memory_id: str | None = None
    user_id: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)
    previous_hash: str
    entry_hash: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))

    def canonical_payload(self) -> str:
        return json.dumps(
            {
                "id": self.id,
                "sequence": self.sequence,
                "operation": self.operation,
                "memory_id": self.memory_id,
                "user_id": self.user_id,
                "details": self.details,
                "previous_hash": self.previous_hash,
                "timestamp": self.timestamp.isoformat(),
            },
            sort_keys=True,
            separators=(",", ":"),
        )


def _compute_hash(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class AuditLogger:
    """Append-only audit log with per-entry SHA-256 chaining."""

    def __init__(self, store: SQLiteStore) -> None:
        self.store = store

    async def initialize(self) -> None:
        conn = self.store._require_conn()
        await conn.executescript(_SCHEMA)
        await conn.commit()

    async def log(
        self,
        operation: str,
        memory_id: str | None = None,
        user_id: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> AuditEntry:
        conn = self.store._require_conn()
        cursor = await conn.execute(
            "SELECT sequence, entry_hash FROM audit_log ORDER BY sequence DESC LIMIT 1"
        )
        row = await cursor.fetchone()
        sequence = (row["sequence"] + 1) if row else 1
        previous_hash = row["entry_hash"] if row else _GENESIS_HASH

        entry = AuditEntry(
            sequence=sequence,
            operation=operation,
            memory_id=memory_id,
            user_id=user_id,
            details=details or {},
            previous_hash=previous_hash,
            entry_hash="",
        )
        entry.entry_hash = _compute_hash(entry.canonical_payload())

        await conn.execute(
            "INSERT INTO audit_log "
            "(id, sequence, operation, memory_id, user_id, details, "
            "previous_hash, entry_hash, timestamp) VALUES (?,?,?,?,?,?,?,?,?)",
            (
                entry.id,
                entry.sequence,
                entry.operation,
                entry.memory_id,
                entry.user_id,
                json.dumps(entry.details),
                entry.previous_hash,
                entry.entry_hash,
                entry.timestamp.isoformat(),
            ),
        )
        await conn.commit()
        return entry

    async def get_history(
        self,
        memory_id: str | None = None,
        user_id: str | None = None,
        limit: int = 1000,
    ) -> list[AuditEntry]:
        conn = self.store._require_conn()
        clauses: list[str] = []
        params: list[Any] = []
        if memory_id is not None:
            clauses.append("memory_id = ?")
            params.append(memory_id)
        if user_id is not None:
            clauses.append("user_id = ?")
            params.append(user_id)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit)
        cursor = await conn.execute(
            f"SELECT * FROM audit_log {where} ORDER BY sequence ASC LIMIT ?",
            params,
        )
        rows = await cursor.fetchall()
        return [
            AuditEntry(
                id=row["id"],
                sequence=int(row["sequence"]),
                operation=row["operation"],
                memory_id=row["memory_id"],
                user_id=row["user_id"],
                details=json.loads(row["details"] or "{}"),
                previous_hash=row["previous_hash"],
                entry_hash=row["entry_hash"],
                timestamp=datetime.fromisoformat(row["timestamp"]),
            )
            for row in rows
        ]

    async def verify_chain(self) -> bool:
        conn = self.store._require_conn()
        cursor = await conn.execute(
            "SELECT * FROM audit_log ORDER BY sequence ASC"
        )
        rows = await cursor.fetchall()
        expected_prev = _GENESIS_HASH
        for row in rows:
            if row["previous_hash"] != expected_prev:
                return False
            entry = AuditEntry(
                id=row["id"],
                sequence=int(row["sequence"]),
                operation=row["operation"],
                memory_id=row["memory_id"],
                user_id=row["user_id"],
                details=json.loads(row["details"] or "{}"),
                previous_hash=row["previous_hash"],
                entry_hash=row["entry_hash"],
                timestamp=datetime.fromisoformat(row["timestamp"]),
            )
            expected_hash = _compute_hash(entry.canonical_payload())
            if expected_hash != row["entry_hash"]:
                return False
            expected_prev = row["entry_hash"]
        return True
