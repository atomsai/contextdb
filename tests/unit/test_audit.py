"""Tests for the hash-chained audit log."""

from __future__ import annotations

import pytest

from contextdb.privacy.audit import AuditLogger
from contextdb.store.sqlite_store import SQLiteStore


@pytest.mark.asyncio
async def test_chain_is_valid(tmp_store: SQLiteStore) -> None:
    log = AuditLogger(tmp_store)
    await log.initialize()
    await log.log("CREATE", memory_id="a", user_id="u")
    await log.log("READ", memory_id="a", user_id="u")
    await log.log("DELETE", memory_id="a", user_id="u")
    assert await log.verify_chain()


@pytest.mark.asyncio
async def test_tampering_is_detected(tmp_store: SQLiteStore) -> None:
    log = AuditLogger(tmp_store)
    await log.initialize()
    await log.log("CREATE", memory_id="a")
    conn = tmp_store._require_conn()
    await conn.execute(
        "UPDATE audit_log SET operation = 'TAMPERED' WHERE sequence = 1"
    )
    await conn.commit()
    assert not await log.verify_chain()
