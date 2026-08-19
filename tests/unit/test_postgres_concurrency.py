"""Real Postgres concurrent-worker tests.

These exercise the cross-process guarantees of the Postgres store with
multiple independent clients (separate pools — the in-process equivalent
of separate workers) over one database:

* concurrent same-value slot writes must not lose corroboration;
* concurrent audit appends must produce a single, unbroken hash chain;
* a write from one worker is visible to another worker's next recall.

Requires ``CONTEXTDB_TEST_POSTGRES_URL`` (skipped otherwise).
"""

from __future__ import annotations

import asyncio
import os
import uuid

import pytest

import contextdb
from contextdb import ContextDB
from contextdb.core.config import ContextDBConfig
from contextdb.store.postgres_store import PostgresStore
from contextdb.store.sqlite_store import scoped_revision_key
from tests.pg_util import fresh_pg_database

pytestmark = pytest.mark.skipif(
    not os.environ.get("CONTEXTDB_TEST_POSTGRES_URL"),
    reason="set CONTEXTDB_TEST_POSTGRES_URL to run Postgres worker tests",
)


def _cfg(url: str) -> ContextDBConfig:
    return ContextDBConfig(
        storage_url=url,
        embedding_model="mock",
        embedding_dim=32,
        llm_model="mock",
        llm_api_key="mock",
        enable_entity_graph=False,
        enable_multi_graph=False,
        enable_auto_link=False,
        enable_audit=True,
    )


def _workers(url: str, n: int) -> list[ContextDB]:
    """Independent clients = independent pools = separate 'workers'."""
    return [contextdb.init(config=_cfg(url)) for _ in range(n)]


async def _close_all(clients: list[ContextDB]) -> None:
    for c in clients:
        await c.close()


@pytest.mark.asyncio
async def test_external_pool_is_shared_and_not_closed_by_clients() -> None:
    import asyncpg

    async with fresh_pg_database() as url:
        pool = await asyncpg.create_pool(url, min_size=1, max_size=4)
        first = contextdb.init(
            config=_cfg(url),
            tenant_id="tenant-shared",
            agent_id="agent-shared",
            postgres_pool=pool,
        )
        second = contextdb.init(
            config=_cfg(url),
            tenant_id="tenant-shared",
            agent_id="agent-shared",
            postgres_pool=pool,
        )
        try:
            stored = await first.factual.add(
                "The shared pool remains open",
                source="user_stated",
                user_id="user-shared",
            )
            await first.close()
            async with pool.acquire() as conn:
                assert await conn.fetchval("SELECT 1") == 1
            memories = await second.factual.list_facts(
                user_id="user-shared",
                limit=10,
            )
            assert stored.id in {memory.id for memory in memories}
            await second.close()
            async with pool.acquire() as conn:
                assert await conn.fetchval("SELECT 1") == 1
        finally:
            await first.close()
            await second.close()
            await pool.close()


@pytest.mark.asyncio
async def test_shared_pool_audit_waiters_cannot_starve_lock_holder() -> None:
    import asyncpg

    async with fresh_pg_database() as url:
        pool = await asyncpg.create_pool(url, min_size=4, max_size=4)
        stores = [PostgresStore(url, pool=pool) for _ in range(4)]
        for store in stores:
            await store.initialize()

        holder_entered = asyncio.Event()
        release_holder = asyncio.Event()

        async def holder() -> None:
            async with stores[0].audit_lock():
                holder_entered.set()
                await release_holder.wait()
                await stores[0]._require_conn().execute("SELECT 1")

        async def waiter(store: PostgresStore) -> None:
            async with store.audit_lock():
                await store._require_conn().execute("SELECT 1")

        tasks = [asyncio.create_task(holder())]
        try:
            await holder_entered.wait()
            tasks.extend(
                asyncio.create_task(waiter(store))
                for store in stores[1:]
            )
            for _ in range(100):
                if pool.get_idle_size() == 0:
                    break
                await asyncio.sleep(0.01)
            assert pool.get_idle_size() == 0
            release_holder.set()
            await asyncio.wait_for(asyncio.gather(*tasks), timeout=5)
        finally:
            release_holder.set()
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            for store in stores:
                await store.close()
            await pool.close()


@pytest.mark.asyncio
async def test_unkeyed_write_and_revision_roll_back_when_audit_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import asyncpg

    async with fresh_pg_database() as url:
        client = contextdb.init(
            config=_cfg(url),
            tenant_id="atomic-org",
            agent_id="atomic-project",
        )
        try:
            await client.stats()
            assert client.audit is not None

            async def unavailable(*args: object, **kwargs: object) -> None:
                raise RuntimeError("audit unavailable")

            monkeypatch.setattr(client.audit, "log", unavailable)
            with pytest.raises(RuntimeError, match="audit unavailable"):
                await client.factual.add(
                    "Atomic write must roll back",
                    source="user_stated",
                    user_id="atomic-user",
                )

            connection = await asyncpg.connect(url)
            try:
                count = await connection.fetchval(
                    "SELECT COUNT(*) FROM memories "
                    "WHERE tenant_id = $1 AND agent_id = $2",
                    "atomic-org",
                    "atomic-project",
                )
                version = await connection.fetchval(
                    "SELECT value FROM contextdb_meta WHERE key = $1",
                    scoped_revision_key(
                        "atomic-org",
                        "atomic-project",
                    ),
                )
            finally:
                await connection.close()
            assert count == 0
            assert version is None
        finally:
            await client.close()


@pytest.mark.asyncio
async def test_project_version_and_wal_token_gate_reads() -> None:
    from contextdb.core.exceptions import StaleReadError

    async with fresh_pg_database() as url:
        writer = contextdb.init(
            config=_cfg(url),
            tenant_id="version-org",
            agent_id="version-project",
        )
        reader = contextdb.init(
            config=_cfg(url),
            tenant_id="version-org",
            agent_id="version-project",
        )
        foreign = contextdb.init(
            config=_cfg(url),
            tenant_id="version-org",
            agent_id="foreign-project",
        )
        try:
            stored = await writer.factual.add(
                "Versioned memory",
                source="user_stated",
                user_id="version-user",
            )
            token = await writer.consistency_token()
            assert token.memory_version == 1
            assert token.primary_wal_lsn is not None

            observed = await reader.require_consistency(
                min_memory_version=token.memory_version,
                min_wal_lsn=token.primary_wal_lsn,
            )
            assert observed.memory_version >= token.memory_version

            with pytest.raises(StaleReadError):
                await foreign.require_consistency(
                    min_memory_version=token.memory_version,
                )

            await writer.confirm(
                stored.id,
                user_id="version-user",
            )
            confirmed = await writer.consistency_token()
            assert confirmed.memory_version > token.memory_version
            await reader.require_consistency(
                min_memory_version=confirmed.memory_version,
                min_wal_lsn=confirmed.primary_wal_lsn,
            )

            await writer.forget(
                user_id="version-user",
                memory_id=stored.id,
            )
            forgotten = await writer.consistency_token()
            assert forgotten.memory_version > confirmed.memory_version
            await reader.require_consistency(
                min_memory_version=forgotten.memory_version,
                min_wal_lsn=forgotten.primary_wal_lsn,
            )
        finally:
            await writer.close()
            await reader.close()
            await foreign.close()


@pytest.mark.asyncio
async def test_concurrent_slot_writes_do_not_lose_corroboration() -> None:
    """Six workers restating the same slot value in different sessions must
    all land as independent corroboration — the cross-process slot lock
    serializes the read-modify-write of ``corroborated_by``."""
    user = f"user-{uuid.uuid4().hex}"
    async with fresh_pg_database() as url:
        # Distinct session per worker → each is an independent speaker.
        workers = [
            contextdb.init(
                config=_cfg(url), session_id=f"session-{i}-{uuid.uuid4().hex[:8]}"
            )
            for i in range(6)
        ]
        try:
            results = await asyncio.gather(
                *[
                    w.factual.add(
                        "A colleague said the office is moving to Denver",
                        source="third_party",
                        confidence=0.6,
                        action_relevant=True,
                        entity="office",
                        attribute="location",
                        user_id=user,
                    )
                    for w in workers
                ]
            )
            # Same slot value from independent speakers corroborates one memory.
            assert {r.id for r in results} == {results[0].id}
            final = await workers[0].get(results[0].id)
            assert final is not None
            assert final.corroboration_count == len(workers), (
                f"lost updates: expected {len(workers)} speakers, "
                f"got {final.corroboration_count} ({final.corroborated_by})"
            )
        finally:
            await _close_all(workers)


@pytest.mark.asyncio
async def test_concurrent_slot_corrections_serialize() -> None:
    """Same speaker correcting a slot from two workers: exactly one current
    occupant at the end (supersede serialized, no double-current race)."""
    user = f"user-{uuid.uuid4().hex}"
    async with fresh_pg_database() as url:
        workers = _workers(url, 4)
        try:
            await asyncio.gather(
                *[
                    w.factual.add(
                        f"The meeting is at {hour}pm",
                        source="user_stated",
                        confidence=0.9,
                        action_relevant=True,
                        entity="meeting",
                        attribute="time",
                        user_id=user,
                    )
                    for w, hour in zip(workers, [3, 4, 5, 6], strict=True)
                ]
            )
            store = workers[0]._require_store()
            rows = await store.list_by_slot("meeting", "time", user_id=user)
            current = [r for r in rows if r.valid_until is None]
            assert len(current) == 1, (
                f"expected exactly one current slot occupant, got "
                f"{[(r.content, r.valid_until) for r in current]}"
            )
        finally:
            await _close_all(workers)


@pytest.mark.asyncio
async def test_concurrent_audit_appends_keep_one_unbroken_chain() -> None:
    """Four workers x five audited writes: sequences are exactly 1..20 and
    the hash chain verifies — appends never fork, even across pools."""
    user = f"user-{uuid.uuid4().hex}"
    async with fresh_pg_database() as url:
        workers = _workers(url, 4)
        try:
            await asyncio.gather(
                *[
                    w.factual.add(
                        f"worker note {i}-{j}", source="user_stated", user_id=user
                    )
                    for i, w in enumerate(workers)
                    for j in range(5)
                ]
            )
            audit = workers[0].audit
            assert audit is not None
            history = await audit.get_history(user_id=user, limit=1000)
            creates = [e for e in history if e.operation == "CREATE"]
            assert len(creates) == 20
            sequences = [e.sequence for e in creates]
            assert len(set(sequences)) == len(sequences), (
                "duplicate sequences: forked chain"
            )
            assert await audit.verify_chain() is True
        finally:
            await _close_all(workers)


@pytest.mark.asyncio
async def test_write_from_one_worker_is_recallable_from_another() -> None:
    """Index coherence: worker B's index is warm before worker A writes;
    B's next recall must still surface A's memory (revision rebuild)."""
    user = f"user-{uuid.uuid4().hex}"
    async with fresh_pg_database() as url:
        a, b = _workers(url, 2)
        try:
            token = uuid.uuid4().hex[:12]
            # Warm B's index.
            assert await b.factual.recall(f"anything {token}", user_id=user) == []
            await a.factual.add(
                f"cross-worker note {token}", source="user_stated", user_id=user
            )
            hits = await b.factual.recall(f"cross-worker note {token}", user_id=user)
            assert any(token in h.content for h in hits)
        finally:
            await _close_all([a, b])
