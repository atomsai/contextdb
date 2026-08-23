"""Real-Postgres contract tests for explicit memory evolution.

Set ``CONTEXTDB_TEST_POSTGRES_URL`` to run these tests.
"""

from __future__ import annotations

import asyncio
import os

import pytest

import contextdb
from contextdb import (
    ContextDB,
    ContextDBConfig,
    EvolutionOutcome,
    EvolutionTargetNotFoundError,
)
from contextdb.store.sqlite_store import scoped_revision_key
from tests.pg_util import fresh_pg_database

pytestmark = pytest.mark.skipif(
    not os.environ.get("CONTEXTDB_TEST_POSTGRES_URL"),
    reason="set CONTEXTDB_TEST_POSTGRES_URL to run Postgres evolution tests",
)


def _config(url: str) -> ContextDBConfig:
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


async def _close_all(clients: list[ContextDB]) -> None:
    for client in clients:
        await client.close()


async def test_postgres_evolution_write_revision_and_audit_commit_together() -> None:
    import asyncpg

    async with fresh_pg_database() as url:
        client = contextdb.init(
            config=_config(url),
            tenant_id="atomic-org",
            agent_id="atomic-project",
        )
        try:
            result = await client.factual.evolve(
                "add",
                "The profile color is blue",
                source="user_stated",
                entity="profile",
                attribute="color",
                user_id="atomic-user",
            )
            assert result.memory is not None
            assert result.consistency_token.memory_version > 0
            assert result.consistency_token.primary_wal_lsn is not None

            connection = await asyncpg.connect(url)
            try:
                row_count = await connection.fetchval(
                    "SELECT COUNT(*) FROM memories WHERE id = $1",
                    result.memory.id,
                )
                version = await connection.fetchval(
                    "SELECT value FROM contextdb_meta WHERE key = $1",
                    scoped_revision_key("atomic-org", "atomic-project"),
                )
                audit_count = await connection.fetchval(
                    "SELECT COUNT(*) FROM audit_log "
                    "WHERE operation = 'CREATE' AND memory_id = $1",
                    result.memory.id,
                )
            finally:
                await connection.close()

            assert row_count == 1
            assert int(version) == result.consistency_token.memory_version
            assert audit_count == 1
        finally:
            await client.close()


async def test_postgres_update_rejects_historical_target_but_delete_accepts_it() -> None:
    async with fresh_pg_database() as url:
        client = contextdb.init(config=_config(url))
        try:
            added = await client.factual.evolve(
                "add",
                "The profile color is blue",
                source="user_stated",
                entity="profile",
                attribute="color",
                user_id="history-user",
            )
            assert added.memory is not None
            updated = await client.factual.evolve(
                "update",
                "The profile color is green",
                source="user_stated",
                target_memory_id=added.memory.id,
                user_id="history-user",
            )
            assert updated.memory is not None
            version = updated.consistency_token.memory_version

            with pytest.raises(EvolutionTargetNotFoundError):
                await client.factual.evolve(
                    "update",
                    "The profile color is red",
                    source="user_stated",
                    target_memory_id=added.memory.id,
                    user_id="history-user",
                )
            assert (await client.consistency_token()).memory_version == version

            deleted = await client.factual.evolve(
                "delete",
                target_memory_id=added.memory.id,
                user_id="history-user",
            )
            assert deleted.deleted_memory_ids == [added.memory.id]
            current = await client._require_store().get_raw(updated.memory.id)
            assert current is not None
            assert current.valid_until is None
        finally:
            await client.close()


async def test_postgres_evolution_rolls_back_if_audit_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import asyncpg

    async with fresh_pg_database() as url:
        client = contextdb.init(
            config=_config(url),
            tenant_id="rollback-org",
            agent_id="rollback-project",
        )
        try:
            seed = await client.factual.evolve(
                "add",
                "The profile color is blue",
                source="user_stated",
                entity="profile",
                attribute="color",
                user_id="rollback-user",
            )
            assert seed.memory is not None
            before = seed.consistency_token.memory_version
            assert client.audit is not None

            async def unavailable(*args: object, **kwargs: object) -> None:
                raise RuntimeError("audit unavailable")

            monkeypatch.setattr(client.audit, "log", unavailable)
            with pytest.raises(RuntimeError, match="audit unavailable"):
                await client.factual.evolve(
                    "update",
                    "The profile color is green",
                    source="user_stated",
                    entity="profile",
                    attribute="color",
                    user_id="rollback-user",
                )

            connection = await asyncpg.connect(url)
            try:
                rows = await connection.fetch(
                    "SELECT id, valid_until, superseded_by FROM memories "
                    "WHERE entity_key = 'profile' AND attribute_key = 'color'"
                )
                version = await connection.fetchval(
                    "SELECT value FROM contextdb_meta WHERE key = $1",
                    scoped_revision_key("rollback-org", "rollback-project"),
                )
            finally:
                await connection.close()
            assert len(rows) == 1
            assert rows[0]["id"] == seed.memory.id
            assert rows[0]["valid_until"] is None
            assert rows[0]["superseded_by"] is None
            assert int(version) == before
        finally:
            await client.close()


async def test_postgres_delete_rolls_back_if_erase_audit_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import asyncpg

    async with fresh_pg_database() as url:
        client = contextdb.init(
            config=_config(url),
            tenant_id="delete-rollback-org",
            agent_id="delete-rollback-project",
        )
        try:
            seed = await client.factual.evolve(
                "add",
                "The profile color is blue",
                source="user_stated",
                entity="profile",
                attribute="color",
                user_id="delete-rollback-user",
            )
            assert seed.memory is not None
            before = seed.consistency_token.memory_version
            store = client._require_store()
            assert seed.memory.id in await store.index_ids()
            assert client.audit is not None

            async def unavailable(*args: object, **kwargs: object) -> None:
                raise RuntimeError("audit unavailable")

            monkeypatch.setattr(client.audit, "log", unavailable)
            with pytest.raises(RuntimeError, match="audit unavailable"):
                await client.factual.evolve(
                    "delete",
                    target_memory_id=seed.memory.id,
                    user_id="delete-rollback-user",
                )

            connection = await asyncpg.connect(url)
            try:
                row_count = await connection.fetchval(
                    "SELECT COUNT(*) FROM memories WHERE id = $1",
                    seed.memory.id,
                )
                version = await connection.fetchval(
                    "SELECT value FROM contextdb_meta WHERE key = $1",
                    scoped_revision_key(
                        "delete-rollback-org",
                        "delete-rollback-project",
                    ),
                )
                erase_count = await connection.fetchval(
                    "SELECT COUNT(*) FROM audit_log "
                    "WHERE operation = 'ERASE' AND memory_id = $1",
                    seed.memory.id,
                )
            finally:
                await connection.close()

            assert row_count == 1
            assert int(version) == before
            assert erase_count == 0
            assert seed.memory.id in await store.index_ids()
        finally:
            await client.close()


async def test_postgres_concurrent_updates_leave_one_current_head() -> None:
    async with fresh_pg_database() as url:
        config = _config(url)
        seed_client = contextdb.init(config=config)
        workers = [contextdb.init(config=config) for _ in range(4)]
        try:
            await seed_client.factual.evolve(
                "add",
                "The meeting is at 2pm",
                source="user_stated",
                entity="meeting",
                attribute="time",
                user_id="concurrent-user",
            )
            await asyncio.gather(
                *[
                    worker.factual.evolve(
                        "update",
                        f"The meeting is at {hour}pm",
                        source="user_stated",
                        entity="meeting",
                        attribute="time",
                        user_id="concurrent-user",
                    )
                    for worker, hour in zip(
                        workers,
                        (3, 4, 5, 6),
                        strict=True,
                    )
                ]
            )

            rows = await seed_client._require_store().list_by_slot(
                "meeting",
                "time",
                user_id="concurrent-user",
            )
            current = [row for row in rows if row.valid_until is None]
            assert len(current) == 1
            assert len({row.id for row in rows}) == len(rows)
        finally:
            await seed_client.close()
            await _close_all(workers)


async def test_postgres_tokens_advance_for_mutations_not_noop_and_delete_cleans_index() -> None:
    import asyncpg

    async with fresh_pg_database() as url:
        config = _config(url)
        writer = contextdb.init(
            config=config,
            tenant_id="version-org",
            agent_id="version-project",
        )
        foreign = contextdb.init(
            config=config,
            tenant_id="version-org",
            agent_id="other-project",
        )
        try:
            added = await writer.factual.evolve(
                "add",
                "The profile color is blue",
                source="user_stated",
                entity="profile",
                attribute="color",
                user_id="version-user",
            )
            assert added.memory is not None
            add_version = added.consistency_token.memory_version
            assert add_version > 0
            assert (await foreign.consistency_token()).memory_version == 0

            noop = await writer.factual.evolve(
                "noop",
                noop_reason="already_processed",
                user_id="version-user",
            )
            assert noop.consistency_token.memory_version == add_version
            connection = await asyncpg.connect(url)
            try:
                version_after_noop = await connection.fetchval(
                    "SELECT value FROM contextdb_meta WHERE key = $1",
                    scoped_revision_key("version-org", "version-project"),
                )
                noop_audit_count = await connection.fetchval(
                    "SELECT COUNT(*) FROM audit_log "
                    "WHERE operation = 'NOOP' AND user_id = $1",
                    "version-user",
                )
            finally:
                await connection.close()
            assert int(version_after_noop) == add_version
            assert noop_audit_count == 1

            updated = await writer.factual.evolve(
                "update",
                "The profile color is green",
                source="user_stated",
                target_memory_id=added.memory.id,
                user_id="version-user",
            )
            assert updated.outcome == EvolutionOutcome.UPDATED
            assert updated.memory is not None
            assert updated.consistency_token.memory_version > add_version

            deleted = await writer.factual.evolve(
                "delete",
                target_memory_id=updated.memory.id,
                user_id="version-user",
            )
            assert (
                deleted.consistency_token.memory_version
                > updated.consistency_token.memory_version
            )
            store = writer._require_store()
            assert await store.get_raw(updated.memory.id) is None
            assert updated.memory.id not in await store.index_ids()
            assert (await foreign.consistency_token()).memory_version == 0
            connection = await asyncpg.connect(url)
            try:
                row_count = await connection.fetchval(
                    "SELECT COUNT(*) FROM memories WHERE id = $1",
                    updated.memory.id,
                )
                version_after_delete = await connection.fetchval(
                    "SELECT value FROM contextdb_meta WHERE key = $1",
                    scoped_revision_key("version-org", "version-project"),
                )
                erase_count = await connection.fetchval(
                    "SELECT COUNT(*) FROM audit_log "
                    "WHERE operation = 'ERASE' AND memory_id = $1",
                    updated.memory.id,
                )
            finally:
                await connection.close()
            assert row_count == 0
            assert (
                int(version_after_delete)
                == deleted.consistency_token.memory_version
            )
            assert erase_count == 1
        finally:
            await writer.close()
            await foreign.close()
