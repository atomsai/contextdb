"""Host-adoption APIs: per-call user, pending confirmations, forget, batch."""

from __future__ import annotations

from pathlib import Path

import pytest

import contextdb
from contextdb.core.config import ContextDBConfig
from contextdb.core.exceptions import (
    ConfigError,
    MemoryNotFoundError,
    SourceRequiredError,
)
from contextdb.pool import ContextDBPool


def _cfg(tmp_path: Path, name: str = "host.db") -> ContextDBConfig:
    return ContextDBConfig(
        storage_url=f"sqlite:///{tmp_path}/{name}",
        embedding_model="mock",
        embedding_dim=32,
        llm_model="mock",
        llm_api_key="mock",
        enable_entity_graph=False,
        enable_multi_graph=False,
        enable_auto_link=False,
        enable_audit=True,
    )


@pytest.mark.asyncio
async def test_shared_client_per_call_user_isolation(tmp_path: Path) -> None:
    db = contextdb.init(config=_cfg(tmp_path))
    try:
        await db.factual.add(
            "Alice PIN is 7788",
            source="user_stated",
            entity="account",
            attribute="pin",
            action_relevant=True,
            user_id="alice",
        )
        await db.factual.add(
            "Bob likes green",
            source="user_stated",
            user_id="bob",
        )
        alice_hits = await db.factual.recall("PIN", user_id="alice")
        bob_hits = await db.factual.recall("PIN", user_id="bob")
        assert any("7788" in h.content for h in alice_hits)
        assert all("7788" not in h.content for h in bob_hits)
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_scoped_client_cannot_widen(tmp_path: Path) -> None:
    db = contextdb.init(user_id="alice", config=_cfg(tmp_path))
    try:
        with pytest.raises(ConfigError, match="cannot operate as"):
            await db.factual.recall("x", user_id="bob")
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_missing_source_warns_and_strict_raises(tmp_path: Path) -> None:
    db = contextdb.init(user_id="u", config=_cfg(tmp_path))
    try:
        with pytest.warns(UserWarning, match="epistemic source was omitted"):
            await db.factual.add("hello")
        db.config.require_source = True
        with pytest.raises(SourceRequiredError):
            await db.factual.add("hello again")
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_pending_confirmations_and_confirm(tmp_path: Path) -> None:
    db = contextdb.init(user_id="u", config=_cfg(tmp_path))
    try:
        wish = await db.factual.add(
            "I'd like to come in Thursday",
            source="user_stated",
            confidence=0.5,
            action_relevant=True,
            entity="caller",
            attribute="preferred_visit_day",
        )
        pending = await db.factual.pending_confirmations()
        assert any(m.id == wish.id for m in pending)
        await db.factual.confirm(wish.id)
        pending_after = await db.factual.pending_confirmations()
        assert all(m.id != wish.id for m in pending_after)
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_forget_one_and_slot(tmp_path: Path) -> None:
    db = contextdb.init(user_id="u", config=_cfg(tmp_path))
    try:
        one = await db.factual.add("keep me", source="user_stated")
        addr = await db.factual.add(
            "I live at 1 Main",
            source="user_stated",
            entity="caller",
            attribute="address",
        )
        assert await db.forget(memory_id=one.id) == 1
        assert await db.get(one.id) is None
        assert await db.forget(entity="caller", attribute="address") == 1
        assert await db.get(addr.id) is None
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_add_many_and_recall_filters(tmp_path: Path) -> None:
    db = contextdb.init(user_id="u", config=_cfg(tmp_path))
    try:
        stored = await db.factual.add_many(
            [
                "turn one",
                {
                    "content": "caller email is a@b.c",
                    "source": "user_stated",
                    "entity": "caller",
                    "attribute": "email",
                    "confidence": 0.9,
                },
            ]
        )
        assert len(stored) == 2
        hits = await db.factual.recall("email", entity="caller", min_confidence=0.8)
        assert hits
        assert all(h.entity_key == "caller" for h in hits)
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_confirm_rejects_foreign_memory_id(tmp_path: Path) -> None:
    """A shared client must never confirm a known foreign memory ID."""
    db = contextdb.init(config=_cfg(tmp_path))
    try:
        item = await db.factual.add(
            "Alice PIN is 7788",
            source="user_stated",
            confidence=0.5,
            action_relevant=True,
            entity="account",
            attribute="pin",
            user_id="alice",
        )
        with pytest.raises(MemoryNotFoundError):
            await db.factual.confirm(item.id, user_id="bob")
        still = await db.get(item.id)
        assert still is not None
        assert still.confirmed is False

        confirmed = await db.factual.confirm(item.id, user_id="alice")
        assert confirmed.confirmed is True
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_forget_rejects_foreign_memory_id(tmp_path: Path) -> None:
    """A shared client must never delete a known foreign memory ID — and a
    missing id keeps the legacy ``0`` return."""
    db = contextdb.init(config=_cfg(tmp_path))
    try:
        item = await db.factual.add(
            "Alice PIN is 7788", source="user_stated", user_id="alice"
        )
        with pytest.raises(MemoryNotFoundError):
            await db.forget(user_id="bob", memory_id=item.id)
        assert await db.get(item.id) is not None

        assert await db.forget(user_id="bob", memory_id="no-such-id") == 0
        assert await db.forget(user_id="alice", memory_id=item.id) == 1
        assert await db.get(item.id) is None
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_unscoped_local_usage_has_no_scope_checks(tmp_path: Path) -> None:
    """No user anywhere: confirm/forget by id work as before. This local
    usage is documented as NOT an authorization boundary."""
    db = contextdb.init(config=_cfg(tmp_path))
    try:
        item = await db.factual.add(
            "local fact", source="user_stated", confidence=0.5, action_relevant=True
        )
        assert item.user_id is None
        confirmed = await db.factual.confirm(item.id)
        assert confirmed.confirmed is True
        assert await db.forget(memory_id=item.id) == 1
        assert await db.get(item.id) is None
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_on_event_hook(tmp_path: Path) -> None:
    db = contextdb.init(user_id="u", config=_cfg(tmp_path))
    seen: list[str] = []

    def hook(event: str, payload: dict[str, object]) -> None:
        seen.append(event)

    db.on("write", hook)
    db.on("recall", hook)
    try:
        await db.factual.add("hi", source="user_stated")
        await db.factual.recall("hi")
        assert "write" in seen
        assert "recall" in seen
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_pool_lru(tmp_path: Path) -> None:
    pool = ContextDBPool(_cfg(tmp_path), max_clients=2)
    a = pool.client("a")
    b = pool.client("b")
    assert len(pool) == 2
    _c = pool.client("c")
    assert len(pool) == 2
    assert a is not pool.client("a")  # evicted, new instance
    await pool.aclose()
    del b
