"""End-to-end tests for the :class:`ContextDB` client."""

from __future__ import annotations

import pytest

from contextdb import ContextDB
from contextdb.core.models import MemoryType


@pytest.mark.asyncio
async def test_add_and_search_roundtrip(client: ContextDB) -> None:
    item = await client.add(
        content="User's birthday is March 5.",
        memory_type=MemoryType.FACTUAL,
    )
    assert item.id
    hits = await client.search("when is the birthday?", top_k=5)
    assert any(h.id == item.id for h in hits)


@pytest.mark.asyncio
async def test_update_and_delete(client: ContextDB) -> None:
    item = await client.add(content="initial", memory_type=MemoryType.FACTUAL)
    updated = await client.update(item.id, content="updated")
    assert updated.content == "updated"
    await client.delete(item.id)


@pytest.mark.asyncio
async def test_stats(client: ContextDB) -> None:
    await client.add(content="a")
    stats = await client.stats()
    assert stats["total_memories"] >= 1
    assert "graphs" in stats


@pytest.mark.asyncio
async def test_working_memory(client: ContextDB) -> None:
    wm = client.working("session-1", max_tokens=4000)
    await wm.push("turn 1")
    await wm.push("turn 2")
    window = await wm.context_window()
    assert "turn 1" in window
    assert "turn 2" in window


@pytest.mark.asyncio
async def test_factual_surface(client: ContextDB) -> None:
    await client.factual.add("Water boils at 100°C.")
    facts = await client.factual.list_facts()
    assert facts


@pytest.mark.asyncio
async def test_experiential_surface(client: ContextDB) -> None:
    traj = await client.experiential.record_trajectory(
        action="send_email",
        outcome="delivered",
        success=True,
    )
    reflection = await client.experiential.add_reflection(
        traj.id, "Next time use a shorter subject line."
    )
    reflections = await client.experiential.list_reflections(trajectory_id=traj.id)
    assert any(r.id == reflection.id for r in reflections)


@pytest.mark.asyncio
async def test_audit_logged_on_write(client: ContextDB) -> None:
    await client._ensure_init()
    assert client.audit is not None
    await client.add(content="audited")
    history = await client.audit.get_history()
    assert any(e.operation == "CREATE" for e in history)
    assert await client.audit.verify_chain()


@pytest.mark.asyncio
async def test_pii_is_redacted(client: ContextDB) -> None:
    item = await client.add(content="email me at foo@bar.com thanks")
    assert "foo@bar.com" not in item.content
    assert "[EMAIL]" in item.content
