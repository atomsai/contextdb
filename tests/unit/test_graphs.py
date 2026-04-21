"""Tests for individual graph implementations."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from contextdb.core.models import MemoryItem
from contextdb.graphs.entity import EntityGraph
from contextdb.graphs.semantic import SemanticGraph
from contextdb.graphs.temporal import TemporalGraph
from contextdb.store.sqlite_store import SQLiteStore
from contextdb.utils.llm import MockLLM


@pytest.mark.asyncio
async def test_semantic_graph_links_similar(tmp_store: SQLiteStore) -> None:
    graph = SemanticGraph(tmp_store, threshold=0.0)
    await graph.initialize()
    a = await tmp_store.add(MemoryItem(content="dogs", embedding=[1.0] + [0.0] * 31))
    b = await tmp_store.add(MemoryItem(content="canines", embedding=[1.0] + [0.0] * 31))
    await graph.add_node(a.id, {"embedding": a.embedding})
    await graph.add_node(b.id, {"embedding": b.embedding})
    neighbors_a = await graph.get_neighbors(a.id)
    assert any(nid == b.id for nid, _ in neighbors_a)


@pytest.mark.asyncio
async def test_temporal_graph_proximity(tmp_store: SQLiteStore) -> None:
    graph = TemporalGraph(tmp_store, proximity_window=timedelta(hours=1))
    await graph.initialize()
    t0 = datetime.now(tz=timezone.utc)
    a = await tmp_store.add(
        MemoryItem(content="first", embedding=[0.0] * 32, event_time=t0)
    )
    b = await tmp_store.add(
        MemoryItem(
            content="second",
            embedding=[0.0] * 32,
            event_time=t0 + timedelta(minutes=10),
        )
    )
    await graph.add_node(a.id, {"event_time": t0})
    await graph.add_node(b.id, {"event_time": t0 + timedelta(minutes=10)})
    neighbors = await graph.get_neighbors(b.id)
    assert any(nid == a.id for nid, _ in neighbors)


@pytest.mark.asyncio
async def test_entity_graph_extraction(tmp_store: SQLiteStore) -> None:
    llm = MockLLM(
        responses={
            "Extract": '{"entities": [{"name": "Alice", "type": "PERSON"}]}',
        }
    )
    graph = EntityGraph(tmp_store, llm)
    await graph.initialize()
    item = await tmp_store.add(
        MemoryItem(content="Alice met Bob.", embedding=[0.0] * 32)
    )
    await graph.add_node(item.id, {"content": item.content})
    profile = await graph.get_entity_profile("Alice")
    assert profile["memories"] == [item.id]
