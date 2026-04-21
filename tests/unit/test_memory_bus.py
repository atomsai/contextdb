"""Tests for :class:`MemoryBus`."""

from __future__ import annotations

import pytest

from contextdb.agents.memory_bus import MemoryBus


@pytest.mark.asyncio
async def test_publish_delivers_to_subscribers() -> None:
    bus = MemoryBus()
    received: list[dict] = []

    async def handler(payload: dict) -> None:
        received.append(payload)

    await bus.subscribe("events", handler)
    delivered = await bus.publish("events", {"x": 1})
    assert delivered == 1
    assert received == [{"x": 1}]


@pytest.mark.asyncio
async def test_unsubscribe_stops_delivery() -> None:
    bus = MemoryBus()

    async def handler(_: dict) -> None:
        pass

    sub_id = await bus.subscribe("events", handler)
    assert await bus.unsubscribe(sub_id)
    delivered = await bus.publish("events", {"x": 1})
    assert delivered == 0


@pytest.mark.asyncio
async def test_filters_exclude_non_matching() -> None:
    bus = MemoryBus()
    matched: list[dict] = []

    async def handler(payload: dict) -> None:
        matched.append(payload)

    await bus.subscribe("events", handler, filters={"kind": "ADD"})
    await bus.publish("events", {"kind": "DELETE"})
    await bus.publish("events", {"kind": "ADD"})
    assert len(matched) == 1
