"""In-process pub/sub bus for multi-agent memory sharing.

The bus is deliberately minimal: ``publish`` + ``subscribe`` with optional
filters. Agents that share a :class:`MemoryBus` can broadcast new memories,
events, or reflections to peers without coupling directly to each other.

For cross-process sharing, swap the backing queue for Redis or NATS — the
:class:`MemoryBus` contract stays identical. That's out of scope for v0.1.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4


@dataclass
class _Subscription:
    id: str
    topic: str
    callback: Callable[[dict[str, Any]], Awaitable[None]]
    filters: dict[str, Any] = field(default_factory=dict)


class MemoryBus:
    """Async, in-process pub/sub across cooperating agents."""

    def __init__(self) -> None:
        self._subscriptions: dict[str, list[_Subscription]] = {}
        self._lock = asyncio.Lock()

    async def publish(self, topic: str, payload: dict[str, Any]) -> int:
        """Deliver ``payload`` to all matching subscribers. Returns delivery count."""
        async with self._lock:
            subs = list(self._subscriptions.get(topic, []))
            wildcard = list(self._subscriptions.get("*", []))
        delivered = 0
        for sub in subs + wildcard:
            if not _matches_filters(payload, sub.filters):
                continue
            try:
                await sub.callback(payload)
                delivered += 1
            except Exception:  # noqa: BLE001
                # Subscriber failure must not break the bus.
                continue
        return delivered

    async def subscribe(
        self,
        topic: str,
        callback: Callable[[dict[str, Any]], Awaitable[None]],
        filters: dict[str, Any] | None = None,
    ) -> str:
        sub = _Subscription(
            id=str(uuid4()), topic=topic, callback=callback, filters=filters or {}
        )
        async with self._lock:
            self._subscriptions.setdefault(topic, []).append(sub)
        return sub.id

    async def unsubscribe(self, subscription_id: str) -> bool:
        async with self._lock:
            for topic, subs in self._subscriptions.items():
                for i, sub in enumerate(subs):
                    if sub.id == subscription_id:
                        del self._subscriptions[topic][i]
                        return True
        return False

    async def topics(self) -> list[str]:
        async with self._lock:
            return [t for t, subs in self._subscriptions.items() if subs]


def _matches_filters(payload: dict[str, Any], filters: dict[str, Any]) -> bool:
    return all(payload.get(key) == expected for key, expected in filters.items())
