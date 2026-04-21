"""Experiential memory — trajectories, reflections, outcomes.

Experiential memories capture *what happened* during an agent run: the
action taken, the outcome observed, and an optional post-hoc reflection.
They are the substrate behind "agent learns from its own rollouts" —
Memory-R1 and MAGMA both derive their gains from a well-organized
experiential store.

Two memories are written per trajectory:

* A :class:`~contextdb.core.models.MemoryType.EXPERIENTIAL` item holding the
  structured trajectory (action / outcome / context).
* An optional linked reflection memory that the agent writes after the fact.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from contextdb.core.models import MemoryItem, MemoryType

if TYPE_CHECKING:
    from contextdb.client import ContextDB


class ExperientialMemory:
    """Trajectories (action+outcome) and reflections (post-hoc insights)."""

    def __init__(self, client: ContextDB, user_id: str | None = None) -> None:
        self.client = client
        self.user_id = user_id

    async def record_trajectory(
        self,
        action: str,
        outcome: str,
        context: dict[str, Any] | None = None,
        success: bool | None = None,
    ) -> MemoryItem:
        content = f"Action: {action}\nOutcome: {outcome}"
        meta: dict[str, Any] = {
            "action": action,
            "outcome": outcome,
            "context": context or {},
        }
        if success is not None:
            meta["success"] = success
        return await self.client.add(
            content=content,
            memory_type=MemoryType.EXPERIENTIAL,
            metadata=meta,
            event_time=datetime.now(tz=timezone.utc),
        )

    async def add_reflection(
        self,
        trajectory_id: str,
        insight: str,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryItem:
        meta = dict(metadata or {})
        meta["reflection_on"] = trajectory_id
        return await self.client.add(
            content=insight,
            memory_type=MemoryType.EXPERIENTIAL,
            metadata=meta,
        )

    async def recall_similar(
        self,
        situation: str,
        top_k: int = 5,
    ) -> list[MemoryItem]:
        return await self.client.search(
            situation, top_k=top_k, memory_type=MemoryType.EXPERIENTIAL
        )

    async def list_trajectories(self, limit: int = 100) -> list[MemoryItem]:
        await self.client._ensure_init()
        store = self.client._require_store()
        items = await store.list_memories(
            user_id=self.user_id, memory_type=MemoryType.EXPERIENTIAL, limit=limit
        )
        return [m for m in items if "reflection_on" not in m.metadata]

    async def list_reflections(
        self,
        trajectory_id: str | None = None,
        limit: int = 100,
    ) -> list[MemoryItem]:
        await self.client._ensure_init()
        store = self.client._require_store()
        items = await store.list_memories(
            user_id=self.user_id, memory_type=MemoryType.EXPERIENTIAL, limit=limit
        )
        reflections = [m for m in items if "reflection_on" in m.metadata]
        if trajectory_id is not None:
            reflections = [
                m for m in reflections if m.metadata.get("reflection_on") == trajectory_id
            ]
        return reflections
