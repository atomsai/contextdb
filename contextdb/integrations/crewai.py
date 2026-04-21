"""CrewAI memory adapter.

CrewAI's memory contract is intentionally small (``save`` / ``search`` /
``reset``). We mirror those method names on a thin wrapper so that a
``Crew`` can consume a :class:`ContextDBCrewMemory` without importing
``crewai`` here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from contextdb.client import ContextDB


class ContextDBCrewMemory:
    """CrewAI-compatible memory wrapper backed by ContextDB."""

    def __init__(self, client: ContextDB, top_k: int = 5) -> None:
        self.client = client
        self.top_k = top_k

    async def save(self, value: str, metadata: dict[str, Any] | None = None) -> str:
        item = await self.client.add(content=value, metadata=metadata)
        return item.id

    async def search(self, query: str, limit: int | None = None) -> list[dict[str, Any]]:
        top_k = limit if limit is not None else self.top_k
        hits = await self.client.search(query, top_k=top_k)
        return [
            {"id": m.id, "content": m.content, "metadata": m.metadata}
            for m in hits
        ]

    async def reset(self) -> None:
        await self.client._ensure_init()
        store = self.client._require_store()
        memories = await store.list_memories(user_id=self.client.user_id, limit=100000)
        for memory in memories:
            await store.delete(memory.id, hard=True)
