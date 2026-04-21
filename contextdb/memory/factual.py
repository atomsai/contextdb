"""Factual memory — durable statements of fact.

The simplest of the three memory surfaces: a typed filter over
:meth:`contextdb.client.ContextDB.add` / :meth:`~contextdb.client.ContextDB.search`
that forces ``memory_type=FACTUAL``. Convenience wrappers add a few
fact-specific affordances (``recall``, ``update_fact``, ``list_facts``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from contextdb.core.models import MemoryItem, MemoryType

if TYPE_CHECKING:
    from contextdb.client import ContextDB


class FactualMemory:
    """Thin typed layer over the general-purpose client."""

    def __init__(self, client: ContextDB, user_id: str | None = None) -> None:
        self.client = client
        self.user_id = user_id

    async def add(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
        entity_mentions: list[str] | None = None,
        confidence: float = 1.0,
    ) -> MemoryItem:
        meta = dict(metadata or {})
        meta.setdefault("confidence", confidence)
        return await self.client.add(
            content=content,
            memory_type=MemoryType.FACTUAL,
            metadata=meta,
            entity_mentions=entity_mentions,
        )

    async def recall(self, query: str, top_k: int = 5) -> list[MemoryItem]:
        return await self.client.search(query, top_k=top_k, memory_type=MemoryType.FACTUAL)

    async def update_fact(
        self,
        memory_id: str,
        content: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryItem:
        return await self.client.update(memory_id, content=content, metadata=metadata)

    async def list_facts(self, limit: int = 100) -> list[MemoryItem]:
        await self.client._ensure_init()
        store = self.client._require_store()
        return await store.list_memories(
            user_id=self.user_id, memory_type=MemoryType.FACTUAL, limit=limit
        )
