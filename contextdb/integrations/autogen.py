"""AutoGen memory adapter.

AutoGen passes strings around as agent "messages" and expects memory to
surface a list of relevant prior messages on demand. We wrap ContextDB to
match that shape.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from contextdb.client import ContextDB


class ContextDBAutoGenMemory:
    """AutoGen-style memory facade over ContextDB."""

    def __init__(self, client: ContextDB, top_k: int = 5) -> None:
        self.client = client
        self.top_k = top_k

    async def add_message(self, role: str, content: str) -> str:
        item = await self.client.add(
            content=f"{role}: {content}",
            metadata={"role": role},
        )
        return item.id

    async def get_relevant(self, query: str) -> list[str]:
        hits = await self.client.search(query, top_k=self.top_k)
        return [m.content for m in hits]

    async def clear(self) -> None:
        await self.client._ensure_init()
        store = self.client._require_store()
        memories = await store.list_memories(user_id=self.client.user_id, limit=100000)
        for memory in memories:
            await store.delete(memory.id, hard=True)
