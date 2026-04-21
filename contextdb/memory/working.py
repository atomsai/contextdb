"""Working memory — short-lived, token-budgeted session scratchpad.

Working memories are the agent's "what am I doing right now" buffer. They are
scoped to a session, bounded by a token budget (characters ÷ 4 as a coarse
proxy for tokens), and trimmed FIFO when the budget is exceeded.

Items are stored with ``memory_type=WORKING`` and a ``session_id`` metadata
tag. The retention policy short-circuits them to a 24-hour TTL by default.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from contextdb.core.models import MemoryItem, MemoryStatus, MemoryType

if TYPE_CHECKING:
    from contextdb.client import ContextDB


def _approx_tokens(text: str) -> int:
    """Rough token estimate: 4 chars ≈ 1 token (GPT-style average)."""
    return max(1, len(text) // 4)


class WorkingMemory:
    """Token-budgeted FIFO scratchpad scoped to a session."""

    def __init__(
        self,
        client: ContextDB,
        session_id: str,
        max_tokens: int = 4000,
    ) -> None:
        self.client = client
        self.session_id = session_id
        self.max_tokens = max_tokens

    async def push(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryItem:
        meta = dict(metadata or {})
        meta["session_id"] = self.session_id
        item = await self.client.add(
            content=content,
            memory_type=MemoryType.WORKING,
            metadata=meta,
        )
        await self._evict_if_over_budget()
        return item

    async def list_session(self) -> list[MemoryItem]:
        await self.client._ensure_init()
        store = self.client._require_store()
        items = await store.list_memories(memory_type=MemoryType.WORKING, limit=10000)
        return [
            m
            for m in items
            if m.metadata.get("session_id") == self.session_id
            and m.status == MemoryStatus.ACTIVE
        ]

    async def context_window(self) -> str:
        items = await self.list_session()
        items.sort(key=lambda m: m.created_at)
        return "\n".join(m.content for m in items)

    async def clear(self) -> int:
        items = await self.list_session()
        await self.client._ensure_init()
        store = self.client._require_store()
        for item in items:
            await store.delete(item.id, hard=True)
        return len(items)

    async def _evict_if_over_budget(self) -> None:
        items = await self.list_session()
        items.sort(key=lambda m: m.created_at)
        total = sum(_approx_tokens(m.content) for m in items)
        if total <= self.max_tokens:
            return
        await self.client._ensure_init()
        store = self.client._require_store()
        for item in items:
            if total <= self.max_tokens:
                break
            await store.delete(item.id, hard=True)
            total -= _approx_tokens(item.content)
