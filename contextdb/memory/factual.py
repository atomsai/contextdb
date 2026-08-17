"""Factual memory — durable statements of fact, with epistemic typing.

A typed filter over :meth:`contextdb.client.ContextDB.add` /
:meth:`~contextdb.client.ContextDB.search` that forces
``memory_type=FACTUAL`` and carries the trust model:

* ``add`` accepts epistemic overrides (``source`` / ``confidence`` /
  ``action_relevant`` / ``entity`` / ``attribute``). Supplying both
  ``entity`` and ``attribute`` enables slot-based dedupe and supersede.
* ``recall`` returns currently-valid facts; pass ``as_of`` for time travel.
* ``recall_for_action`` returns only facts an agent may act on without
  confirming first.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

from contextdb.core.models import EpistemicSource, MemoryItem, MemoryType

if TYPE_CHECKING:
    from contextdb.client import ContextDB


class FactualMemory:
    """Thin typed layer over the general-purpose client."""

    def __init__(self, client: ContextDB, user_id: str | None = None) -> None:
        self.client = client
        self.user_id = user_id

    def _user(self, user_id: str | None) -> str | None:
        return user_id if user_id is not None else self.user_id

    async def add(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
        entity_mentions: list[str] | None = None,
        confidence: float = 1.0,
        *,
        source: EpistemicSource | None = None,
        action_relevant: bool | None = None,
        entity: str | None = None,
        attribute: str | None = None,
        user_id: str | None = None,
    ) -> MemoryItem:
        """Store a fact.

        ``source`` is the epistemic provenance (who vouches for it); the
        legacy free-form provenance string on the client is untouched.
        ``entity`` + ``attribute`` together define the dedupe/contradiction
        slot. Omit ``source`` and the SDK warns; HTTP/MCP remember reject it.
        """
        meta = dict(metadata or {})
        meta.setdefault("confidence", confidence)
        return await self.client.add(
            content=content,
            memory_type=MemoryType.FACTUAL,
            metadata=meta,
            entity_mentions=entity_mentions,
            epistemic_source=source,
            confidence=confidence,
            action_relevant=action_relevant,
            entity_key=entity,
            attribute_key=attribute,
            user_id=self._user(user_id),
        )

    async def add_fast(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
        entity_mentions: list[str] | None = None,
        *,
        user_id: str | None = None,
    ) -> MemoryItem:
        """Realtime write path — never calls an LLM.

        Stores raw content with an inline embedding and marks it
        ``pending_consolidation``; recall sees it immediately with
        ``confidence=0.5`` until the consolidator re-types it.
        """
        return await self.client.add_fast(
            content,
            memory_type=MemoryType.FACTUAL,
            metadata=metadata,
            entity_mentions=entity_mentions,
            user_id=self._user(user_id),
        )

    async def add_many(
        self,
        items: list[Any],
        *,
        user_id: str | None = None,
    ) -> list[MemoryItem]:
        """Batch write. Dicts without ``source`` use the LLM-free path."""
        return await self.client.add_many(items, user_id=self._user(user_id))

    async def recall(
        self,
        query: str,
        top_k: int = 5,
        as_of: datetime | None = None,
        *,
        user_id: str | None = None,
        entity: str | None = None,
        min_confidence: float | None = None,
        include_third_party: bool = True,
    ) -> list[MemoryItem]:
        """Recall currently-valid facts (or those valid at ``as_of``)."""
        return await self.client.search(
            query,
            top_k=top_k,
            memory_type=MemoryType.FACTUAL,
            as_of=as_of,
            compose=True,
            user_id=self._user(user_id),
            entity=entity,
            min_confidence=min_confidence,
            include_third_party=include_third_party,
        )

    async def recall_for_action(
        self,
        query: str,
        top_k: int = 5,
        as_of: datetime | None = None,
        *,
        user_id: str | None = None,
    ) -> list[MemoryItem]:
        """Recall only facts an agent may act on without confirming first."""
        policy = self.client.trust_policy
        candidates = await self.recall(
            query, top_k=top_k * 4, as_of=as_of, user_id=self._user(user_id)
        )
        return [m for m in candidates if policy.is_trusted(m)][:top_k]

    async def confirm(self, memory_id: str, user_id: str | None = None) -> MemoryItem:
        """Graduate a fact after the user said yes. Closes the verify loop."""
        return await self.client.confirm(memory_id, user_id=self._user(user_id))

    async def pending_confirmations(
        self,
        user_id: str | None = None,
        limit: int = 100,
    ) -> list[MemoryItem]:
        """Action-relevant facts that still need a yes from the user."""
        return await self.client.pending_confirmations(
            user_id=self._user(user_id), limit=limit
        )

    async def update_fact(
        self,
        memory_id: str,
        content: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryItem:
        return await self.client.update(memory_id, content=content, metadata=metadata)

    async def list_facts(self, limit: int = 100, user_id: str | None = None) -> list[MemoryItem]:
        await self.client._ensure_init()
        store = self.client._require_store()
        return await store.list_memories(
            user_id=self._user(user_id), memory_type=MemoryType.FACTUAL, limit=limit
        )
