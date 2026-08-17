"""LiveKit Agents integration — pre-LLM memory hook (Epic 8).

Same contract as the Pipecat bridge, shaped for LiveKit Agents: store each
final user turn via the LLM-free fast path, and inject budgeted,
wrapper-demoted recall before the LLM call. No hard ``livekit`` dependency.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from contextdb.integrations.act import VerifyBeforeAct
from contextdb.integrations.prompting import render_recalled_context

if TYPE_CHECKING:
    from contextdb.client import ContextDB
    from contextdb.core.models import MemoryItem


class ContextDBLiveKitMemory:
    """Memory hook for LiveKit Agents.

    Usage::

        memory = ContextDBLiveKitMemory(db, user_id=caller_id)

        # session.on("user_speech_committed") -> memory.on_user_turn(text)
        # before llm.chat(): system_context = await memory.pre_llm_hook(query)
    """

    def __init__(
        self,
        client: ContextDB,
        user_id: str | None = None,
        *,
        top_k: int = 5,
        max_tokens: int = 512,
    ) -> None:
        self.client = client
        self.user_id = user_id
        self.top_k = top_k
        self.max_tokens = max_tokens
        self.gate = VerifyBeforeAct(client, top_k=top_k, max_tokens=max_tokens)

    async def on_user_turn(self, text: str) -> MemoryItem | None:
        """Store a final user turn (LLM-free; safe inside a voice loop)."""
        text = text.strip()
        if not text:
            return None
        return await self.client.add_fast(text)

    async def pre_llm_hook(self, query: str, *, for_action: bool = False) -> str:
        """Budgeted, wrapper-demoted recall to inject before the LLM call."""
        if for_action:
            decision = await self.gate.decide(query)
            return decision.context if decision.kind == "act" else ""
        items = await self.client.factual.recall(query, top_k=self.top_k)
        return render_recalled_context(items, max_tokens=self.max_tokens)

    async def augment_messages(
        self, messages: list[dict[str, Any]], query: str
    ) -> list[dict[str, Any]]:
        """Return ``messages`` with the wrapped recall block prepended."""
        context = await self.pre_llm_hook(query)
        if not context:
            return messages
        return [{"role": "system", "content": context}, *messages]
