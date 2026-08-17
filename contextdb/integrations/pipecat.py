"""Pipecat integration — per-turn memory for realtime voice agents (Epic 8).

Two hard-won rules are baked in:

* **Store per turn, never at disconnect.** A voice process can be killed
  between "caller hangs up" and "disconnect handler runs"; anything buffered
  for end-of-call is lost. :meth:`handle_final_transcript` writes each final
  caller transcript through ``add_fast`` — the LLM-free path that cannot
  block a turn.
* **Recall is injected delimited and demoted.** :meth:`recall_context`
  renders through the Epic 5 wrapper with a token budget.

The module intentionally does NOT import ``pipecat``: the dependency stays
optional and the processor is duck-typed against frame shape (``text`` +
finality markers), which is what makes the seed/recall pair testable
across processes without a telephony stack.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from contextdb.integrations.act import VerifyBeforeAct
from contextdb.integrations.prompting import render_recalled_context

if TYPE_CHECKING:
    from contextdb.client import ContextDB
    from contextdb.core.models import MemoryItem


class ContextDBPipecatProcessor:
    """FrameProcessor-shaped bridge between Pipecat pipelines and ContextDB.

    Usage inside a pipeline::

        processor = ContextDBPipecatProcessor(db, user_id=caller_id)
        # in your frame flow: transcription frames are forwarded to
        # process_frame(); before the LLM node, call recall_context() with
        # the latest caller utterance and prepend it to the LLM context.
    """

    def __init__(
        self,
        client: ContextDB,
        user_id: str | None = None,
        *,
        top_k: int = 5,
        max_tokens: int = 512,
        store_agent_turns: bool = False,
    ) -> None:
        self.client = client
        self.user_id = user_id
        self.top_k = top_k
        self.max_tokens = max_tokens
        self.store_agent_turns = store_agent_turns
        self.gate = VerifyBeforeAct(client, top_k=top_k, max_tokens=max_tokens)

    async def handle_final_transcript(
        self, text: str, role: str = "user"
    ) -> MemoryItem | None:
        """Store one FINAL transcript turn. Call this per turn, not at
        disconnect. Returns None for empty turns and (by default) agent
        turns — the caller's words are the memory substrate."""
        text = text.strip()
        if not text:
            return None
        if role != "user" and not self.store_agent_turns:
            return None
        return await self.client.add_fast(text)

    async def recall_context(self, query: str, *, for_action: bool = False) -> str:
        """Budgeted, wrapper-demoted recall for injection pre-turn.

        ``for_action=True`` restricts to facts the agent may act on without
        confirming first (Epic 1 trust bar).
        """
        if for_action:
            decision = await self.gate.decide(query)
            # Untrusted facts never reach an action-shaped prompt.
            return decision.context if decision.kind == "act" else ""
        items = await self.client.factual.recall(query, top_k=self.top_k)
        return render_recalled_context(items, max_tokens=self.max_tokens)

    async def augment_messages(
        self, messages: list[dict[str, Any]], query: str
    ) -> list[dict[str, Any]]:
        """Return ``messages`` with budgeted recall prepended.

        The recalled block rides in the system slot but ALWAYS inside the
        Epic 5 wrapper — demoted to data, never bare instructions.
        """
        context = await self.recall_context(query)
        if not context:
            return messages
        return [{"role": "system", "content": context}, *messages]

    async def process_frame(self, frame: Any, direction: Any = None) -> None:
        """Duck-typed Pipecat frame handling.

        Stores transcription frames once final; interim frames are skipped
        (they would double-store when the final arrives).
        """
        del direction  # frame direction is irrelevant to memory writes
        frame_name = type(frame).__name__
        if "Interim" in frame_name:
            return
        text = getattr(frame, "text", None)
        if not isinstance(text, str):
            return
        is_final = getattr(frame, "is_final", getattr(frame, "final", True))
        if not is_final:
            return
        role = getattr(frame, "role", "user")
        await self.handle_final_transcript(text, role=str(role))
