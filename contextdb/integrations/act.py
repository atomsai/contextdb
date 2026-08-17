"""Verify-before-act interceptor — stock hosts must consume the flags.

The memory layer types; the host decides. If the *stock* hosts only
print ``REQUIRES CONFIRMATION`` and then put the fact into an
action-shaped prompt anyway, every demo looks like the flags are
decorative. This interceptor is the decision: an untrusted fact never
reaches an action prompt, and a user "yes" calls ``confirm()``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from contextdb.integrations.prompting import render_recalled_context

if TYPE_CHECKING:
    from contextdb.client import ContextDB
    from contextdb.core.models import MemoryItem

ActionKind = Literal["ask", "act", "abstain"]


@dataclass
class ActionDecision:
    """What a host is allowed to do with a recall."""

    kind: ActionKind
    memories: list[MemoryItem]
    context: str
    pending_confirmation: list[str]
    reason: str


class VerifyBeforeAct:
    """Interceptor for Pipecat / LiveKit / MCP action turns."""

    def __init__(self, client: ContextDB, *, top_k: int = 5, max_tokens: int = 512) -> None:
        self.client = client
        self.top_k = top_k
        self.max_tokens = max_tokens

    async def decide(self, query: str) -> ActionDecision:
        """Classify a turn: act on trusted facts, ask on untrusted, or abstain."""
        trusted = await self.client.factual.recall_for_action(query, top_k=self.top_k)
        trusted = [m for m in trusted if not m.injection_suspect]
        if trusted:
            return ActionDecision(
                kind="act",
                memories=trusted,
                context=render_recalled_context(trusted, max_tokens=self.max_tokens),
                pending_confirmation=[],
                reason="trusted facts available",
            )
        recalled = [
            m
            for m in await self.client.factual.recall(query, top_k=self.top_k)
            if not m.injection_suspect
        ]
        if recalled:
            pending = [m.id for m in recalled if m.requires_confirmation]
            return ActionDecision(
                kind="ask",
                memories=recalled,
                context=render_recalled_context(recalled, max_tokens=self.max_tokens),
                pending_confirmation=pending,
                reason="facts present but require confirmation",
            )
        return ActionDecision(
            kind="abstain",
            memories=[],
            context="",
            pending_confirmation=[],
            reason="nothing on file (relevance floor or empty store)",
        )

    async def confirm_pending(self, memory_ids: list[str]) -> list[MemoryItem]:
        """Write back the user's yes. This is the closed loop."""
        confirmed = []
        for mid in memory_ids:
            confirmed.append(await self.client.factual.confirm(mid))
        return confirmed
