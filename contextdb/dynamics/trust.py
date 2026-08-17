"""Trust write path — slot-based dedupe, corroboration, and supersede.

The core mechanism of the trust model: memories that occupy the same
``(entity_key, attribute_key)`` slot are *about the same thing*, so a second
write into that slot is never just "another memory" — it is either

* the **same value** again → increment ``corroboration_count`` on the
  existing memory (string similarity cannot see this; "Thursday works for
  me" and "yes, Thursday" are the same slot value), or
* a **different value** → the old memory is closed out
  (``valid_until=now``, ``superseded_by=<new id>``) and the new one becomes
  the currently-valid occupant.

Writes without slot keys cannot be matched and are stored as-is; extraction
(Epic 1 prompt) and explicit ``factual.add`` overrides supply the keys.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Literal

from contextdb.core.models import MemoryItem

if TYPE_CHECKING:
    from contextdb.privacy.audit import AuditLogger
    from contextdb.store.sqlite_store import SQLiteStore

WriteOutcome = Literal["added", "corroborated", "superseded"]

# Heuristic action-relevance screen for writes that bypass LLM extraction
# (add_fast, direct factual.add). False negatives here are safe — the fact
# is still stored and recallable; it just doesn't gate actions until
# consolidation re-types it. False positives are also safe — the fact merely
# carries requires_confirmation until corroborated.
_ACTION_RELEVANT = re.compile(
    r"\b(book|booking|booked|appointment|reservation|reserve|schedule|meeting|"
    r"come in|stop by|"
    r"price|pricing|cost|fee|invoice|payment|pay|salary|account number|routing|"
    r"password|passcode|pin\b|"
    r"phone|email|e-mail|address|contact|"
    r"allerg|medication|medicine|diagnos|prescri|doctor|"
    r"legal|lawsuit|contract|court|lawyer|"
    r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"tomorrow|tonight|"
    r"\d{1,2}\s?(?::\d{2})?\s?(?:am|pm))\b",
    re.IGNORECASE,
)

# Content classes that outrank ordinary facts for salience criticality
# (Epic 4): health/safety and legal constraints beat preferences beat trivia.
_CRITICAL_CLASS = re.compile(
    r"\b(allerg|anaphylax|medication|medicine|diagnos|prescri|insulin|"
    r"do not resuscitate|dnr\b|"
    r"legal|lawsuit|court|restraining order|compliance|"
    r"ssn|social security|credit card|routing number)\b",
    re.IGNORECASE,
)


def infer_action_relevant(content: str) -> bool:
    """Keyword heuristic: does this fact gate a real-world action?"""
    return bool(_ACTION_RELEVANT.search(content))


def criticality_class(content: str) -> str:
    """Bucket content into a criticality class for salience boosting."""
    if _CRITICAL_CLASS.search(content):
        return "critical"
    if infer_action_relevant(content):
        return "action"
    return "ordinary"


def normalize_value(content: str) -> str:
    """Canonical form for same-value comparison within a slot."""
    return " ".join(content.casefold().split()).strip(" .!?")


class TrustEngine:
    """Slot-aware write path used by ``factual.add`` and consolidation."""

    def __init__(self, store: SQLiteStore, audit: AuditLogger | None = None) -> None:
        self.store = store
        self.audit = audit

    async def write(
        self,
        item: MemoryItem,
        user_id: str | None = None,
    ) -> tuple[MemoryItem, WriteOutcome]:
        """Store ``item`` with dedupe/contradiction semantics.

        Returns the winning memory (the existing one when corroborating) and
        what happened. Falls back to a plain insert when no slot keys are
        present — you cannot dedupe what you cannot key.
        """
        now = datetime.now(tz=timezone.utc)
        if item.valid_from is None:
            item.valid_from = now

        if not (item.entity_key and item.attribute_key):
            stored = await self.store.add(item)
            await self._log("CREATE", stored.id, user_id, {"trust": "unkeyed"})
            return stored, "added"

        slot = await self.store.list_by_slot(item.entity_key, item.attribute_key)
        current = [c for c in slot if c.id != item.id and c.is_valid_at(now)]

        for candidate in current:
            if normalize_value(candidate.content) == normalize_value(item.content):
                updated = await self.store.update(
                    candidate.id,
                    corroboration_count=candidate.corroboration_count + 1,
                )
                await self._log(
                    "CORROBORATE",
                    candidate.id,
                    user_id,
                    {
                        "corroboration_count": updated.corroboration_count,
                        "entity": item.entity_key,
                        "attribute": item.attribute_key,
                    },
                )
                return updated, "corroborated"

        stored = await self.store.add(item)
        outcome: WriteOutcome = "added"
        for candidate in current:
            await self.store.update(
                candidate.id,
                valid_until=now,
                superseded_by=stored.id,
            )
            await self._log(
                "SUPERSEDE",
                candidate.id,
                user_id,
                {
                    "superseded_by": stored.id,
                    "entity": item.entity_key,
                    "attribute": item.attribute_key,
                    "old_value": candidate.content,
                    "new_value": stored.content,
                },
            )
            outcome = "superseded"
        await self._log(
            "CREATE",
            stored.id,
            user_id,
            {"trust": outcome, "entity": item.entity_key, "attribute": item.attribute_key},
        )
        return stored, outcome

    async def _log(
        self,
        operation: str,
        memory_id: str,
        user_id: str | None,
        details: dict[str, object],
    ) -> None:
        if self.audit is not None:
            await self.audit.log(
                operation=operation,
                memory_id=memory_id,
                user_id=user_id,
                details=details,
            )
