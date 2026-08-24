"""Trust write path — slot-based dedupe, corroboration, and supersede.

The core mechanism of the trust model: memories that occupy the same
``(entity_key, attribute_key)`` slot are *about the same thing*, so a second
write into that slot is never just "another memory" — it is either

* the **same value** again from a *new* speaker → add them to
  ``corroborated_by`` (repeats from the same speaker do not count);
* the **same value** from the same speaker → no-op on the count;
* a **different value from the same speaker** → the old memory is closed
  out (``valid_until=now``, ``superseded_by=<new id>``) unless the
  incoming write is a pending raw racing a newer typed fact;
* a **different value from an independent speaker** → the slot is
  *contested*: both values stay current and neither may gate an action
  until ``confirm()``. Last-write-wins across speakers is how agents
  invent facts.

Writes without slot keys cannot be matched and are stored as-is; the
deterministic slotter and explicit ``factual.add`` overrides supply the keys.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Literal

from contextdb.core.clock import Clock, utc_now
from contextdb.core.exceptions import (
    EvolutionOperationConflictError,
    EvolutionTargetNotFoundError,
    EvolutionTargetRequiredError,
)
from contextdb.core.models import (
    EvolutionOperation,
    EvolutionOutcome,
    MemoryItem,
    MemoryStatus,
)
from contextdb.core.slots import Slot, canonical_slot_value, canonicalize_slot

if TYPE_CHECKING:
    from contextdb.privacy.audit import AuditLogger
    from contextdb.store.base import BaseStore

WriteOutcome = Literal[
    "added",
    "corroborated",
    "superseded",
    "ignored",
    "contested",
    "noop",
]


@dataclass(frozen=True)
class EvolutionWrite:
    """Internal transaction result completed with a token by ``ContextDB``."""

    applied_operation: EvolutionOperation
    outcome: EvolutionOutcome
    memory: MemoryItem | None
    previous_memory_ids: tuple[str, ...] = ()
    noop_reason: str | None = None

# Heuristic action-relevance screen for writes that bypass LLM extraction
# (add_fast, direct factual.add). False negatives here are safe — the fact
# is still stored and recallable; it just doesn't gate actions until
# consolidation re-types it. False positives are also safe — the fact merely
# carries requires_confirmation until corroborated.
# NOTE on regex shape: stem alternatives (book, allerg, prescri, ...) use
# prefix matching — a trailing \b after the group would never fire for
# "allergy"/"booking" because the following character is a word character.
# Alternatives that MUST terminate (pin, am/pm times) carry their own \b.
_ACTION_RELEVANT = re.compile(
    r"\b(?:book|appointment|reservation|reserve|schedule|meeting|"
    r"come in|stop by|"
    r"price|pricing|cost|fee|invoice|payment|pay|salary|account number|routing|"
    r"password|passcode|pin\b|"
    r"phone|email|e-mail|address|contact|"
    r"allerg|medication|medicine|diagnos|prescri|doctor|"
    r"legal|lawsuit|contract|court|lawyer|"
    r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"tomorrow|tonight|"
    r"\d{1,2}\s?(?::\d{2})?\s?(?:am|pm)\b)",
    re.IGNORECASE,
)

# Content classes that outrank ordinary facts for salience criticality
# (Epic 4): health/safety and legal constraints beat preferences beat trivia.
_CRITICAL_CLASS = re.compile(
    r"\b(?:allerg|anaphylax|medication|medicine|diagnos|prescri|insulin|"
    r"do not resuscitate|dnr\b|"
    r"legal|lawsuit|court|restraining order|compliance|"
    r"ssn|social security|credit card|routing number)",
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


def normalize_value(content: str, slot: Slot | None = None) -> str:
    """Canonical form for same-value comparison within a slot."""
    return canonical_slot_value(content, slot)


def speaker_id(
    user_id: str | None,
    session_id: str | None = None,
    agent_id: str | None = None,
) -> str:
    """Stable identity of *who* is asserting a value.

    Session wins when present (two sessions of the same user are
    independent evidence); otherwise user; otherwise agent; otherwise
    a sentinel so anonymous writes still occupy a bucket.
    """
    if session_id:
        return f"session:{session_id}"
    if user_id:
        return f"user:{user_id}"
    if agent_id:
        return f"agent:{agent_id}"
    return "anonymous"


class TrustEngine:
    """Slot-aware write path used by ``factual.add`` and consolidation."""

    def __init__(
        self,
        store: BaseStore,
        audit: AuditLogger | None = None,
        clock: Clock = utc_now,
    ) -> None:
        self.store = store
        self.audit = audit
        self.clock = clock

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
        incoming_speaker = speaker_id(user_id, item.session_id, item.agent_id)
        if incoming_speaker not in item.corroborated_by:
            item.corroborated_by = [incoming_speaker, *item.corroborated_by]

        slot = canonicalize_slot(item.entity_key, item.attribute_key)
        if slot is not None:
            item.entity_key = slot.entity
            item.attribute_key = slot.attribute
            item.slot_class = slot.slot_class
            if item.slot_value is None:
                item.slot_value = canonical_slot_value(item.content, slot)

        if not (item.entity_key and item.attribute_key):
            if item.valid_from is None:
                item.valid_from = self.clock()
            stored = await self.store.add(item)
            await self._log("CREATE", stored.id, user_id, {"trust": "unkeyed"})
            return stored, "added"

        lock = await self.store.slot_lock(
            item.entity_key, item.attribute_key, user_id=user_id
        )
        async with lock:
            now = self.clock()
            item.valid_from = now
            return await self._write_locked(item, user_id, now, slot)

    async def _write_locked(
        self,
        item: MemoryItem,
        user_id: str | None,
        now: datetime,
        slot: Slot | None,
    ) -> tuple[MemoryItem, WriteOutcome]:
        slot_rows = await self.store.list_by_slot(
            item.entity_key or "", item.attribute_key or "", user_id=user_id
        )
        current = [c for c in slot_rows if c.id != item.id and c.is_valid_at(now)]
        max_gen = max((c.write_generation for c in slot_rows), default=0)

        # Pending raws must not supersede a newer typed fact (consolidator race).
        if item.pending_consolidation and any(
            (not c.pending_consolidation) and c.write_generation >= item.write_generation
            for c in current
        ):
            stored = await self.store.add(item)
            await self._log(
                "CREATE",
                stored.id,
                user_id,
                {"trust": "pending_ignored_for_supersede", "generation": max_gen},
            )
            return stored, "ignored"

        incoming_value = item.slot_value or normalize_value(item.content, slot)
        incoming_speaker = speaker_id(user_id, item.session_id, item.agent_id)

        for candidate in current:
            cand_value = candidate.slot_value or normalize_value(candidate.content, slot)
            if cand_value != incoming_value:
                continue
            speakers = list(candidate.corroborated_by)
            if incoming_speaker in speakers:
                await self._log_noop(
                    candidate.id,
                    user_id,
                    reason="same_speaker_same_value",
                    entity=item.entity_key,
                    attribute=item.attribute_key,
                )
                return candidate, "noop"
            speakers.append(incoming_speaker)
            updated = await self.store.update(
                candidate.id,
                corroboration_count=len(speakers),
                corroborated_by=speakers,
            )
            await self._log(
                "CORROBORATE",
                candidate.id,
                user_id,
                {
                    "corroboration_count": updated.corroboration_count,
                    "independent": len(speakers),
                    "speaker": incoming_speaker,
                    "entity": item.entity_key,
                    "attribute": item.attribute_key,
                },
            )
            return updated, "corroborated"

        same_speaker = [
            c for c in current if incoming_speaker in (c.corroborated_by or [])
        ]
        other_speaker = [
            c for c in current if incoming_speaker not in (c.corroborated_by or [])
        ]

        item.write_generation = max_gen + 1
        # Independent speakers asserting different values: contest, do not
        # last-write-win. Same-speaker corrections still supersede.
        if other_speaker and not same_speaker and not item.pending_consolidation:
            item.contested = True
            stored = await self.store.add(item)
            for candidate in other_speaker:
                await self.store.update(candidate.id, contested=True)
                await self._log(
                    "CONFLICT",
                    candidate.id,
                    user_id,
                    {
                        "contested_by": stored.id,
                        "entity": item.entity_key,
                        "attribute": item.attribute_key,
                        "old_value": candidate.content,
                        "new_value": stored.content,
                        "speaker": incoming_speaker,
                    },
                )
            await self._log(
                "CREATE",
                stored.id,
                user_id,
                {
                    "trust": "contested",
                    "entity": item.entity_key,
                    "attribute": item.attribute_key,
                },
            )
            return stored, "contested"

        stored = await self.store.add(item)
        outcome: WriteOutcome = "added"
        for candidate in current:
            if item.pending_consolidation and not candidate.pending_consolidation:
                continue
            await self.store.update(
                candidate.id,
                valid_until=now,
                superseded_by=stored.id,
                contested=False,
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
                    "generation": item.write_generation,
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

    async def evolve_write(
        self,
        item: MemoryItem,
        operation: EvolutionOperation,
        user_id: str | None = None,
        *,
        target_memory_id: str | None = None,
    ) -> EvolutionWrite:
        """Apply an explicit ADD or UPDATE under the normal per-slot lock.

        This method deliberately does not open a store transaction. The
        client owns the outer mutation so memory rows, revision bumps, and
        audit entries share one Postgres transaction.
        """
        if operation not in {EvolutionOperation.ADD, EvolutionOperation.UPDATE}:
            raise ValueError("evolve_write supports only add and update")

        incoming_speaker = speaker_id(user_id, item.session_id, item.agent_id)
        if incoming_speaker not in item.corroborated_by:
            item.corroborated_by = [incoming_speaker, *item.corroborated_by]

        slot = canonicalize_slot(item.entity_key, item.attribute_key)
        if target_memory_id is not None:
            target = await self._evolution_target(target_memory_id, user_id)
            target_slot = canonicalize_slot(target.entity_key, target.attribute_key)
            if target_slot is None:
                raise EvolutionTargetRequiredError(
                    "target memory does not resolve to an entity/attribute slot"
                )
            if slot is not None and (
                slot.entity,
                slot.attribute,
            ) != (
                target_slot.entity,
                target_slot.attribute,
            ):
                raise EvolutionOperationConflictError(
                    "target memory and requested entity/attribute identify different slots"
                )
            slot = target_slot

        if slot is not None:
            item.entity_key = slot.entity
            item.attribute_key = slot.attribute
            item.slot_class = slot.slot_class
            if item.slot_value is None:
                item.slot_value = canonical_slot_value(item.content, slot)

        if operation == EvolutionOperation.UPDATE and slot is None:
            raise EvolutionTargetRequiredError(
                "UPDATE requires entity+attribute or a target memory with a slot"
            )

        if slot is None:
            if item.valid_from is None:
                item.valid_from = self.clock()
            stored = await self.store.add(item)
            await self._log(
                "CREATE",
                stored.id,
                user_id,
                {"operation": operation.value, "trust": "explicit_unkeyed"},
            )
            return EvolutionWrite(
                applied_operation=EvolutionOperation.ADD,
                outcome=EvolutionOutcome.ADDED,
                memory=stored,
            )

        lock = await self.store.slot_lock(
            slot.entity,
            slot.attribute,
            user_id=user_id,
        )
        async with lock:
            now = self.clock()
            item.valid_from = now
            if target_memory_id is not None:
                # Re-read after acquiring the slot lock. A target resolved
                # before waiting may have been deleted by the prior holder.
                target = await self._evolution_target(
                    target_memory_id,
                    user_id,
                    moment=now,
                )
                target_slot = canonicalize_slot(
                    target.entity_key,
                    target.attribute_key,
                )
                if target_slot is None or (
                    target_slot.entity,
                    target_slot.attribute,
                ) != (slot.entity, slot.attribute):
                    raise EvolutionOperationConflictError(
                        "target memory no longer resolves to the requested slot"
                    )
            return await self._evolve_write_locked(
                item,
                operation,
                user_id,
                now,
                slot,
            )

    async def _evolve_write_locked(
        self,
        item: MemoryItem,
        operation: EvolutionOperation,
        user_id: str | None,
        now: datetime,
        slot: Slot,
    ) -> EvolutionWrite:
        slot_rows = await self.store.list_by_slot(
            slot.entity,
            slot.attribute,
            user_id=user_id,
        )
        current = [
            candidate
            for candidate in slot_rows
            if candidate.id != item.id and candidate.is_valid_at(now)
        ]
        if len(current) > 100:
            raise EvolutionOperationConflictError(
                "slot has too many current occupants for bounded lineage"
            )
        previous_ids = tuple(sorted(candidate.id for candidate in current))
        max_generation = max(
            (candidate.write_generation for candidate in slot_rows),
            default=0,
        )
        incoming_value = item.slot_value or normalize_value(item.content, slot)
        incoming_speaker = speaker_id(user_id, item.session_id, item.agent_id)

        if operation == EvolutionOperation.ADD:
            if not current:
                item.write_generation = max_generation + 1
                stored = await self.store.add(item)
                await self._log(
                    "CREATE",
                    stored.id,
                    user_id,
                    {
                        "operation": operation.value,
                        "trust": "explicit_add",
                        "entity": slot.entity,
                        "attribute": slot.attribute,
                    },
                )
                return EvolutionWrite(
                    applied_operation=EvolutionOperation.ADD,
                    outcome=EvolutionOutcome.ADDED,
                    memory=stored,
                )

            if len(current) != 1:
                raise EvolutionOperationConflictError(
                    f"ADD requires an empty or unambiguous slot: "
                    f"{slot.entity}/{slot.attribute}"
                )
            candidate = current[0]
            candidate_value = candidate.slot_value or normalize_value(
                candidate.content,
                slot,
            )
            if candidate_value != incoming_value:
                raise EvolutionOperationConflictError(
                    f"ADD conflicts with occupied slot "
                    f"{slot.entity}/{slot.attribute}"
                )
            speakers = list(candidate.corroborated_by)
            if incoming_speaker in speakers:
                reason = "same_speaker_same_value"
                await self._log_noop(
                    candidate.id,
                    user_id,
                    reason=reason,
                    entity=slot.entity,
                    attribute=slot.attribute,
                )
                return EvolutionWrite(
                    applied_operation=EvolutionOperation.NOOP,
                    outcome=EvolutionOutcome.NOOP,
                    memory=candidate,
                    noop_reason=reason,
                )
            speakers.append(incoming_speaker)
            updated = await self.store.update(
                candidate.id,
                corroboration_count=len(speakers),
                corroborated_by=speakers,
            )
            await self._log(
                "CORROBORATE",
                candidate.id,
                user_id,
                {
                    "operation": operation.value,
                    "corroboration_count": updated.corroboration_count,
                    "independent": len(speakers),
                    "speaker": incoming_speaker,
                    "entity": slot.entity,
                    "attribute": slot.attribute,
                },
            )
            return EvolutionWrite(
                applied_operation=EvolutionOperation.UPDATE,
                outcome=EvolutionOutcome.UPDATED,
                memory=updated,
                previous_memory_ids=(candidate.id,),
            )

        if not current:
            raise EvolutionTargetNotFoundError(
                f"{slot.entity}/{slot.attribute}"
            )

        for candidate in current:
            candidate_value = candidate.slot_value or normalize_value(
                candidate.content,
                slot,
            )
            if candidate_value != incoming_value:
                continue
            speakers = list(candidate.corroborated_by)
            if incoming_speaker in speakers:
                reason = "same_speaker_same_value"
                await self._log_noop(
                    candidate.id,
                    user_id,
                    reason=reason,
                    entity=slot.entity,
                    attribute=slot.attribute,
                )
                return EvolutionWrite(
                    applied_operation=EvolutionOperation.NOOP,
                    outcome=EvolutionOutcome.NOOP,
                    memory=candidate,
                    noop_reason=reason,
                )
            speakers.append(incoming_speaker)
            updated = await self.store.update(
                candidate.id,
                corroboration_count=len(speakers),
                corroborated_by=speakers,
            )
            await self._log(
                "CORROBORATE",
                candidate.id,
                user_id,
                {
                    "operation": operation.value,
                    "corroboration_count": updated.corroboration_count,
                    "independent": len(speakers),
                    "speaker": incoming_speaker,
                    "entity": slot.entity,
                    "attribute": slot.attribute,
                },
            )
            return EvolutionWrite(
                applied_operation=EvolutionOperation.UPDATE,
                outcome=EvolutionOutcome.UPDATED,
                memory=updated,
                previous_memory_ids=previous_ids,
            )

        same_speaker = [
            candidate
            for candidate in current
            if incoming_speaker in candidate.corroborated_by
        ]
        other_speaker = [
            candidate
            for candidate in current
            if incoming_speaker not in candidate.corroborated_by
        ]
        item.write_generation = max_generation + 1

        if other_speaker and not same_speaker:
            item.contested = True
            stored = await self.store.add(item)
            for candidate in other_speaker:
                await self.store.update(candidate.id, contested=True)
                await self._log(
                    "CONFLICT",
                    candidate.id,
                    user_id,
                    {
                        "contested_by": stored.id,
                        "entity": slot.entity,
                        "attribute": slot.attribute,
                        "old_value": candidate.content,
                        "new_value": stored.content,
                        "speaker": incoming_speaker,
                    },
                )
            await self._log(
                "CREATE",
                stored.id,
                user_id,
                {
                    "operation": operation.value,
                    "trust": "contested",
                    "entity": slot.entity,
                    "attribute": slot.attribute,
                },
            )
            return EvolutionWrite(
                applied_operation=EvolutionOperation.UPDATE,
                outcome=EvolutionOutcome.CONTESTED,
                memory=stored,
                previous_memory_ids=previous_ids,
            )

        stored = await self.store.add(item)
        for candidate in current:
            await self.store.update(
                candidate.id,
                valid_until=now,
                superseded_by=stored.id,
                contested=False,
            )
            await self._log(
                "SUPERSEDE",
                candidate.id,
                user_id,
                {
                    "superseded_by": stored.id,
                    "entity": slot.entity,
                    "attribute": slot.attribute,
                    "old_value": candidate.content,
                    "new_value": stored.content,
                    "generation": item.write_generation,
                },
            )
        await self._log(
            "CREATE",
            stored.id,
            user_id,
            {
                "operation": operation.value,
                "trust": "superseded",
                "entity": slot.entity,
                "attribute": slot.attribute,
            },
        )
        return EvolutionWrite(
            applied_operation=EvolutionOperation.UPDATE,
            outcome=EvolutionOutcome.UPDATED,
            memory=stored,
            previous_memory_ids=previous_ids,
        )

    async def _evolution_target(
        self,
        memory_id: str,
        user_id: str | None,
        *,
        moment: datetime | None = None,
    ) -> MemoryItem:
        target = await self.store.get_raw(memory_id)
        current_at = moment or self.clock()
        if (
            target is None
            or target.status != MemoryStatus.ACTIVE
            or (user_id is not None and target.user_id != user_id)
            or not target.is_valid_at(current_at)
        ):
            raise EvolutionTargetNotFoundError(memory_id)
        return target

    async def confirm(
        self,
        memory_id: str,
        user_id: str | None = None,
    ) -> MemoryItem:
        """Graduate a fact via explicit user confirmation.

        This is the writeback the verify-before-act loop was missing: the
        agent asked, the user said yes, and now the fact is trusted under
        any policy that honours ``confirmed``.

        When ``user_id`` is supplied, the target must belong to that user;
        a foreign id raises :class:`MemoryNotFoundError`, indistinguishable
        from a missing one. ``user_id=None`` is the unscoped local/admin
        context and is not an authorization boundary.
        """
        from contextdb.core.exceptions import MemoryNotFoundError

        item = await self.store.get_raw(memory_id)
        if item is None or (user_id is not None and item.user_id != user_id):
            raise MemoryNotFoundError(memory_id)
        now = self.clock()
        speakers = list(item.corroborated_by)
        sid = speaker_id(user_id, item.session_id, item.agent_id)
        if sid not in speakers:
            speakers.append(sid)
        updated = await self.store.update(
            memory_id,
            confirmed=True,
            confirmed_at=now,
            contested=False,
            corroboration_count=len(speakers),
            corroborated_by=speakers,
            epistemic_source="user_stated",
            confidence=max(item.confidence, 0.95),
        )
        # Resolving a contest: the confirmed value closes every other
        # current occupant of the slot.
        if updated.entity_key and updated.attribute_key:
            rivals = await self.store.list_by_slot(
                updated.entity_key, updated.attribute_key, user_id=user_id
            )
            for rival in rivals:
                if rival.id == updated.id or not rival.is_valid_at(now):
                    continue
                await self.store.update(
                    rival.id,
                    valid_until=now,
                    superseded_by=updated.id,
                    contested=False,
                )
                await self._log(
                    "SUPERSEDE",
                    rival.id,
                    user_id,
                    {
                        "superseded_by": updated.id,
                        "entity": updated.entity_key,
                        "attribute": updated.attribute_key,
                        "reason": "confirm_resolved_contest",
                    },
                )
        await self._log(
            "CONFIRM",
            memory_id,
            user_id,
            {"confirmed_at": now.isoformat(), "speaker": sid},
        )
        return updated

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

    async def _log_noop(
        self,
        memory_id: str,
        user_id: str | None,
        *,
        reason: str,
        entity: str | None,
        attribute: str | None,
    ) -> None:
        """Append scoped, content-free NOOP lineage without mutating memory."""
        if self.audit is not None:
            details: dict[str, object] = {
                "operation": EvolutionOperation.NOOP.value,
                "reason": reason,
            }
            slot = canonicalize_slot(entity, attribute)
            if slot is not None:
                details["entity"] = slot.entity
                details["attribute"] = slot.attribute
            await self.audit.log(
                operation="NOOP",
                memory_id=memory_id,
                user_id=user_id,
                details=details,
            )
