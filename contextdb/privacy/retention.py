"""Retention policy enforcement.

Applies the configured :class:`~contextdb.core.models.RetentionPolicy` to
the store: anything older than the per-type TTL gets archived (soft) or
erased (hard, when ``right_to_erasure`` is honored and the user asks).

The enforcer is explicit — nothing is deleted on a schedule unless
:meth:`RetentionManager.enforce` is called. Operators can wire it to a
cron / periodic task, but the library does not start background threads.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

from contextdb.core.models import MemoryStatus, MemoryType, RetentionPolicy

if TYPE_CHECKING:
    from contextdb.privacy.audit import AuditLogger
    from contextdb.store.sqlite_store import SQLiteStore


_TYPE_TO_TTL: dict[MemoryType, str] = {
    MemoryType.FACTUAL: "factual_ttl",
    MemoryType.EXPERIENTIAL: "experiential_ttl",
    MemoryType.WORKING: "working_ttl",
}


class RetentionManager:
    """Apply retention TTLs and honor right-to-erasure requests."""

    def __init__(
        self,
        store: SQLiteStore,
        audit: AuditLogger | None,
        policy: RetentionPolicy,
    ) -> None:
        self.store = store
        self.audit = audit
        self.policy = policy

    def _ttl_for(self, memory_type: MemoryType) -> timedelta | None:
        field = _TYPE_TO_TTL[memory_type]
        ttl = getattr(self.policy, field, None)
        if ttl is None:
            return self.policy.default_ttl
        assert isinstance(ttl, timedelta)
        return ttl

    async def enforce(self, hard: bool = False) -> int:
        """Archive (or hard-delete) memories past their TTL.

        Returns the number of affected rows.
        """
        now = datetime.now(tz=timezone.utc)
        affected = 0
        for memory_type in MemoryType:
            ttl = self._ttl_for(memory_type)
            if ttl is None:
                continue
            cutoff = now - ttl
            memories = await self.store.list_memories(memory_type=memory_type, limit=100000)
            for memory in memories:
                if memory.created_at > cutoff:
                    continue
                if hard:
                    await self.store.delete(memory.id, hard=True)
                    operation = "ERASE"
                else:
                    await self.store.update(memory.id, status=MemoryStatus.ARCHIVED)
                    operation = "ARCHIVE"
                affected += 1
                if self.audit is not None:
                    await self.audit.log(
                        operation=operation,
                        memory_id=memory.id,
                        details={"reason": "retention_ttl", "type": memory_type.value},
                    )
        return affected

    async def erase_user(self, user_id: str) -> int:
        """Honor a right-to-erasure request for a specific user."""
        if not self.policy.right_to_erasure:
            return 0
        memories = await self.store.list_memories(user_id=user_id, limit=100000)
        deleted = 0
        for memory in memories:
            await self.store.delete(memory.id, hard=True)
            deleted += 1
            if self.audit is not None:
                await self.audit.log(
                    operation="ERASE",
                    memory_id=memory.id,
                    user_id=user_id,
                    details={"reason": "right_to_erasure"},
                )
        return deleted
