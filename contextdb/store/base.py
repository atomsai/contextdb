"""Abstract storage backend contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

from contextdb.core.exceptions import StaleReadError
from contextdb.core.models import MemoryConsistencyToken, MemoryStatus

if TYPE_CHECKING:
    from contextlib import AbstractAsyncContextManager

    from contextdb.core.models import MemoryItem, MemoryType


class BaseStore(ABC):
    """Abstract base for all persistent memory stores.

    Implementations must be async and safe to call concurrently from one
    event loop. They are not required to be safe across processes.
    """

    @abstractmethod
    async def initialize(self) -> None:
        """Create schema / open connections. Idempotent."""

    @abstractmethod
    async def add(self, item: MemoryItem) -> MemoryItem:
        """Persist a memory and return the stored copy."""

    @abstractmethod
    async def get(self, memory_id: str) -> MemoryItem | None:
        """Fetch one memory by id; increments access counters."""

    @abstractmethod
    async def update(self, memory_id: str, **kwargs: object) -> MemoryItem:
        """Partial update; unknown keys raise ValueError."""

    @abstractmethod
    async def delete(self, memory_id: str, hard: bool = False) -> bool:
        """Soft delete by default (status=DELETED); ``hard=True`` removes the row.

        Returns True when a row was actually affected; False for missing or
        out-of-scope ids.
        """

    @abstractmethod
    async def search_by_embedding(
        self,
        embedding: list[float],
        top_k: int = 10,
        filters: dict[str, object] | None = None,
        user_id: str | None = None,
    ) -> list[MemoryItem]:
        """Return top-k most similar memories by cosine similarity.

        ``user_id`` filters the result. A store constructed with a fixed
        user scope ignores a conflicting per-call value and stays scoped.
        """

    @abstractmethod
    async def list_memories(
        self,
        user_id: str | None = None,
        memory_type: MemoryType | None = None,
        status: MemoryStatus = MemoryStatus.ACTIVE,
        limit: int = 100,
        offset: int = 0,
    ) -> list[MemoryItem]:
        """List memories with optional filters."""

    @abstractmethod
    async def count(self, user_id: str | None = None) -> int:
        """Count active memories, optionally scoped to a user."""

    @abstractmethod
    async def close(self) -> None:
        """Release resources."""

    @asynccontextmanager
    async def mutation(self) -> AsyncIterator[None]:
        """Wrap one logical mutation.

        Networked stores override this to make the memory change, revision
        bump, and audit append one transaction. Local stores may rely on their
        existing per-operation commits.
        """
        yield

    async def consistency_token(self) -> MemoryConsistencyToken:
        """Return the latest consistency floor visible to this scoped store."""
        return MemoryConsistencyToken(memory_version=0)

    async def require_consistency(
        self,
        *,
        min_memory_version: int | None = None,
        min_wal_lsn: str | None = None,
    ) -> MemoryConsistencyToken:
        token = await self.consistency_token()
        version_stale = (
            min_memory_version is not None
            and token.memory_version < min_memory_version
        )
        wal_unavailable = (
            min_wal_lsn is not None
            and token.primary_wal_lsn is None
        )
        if version_stale or wal_unavailable:
            raise StaleReadError(
                required_version=min_memory_version,
                current_version=token.memory_version,
                required_wal_lsn=min_wal_lsn,
                current_wal_lsn=token.primary_wal_lsn,
            )
        return token

    def _require_conn(self) -> Any:
        """SQLite/Postgres connection adapter used by graphs and the audit log."""
        raise NotImplementedError

    async def get_raw(self, memory_id: str) -> MemoryItem | None:
        """Fetch without bumping access counters. Default: :meth:`get`."""
        return await self.get(memory_id)

    async def list_by_slot(
        self,
        entity_key: str,
        attribute_key: str,
        status: MemoryStatus | None = MemoryStatus.ACTIVE,
        user_id: str | None = None,
    ) -> list[MemoryItem]:
        raise NotImplementedError

    async def list_by_entity(
        self,
        entity_key: str,
        status: MemoryStatus | None = MemoryStatus.ACTIVE,
        user_id: str | None = None,
    ) -> list[MemoryItem]:
        raise NotImplementedError

    async def list_pending_consolidation(self, limit: int = 100) -> list[MemoryItem]:
        raise NotImplementedError

    async def count_any_status(self, user_id: str) -> int:
        raise NotImplementedError

    async def count_by_type(self, user_id: str | None = None) -> dict[str, int]:
        raise NotImplementedError

    def iter_memories(
        self,
        user_id: str | None = None,
        memory_type: MemoryType | None = None,
        status: MemoryStatus | None = MemoryStatus.ACTIVE,
        batch_size: int = 500,
    ) -> AsyncIterator[MemoryItem]:
        raise NotImplementedError

    async def delete_older_than(
        self,
        iso_cutoff: str,
        user_id: str | None = None,
        hard: bool = True,
    ) -> int:
        raise NotImplementedError

    async def index_ids(self) -> set[str]:
        raise NotImplementedError

    async def slot_lock(
        self,
        entity_key: str,
        attribute_key: str,
        user_id: str | None = None,
    ) -> AbstractAsyncContextManager[Any]:
        """Mutual exclusion for one (user, tenant, entity, attribute) slot.

        Networked stores must serialize across processes (e.g. a Postgres
        advisory lock), not just within one event loop.
        """
        raise NotImplementedError

    def audit_lock(self) -> AbstractAsyncContextManager[Any]:
        """Serialize audit-chain appends.

        The chain is read-modify-write (latest hash becomes the next
        entry's ``previous_hash``), so concurrent appends must be fully
        ordered — within one process and, for networked stores, across
        processes. Single-process stores return a per-instance lock.
        """
        raise NotImplementedError
