"""Abstract storage backend contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from contextdb.core.models import MemoryStatus

if TYPE_CHECKING:
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
    async def delete(self, memory_id: str, hard: bool = False) -> None:
        """Soft delete by default (status=DELETED); ``hard=True`` removes the row."""

    @abstractmethod
    async def search_by_embedding(
        self,
        embedding: list[float],
        top_k: int = 10,
        filters: dict[str, object] | None = None,
    ) -> list[MemoryItem]:
        """Return top-k most similar memories by cosine similarity."""

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
