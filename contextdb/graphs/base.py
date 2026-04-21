"""Shared scaffolding for graph implementations.

Each graph stores its edges in its own SQLite table (same file as the memory
store) so the free-tier deployment stays zero-config. :class:`BaseGraph`
defines the contract and supplies a small helper for table creation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from contextdb.core.models import Edge
    from contextdb.store.sqlite_store import SQLiteStore


class BaseGraph(ABC):
    """Common contract for all graphs over memory items."""

    def __init__(self, store: SQLiteStore) -> None:
        self.store = store

    @abstractmethod
    async def initialize(self) -> None:
        """Create tables / indices. Idempotent."""

    @abstractmethod
    async def add_node(self, memory_id: str, data: dict[str, Any]) -> None: ...

    @abstractmethod
    async def add_edge(self, edge: Edge) -> None: ...

    @abstractmethod
    async def get_neighbors(
        self,
        memory_id: str,
        depth: int = 1,
        max_results: int = 20,
    ) -> list[tuple[str, float]]: ...

    @abstractmethod
    async def remove_node(self, memory_id: str) -> None: ...

    @abstractmethod
    async def get_edges(self, memory_id: str) -> list[Edge]: ...
