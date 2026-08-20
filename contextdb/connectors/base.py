"""Portable connector contracts for local and hosted ingestion."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol


class ConnectorError(Exception):
    """Base connector failure."""


class ConnectorConfigError(ConnectorError):
    """Source configuration or mapping is invalid."""


class ConnectorSourceError(ConnectorError):
    """Source is unavailable or rejects access."""


@dataclass(frozen=True)
class ConnectorCursor:
    """Stable timestamp-plus-key cursor; timestamp alone can skip ties."""

    updated_at: datetime
    primary_key: str

    def as_dict(self) -> dict[str, str]:
        return {
            "updated_at": self.updated_at.isoformat(),
            "primary_key": self.primary_key,
        }


@dataclass(frozen=True)
class ConnectorRecord:
    """One source record after mapping, before memory formation."""

    source_key: str
    source_version: str
    user_id: str
    content: str
    updated_at: datetime
    deleted: bool = False
    provenance: dict[str, Any] = field(default_factory=dict)


class ConnectorReader(Protocol):
    """Read-only source contract. Scheduling and sinks are host concerns."""

    async def validate(self) -> None: ...

    async def preview(self, *, limit: int = 50) -> list[ConnectorRecord]: ...

    def records(
        self,
        *,
        cursor: ConnectorCursor | None = None,
        limit: int = 100_000,
    ) -> AsyncIterator[list[ConnectorRecord]]: ...
