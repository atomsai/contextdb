"""Migration tools — import from competitors, export to JSON.

Each importer accepts an iterable of dicts in the source system's shape and
translates them into :class:`~contextdb.core.models.MemoryItem` writes. We
keep the mapping forgiving: missing fields fall back to sensible defaults
so a partial dump still loads cleanly.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from contextdb.core.models import MemoryItem, MemoryType

if TYPE_CHECKING:
    from contextdb.client import ContextDB


def _parse_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None


class Mem0Importer:
    """Import from a Mem0 JSON export.

    Mem0 records typically carry ``memory`` (text), ``user_id``, ``metadata``,
    and ISO ``created_at``. Anything beyond that lands in ``metadata``.
    """

    def __init__(self, client: ContextDB) -> None:
        self.client = client

    async def import_records(self, records: Iterable[dict[str, Any]]) -> int:
        count = 0
        for record in records:
            content = record.get("memory") or record.get("content") or ""
            if not content:
                continue
            await self.client.add(
                content=str(content),
                metadata=dict(record.get("metadata", {})),
                event_time=_parse_datetime(record.get("created_at")),
            )
            count += 1
        return count


class ZepImporter:
    """Import from a Zep session dump."""

    def __init__(self, client: ContextDB) -> None:
        self.client = client

    async def import_records(self, records: Iterable[dict[str, Any]]) -> int:
        count = 0
        for record in records:
            content = record.get("content") or record.get("message") or ""
            if not content:
                continue
            meta = dict(record.get("metadata", {}))
            if "role" in record:
                meta.setdefault("role", record["role"])
            await self.client.add(
                content=str(content),
                metadata=meta,
                event_time=_parse_datetime(record.get("created_at")),
            )
            count += 1
        return count


class LangChainImporter:
    """Import from LangChain ``ChatMessageHistory`` dumps."""

    def __init__(self, client: ContextDB) -> None:
        self.client = client

    async def import_records(self, messages: Iterable[dict[str, Any]]) -> int:
        count = 0
        for message in messages:
            text = message.get("content") or message.get("text") or ""
            if not text:
                continue
            role = message.get("type") or message.get("role") or "human"
            await self.client.add(
                content=f"{role}: {text}",
                metadata={"role": role},
            )
            count += 1
        return count


class JSONExporter:
    """Dump the entire store as a JSON array of memory records."""

    def __init__(self, client: ContextDB) -> None:
        self.client = client

    async def export(self, path: str | Path) -> int:
        await self.client._ensure_init()
        store = self.client._require_store()
        memories = await store.list_memories(limit=100000)
        records = [self._to_record(m) for m in memories]
        Path(path).write_text(json.dumps(records, indent=2, default=str))
        return len(records)

    @staticmethod
    def _to_record(item: MemoryItem) -> dict[str, Any]:
        return {
            "id": item.id,
            "content": item.content,
            "memory_type": item.memory_type.value,
            "source": item.source,
            "metadata": item.metadata,
            "event_time": item.event_time.isoformat() if item.event_time else None,
            "created_at": item.created_at.isoformat(),
            "updated_at": item.updated_at.isoformat(),
            "entity_mentions": item.entity_mentions,
            "tags": item.tags,
            "confidence": item.confidence,
            "status": item.status.value,
        }


class JSONImporter:
    """Re-import a :class:`JSONExporter` dump."""

    def __init__(self, client: ContextDB) -> None:
        self.client = client

    async def import_path(self, path: str | Path) -> int:
        records = json.loads(Path(path).read_text())
        assert isinstance(records, list)
        count = 0
        for record in records:
            content = record.get("content")
            if not content:
                continue
            mt = MemoryType(record.get("memory_type", "FACTUAL"))
            await self.client.add(
                content=content,
                memory_type=mt,
                metadata=dict(record.get("metadata", {})),
                event_time=_parse_datetime(record.get("event_time")),
                entity_mentions=list(record.get("entity_mentions", [])),
            )
            count += 1
        return count
