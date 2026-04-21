"""Temporal graph — edges carry ordering and proximity in event-time."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

from contextdb.core.models import Edge, MemoryItem
from contextdb.graphs.base import BaseGraph

if TYPE_CHECKING:
    from contextdb.store.sqlite_store import SQLiteStore

_SCHEMA = """
CREATE TABLE IF NOT EXISTS temporal_edges (
    source_id TEXT NOT NULL,
    target_id TEXT NOT NULL,
    relation TEXT NOT NULL,
    weight REAL NOT NULL,
    time_diff_seconds REAL NOT NULL,
    metadata TEXT DEFAULT '{}',
    created_at TEXT NOT NULL,
    PRIMARY KEY (source_id, target_id)
);
CREATE INDEX IF NOT EXISTS idx_temporal_source ON temporal_edges(source_id);
"""


class TemporalGraph(BaseGraph):
    """Link memories that occur near each other in event-time."""

    def __init__(
        self,
        store: SQLiteStore,
        proximity_window: timedelta = timedelta(hours=24),
    ) -> None:
        super().__init__(store)
        self.proximity_window = proximity_window

    async def initialize(self) -> None:
        conn = self.store._require_conn()
        await conn.executescript(_SCHEMA)
        await conn.commit()

    async def add_node(self, memory_id: str, data: dict[str, Any]) -> None:
        event_time = data.get("event_time")
        if event_time is None:
            return
        nearby = await self._find_temporally_nearby(memory_id, event_time)
        for other_id, other_time in nearby:
            diff = (event_time - other_time).total_seconds()
            if abs(diff) < 300:
                relation = "CONCURRENT"
            elif diff > 0:
                relation = "AFTER"
            else:
                relation = "BEFORE"
            weight = 1.0 / (1.0 + abs(diff) / 3600.0)
            await self.add_edge(
                Edge(
                    source_id=memory_id,
                    target_id=other_id,
                    graph_type="temporal",
                    weight=weight,
                    metadata={"relation": relation, "time_diff_seconds": diff},
                )
            )

    async def _find_temporally_nearby(
        self, memory_id: str, event_time: datetime
    ) -> list[tuple[str, datetime]]:
        conn = self.store._require_conn()
        window_start = (event_time - self.proximity_window).isoformat()
        window_end = (event_time + self.proximity_window).isoformat()
        cursor = await conn.execute(
            "SELECT id, event_time FROM memories "
            "WHERE id != ? AND event_time IS NOT NULL "
            "AND event_time BETWEEN ? AND ? AND status = 'ACTIVE'",
            (memory_id, window_start, window_end),
        )
        rows = await cursor.fetchall()
        return [(row["id"], datetime.fromisoformat(row["event_time"])) for row in rows]

    async def add_edge(self, edge: Edge) -> None:
        conn = self.store._require_conn()
        meta = edge.metadata or {}
        await conn.execute(
            "INSERT OR REPLACE INTO temporal_edges "
            "(source_id, target_id, relation, weight, time_diff_seconds, metadata, created_at) "
            "VALUES (?,?,?,?,?,?,?)",
            (
                edge.source_id,
                edge.target_id,
                str(meta.get("relation", "ADJACENT")),
                edge.weight,
                float(meta.get("time_diff_seconds", 0.0)),
                json.dumps(meta),
                edge.created_at.isoformat(),
            ),
        )
        await conn.commit()

    async def remove_node(self, memory_id: str) -> None:
        conn = self.store._require_conn()
        await conn.execute(
            "DELETE FROM temporal_edges WHERE source_id = ? OR target_id = ?",
            (memory_id, memory_id),
        )
        await conn.commit()

    async def get_edges(self, memory_id: str) -> list[Edge]:
        conn = self.store._require_conn()
        cursor = await conn.execute(
            "SELECT source_id, target_id, weight, metadata, created_at, relation, "
            "time_diff_seconds FROM temporal_edges WHERE source_id = ?",
            (memory_id,),
        )
        rows = await cursor.fetchall()
        out: list[Edge] = []
        for row in rows:
            meta = json.loads(row["metadata"] or "{}")
            meta.setdefault("relation", row["relation"])
            meta.setdefault("time_diff_seconds", row["time_diff_seconds"])
            out.append(
                Edge(
                    source_id=row["source_id"],
                    target_id=row["target_id"],
                    graph_type="temporal",
                    weight=float(row["weight"]),
                    metadata=meta,
                    created_at=datetime.fromisoformat(row["created_at"]),
                )
            )
        return out

    async def get_neighbors(
        self,
        memory_id: str,
        depth: int = 1,
        max_results: int = 20,
    ) -> list[tuple[str, float]]:
        conn = self.store._require_conn()
        cursor = await conn.execute(
            "SELECT target_id, weight FROM temporal_edges WHERE source_id = ? "
            "ORDER BY weight DESC LIMIT ?",
            (memory_id, max_results),
        )
        rows = await cursor.fetchall()
        return [(row["target_id"], float(row["weight"])) for row in rows]

    async def get_timeline(
        self,
        entity: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> list[MemoryItem]:
        store = self.store
        items = await store.list_memories(limit=10000)
        filtered = [m for m in items if m.event_time is not None]
        if start is not None:
            filtered = [m for m in filtered if m.event_time and m.event_time >= start]
        if end is not None:
            filtered = [m for m in filtered if m.event_time and m.event_time <= end]
        if entity is not None:
            ent = entity.lower()
            filtered = [
                m
                for m in filtered
                if ent in (e.lower() for e in m.entity_mentions) or ent in m.content.lower()
            ]
        filtered.sort(key=lambda m: m.event_time or datetime.min.replace(tzinfo=timezone.utc))
        return filtered
