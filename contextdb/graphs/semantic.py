"""Semantic graph — edges between memories with high embedding similarity."""

from __future__ import annotations

import json
from datetime import datetime
from typing import TYPE_CHECKING, Any

import numpy as np

from contextdb.core.models import Edge, MemoryStatus
from contextdb.graphs.base import BaseGraph

if TYPE_CHECKING:
    from contextdb.store.sqlite_store import SQLiteStore

_SCHEMA = """
CREATE TABLE IF NOT EXISTS semantic_edges (
    source_id TEXT NOT NULL,
    target_id TEXT NOT NULL,
    weight REAL NOT NULL,
    metadata TEXT DEFAULT '{}',
    created_at TEXT NOT NULL,
    PRIMARY KEY (source_id, target_id)
);
CREATE INDEX IF NOT EXISTS idx_semantic_source ON semantic_edges(source_id);
CREATE INDEX IF NOT EXISTS idx_semantic_target ON semantic_edges(target_id);
"""


def _cosine(a: list[float], b: list[float]) -> float:
    va = np.asarray(a, dtype=np.float32)
    vb = np.asarray(b, dtype=np.float32)
    na = float(np.linalg.norm(va))
    nb = float(np.linalg.norm(vb))
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


class SemanticGraph(BaseGraph):
    """Edges represent cosine similarity above a threshold."""

    def __init__(self, store: SQLiteStore, threshold: float = 0.6) -> None:
        super().__init__(store)
        self.threshold = threshold

    async def initialize(self) -> None:
        conn = self.store._require_conn()
        await conn.executescript(_SCHEMA)
        await conn.commit()

    async def add_node(self, memory_id: str, data: dict[str, Any]) -> None:
        embedding = data.get("embedding")
        if embedding is None:
            return
        similar = await self.store.search_by_embedding(embedding, top_k=50)
        for item in similar:
            if item.id == memory_id or item.embedding is None:
                continue
            if item.status != MemoryStatus.ACTIVE:
                continue
            sim = _cosine(embedding, item.embedding)
            if sim >= self.threshold:
                await self.add_edge(
                    Edge(
                        source_id=memory_id,
                        target_id=item.id,
                        graph_type="semantic",
                        weight=sim,
                    )
                )

    async def add_edge(self, edge: Edge) -> None:
        conn = self.store._require_conn()
        await conn.execute(
            "INSERT OR REPLACE INTO semantic_edges "
            "(source_id, target_id, weight, metadata, created_at) VALUES (?,?,?,?,?)",
            (
                edge.source_id,
                edge.target_id,
                edge.weight,
                json.dumps(edge.metadata),
                edge.created_at.isoformat(),
            ),
        )
        # Bidirectional for undirected similarity semantics.
        await conn.execute(
            "INSERT OR REPLACE INTO semantic_edges "
            "(source_id, target_id, weight, metadata, created_at) VALUES (?,?,?,?,?)",
            (
                edge.target_id,
                edge.source_id,
                edge.weight,
                json.dumps(edge.metadata),
                edge.created_at.isoformat(),
            ),
        )
        await conn.commit()

    async def get_neighbors(
        self,
        memory_id: str,
        depth: int = 1,
        max_results: int = 20,
    ) -> list[tuple[str, float]]:
        conn = self.store._require_conn()
        visited: dict[str, float] = {memory_id: 0.0}
        frontier: list[tuple[str, float]] = [(memory_id, 1.0)]
        for _ in range(depth):
            next_frontier: list[tuple[str, float]] = []
            for node, cum in frontier:
                cursor = await conn.execute(
                    "SELECT target_id, weight FROM semantic_edges WHERE source_id = ?",
                    (node,),
                )
                rows = await cursor.fetchall()
                for row in rows:
                    tid = row["target_id"]
                    w = cum * float(row["weight"])
                    if tid in visited and visited[tid] >= w:
                        continue
                    visited[tid] = w
                    next_frontier.append((tid, w))
            frontier = next_frontier
        visited.pop(memory_id, None)
        ranked = sorted(visited.items(), key=lambda kv: kv[1], reverse=True)
        return ranked[:max_results]

    async def remove_node(self, memory_id: str) -> None:
        conn = self.store._require_conn()
        await conn.execute(
            "DELETE FROM semantic_edges WHERE source_id = ? OR target_id = ?",
            (memory_id, memory_id),
        )
        await conn.commit()

    async def get_edges(self, memory_id: str) -> list[Edge]:
        conn = self.store._require_conn()
        cursor = await conn.execute(
            "SELECT source_id, target_id, weight, metadata, created_at "
            "FROM semantic_edges WHERE source_id = ?",
            (memory_id,),
        )
        rows = await cursor.fetchall()
        return [
            Edge(
                source_id=row["source_id"],
                target_id=row["target_id"],
                graph_type="semantic",
                weight=float(row["weight"]),
                metadata=json.loads(row["metadata"] or "{}"),
                created_at=datetime.fromisoformat(row["created_at"]),
            )
            for row in rows
        ]
