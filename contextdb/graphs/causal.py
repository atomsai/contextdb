"""Causal graph — LLM-inferred cause/effect links between memories.

The causal relationship is inherently a judgement call, so we ask the LLM
for a structured verdict over a pair of memories. Edges carry a confidence
weight in ``[0, 1]`` and a free-form ``reasoning`` payload.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import TYPE_CHECKING, Any

from contextdb.core.models import Edge
from contextdb.graphs.base import BaseGraph

if TYPE_CHECKING:
    from contextdb.store.sqlite_store import SQLiteStore
    from contextdb.utils.llm import LLMProvider

_SCHEMA = """
CREATE TABLE IF NOT EXISTS causal_edges (
    source_id TEXT NOT NULL,
    target_id TEXT NOT NULL,
    relation TEXT NOT NULL,
    weight REAL NOT NULL,
    reasoning TEXT DEFAULT '',
    metadata TEXT DEFAULT '{}',
    created_at TEXT NOT NULL,
    PRIMARY KEY (source_id, target_id)
);
CREATE INDEX IF NOT EXISTS idx_causal_source ON causal_edges(source_id);
CREATE INDEX IF NOT EXISTS idx_causal_target ON causal_edges(target_id);
"""

_INFER_PROMPT = """You are a reasoning engine. Decide whether memory A caused,
enabled, or has no causal link to memory B. Return strict JSON.

Schema:
{"relation": "CAUSES|ENABLES|NONE", "confidence": 0.0-1.0, "reasoning": "string"}

Memory A (earlier): "{a_content}"
Memory B (later):   "{b_content}"
"""


def _safe_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(line for line in lines if not line.startswith("```"))
    try:
        loaded = json.loads(text)
        return loaded if isinstance(loaded, dict) else {}
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                loaded = json.loads(text[start : end + 1])
                return loaded if isinstance(loaded, dict) else {}
            except json.JSONDecodeError:
                return {}
        return {}


class CausalGraph(BaseGraph):
    """LLM-inferred cause/effect edges over memory items."""

    def __init__(
        self,
        store: SQLiteStore,
        llm: LLMProvider,
        confidence_threshold: float = 0.5,
        candidate_window: int = 10,
    ) -> None:
        super().__init__(store)
        self.llm = llm
        self.confidence_threshold = confidence_threshold
        self.candidate_window = candidate_window

    async def initialize(self) -> None:
        conn = self.store._require_conn()
        await conn.executescript(_SCHEMA)
        await conn.commit()

    async def add_node(self, memory_id: str, data: dict[str, Any]) -> None:
        content = data.get("content", "")
        if not content:
            return
        candidates = await self._recent_candidates(memory_id)
        for other_id, other_content in candidates:
            verdict = await self._infer(other_content, content)
            relation = str(verdict.get("relation", "NONE")).upper()
            confidence = float(verdict.get("confidence", 0.0))
            if relation == "NONE" or confidence < self.confidence_threshold:
                continue
            reasoning = str(verdict.get("reasoning", ""))
            await self.add_edge(
                Edge(
                    source_id=other_id,
                    target_id=memory_id,
                    graph_type="causal",
                    weight=confidence,
                    metadata={"relation": relation, "reasoning": reasoning},
                )
            )

    async def _infer(self, a_content: str, b_content: str) -> dict[str, Any]:
        prompt = _INFER_PROMPT.replace("{a_content}", a_content).replace(
            "{b_content}", b_content
        )
        response = await self.llm.generate(prompt, temperature=0.0, max_tokens=200)
        return _safe_json(response)

    async def _recent_candidates(self, memory_id: str) -> list[tuple[str, str]]:
        conn = self.store._require_conn()
        cursor = await conn.execute(
            "SELECT id, content FROM memories WHERE id != ? AND status = 'ACTIVE' "
            "ORDER BY created_at DESC LIMIT ?",
            (memory_id, self.candidate_window),
        )
        rows = await cursor.fetchall()
        return [(row["id"], row["content"]) for row in rows]

    async def add_edge(self, edge: Edge) -> None:
        conn = self.store._require_conn()
        meta = edge.metadata or {}
        await conn.execute(
            "INSERT OR REPLACE INTO causal_edges "
            "(source_id, target_id, relation, weight, reasoning, metadata, created_at) "
            "VALUES (?,?,?,?,?,?,?)",
            (
                edge.source_id,
                edge.target_id,
                str(meta.get("relation", "CAUSES")),
                edge.weight,
                str(meta.get("reasoning", "")),
                json.dumps(meta),
                edge.created_at.isoformat(),
            ),
        )
        await conn.commit()

    async def remove_node(self, memory_id: str) -> None:
        conn = self.store._require_conn()
        await conn.execute(
            "DELETE FROM causal_edges WHERE source_id = ? OR target_id = ?",
            (memory_id, memory_id),
        )
        await conn.commit()

    async def get_edges(self, memory_id: str) -> list[Edge]:
        conn = self.store._require_conn()
        cursor = await conn.execute(
            "SELECT source_id, target_id, relation, weight, reasoning, metadata, created_at "
            "FROM causal_edges WHERE source_id = ? OR target_id = ?",
            (memory_id, memory_id),
        )
        rows = await cursor.fetchall()
        out: list[Edge] = []
        for row in rows:
            meta = json.loads(row["metadata"] or "{}")
            meta.setdefault("relation", row["relation"])
            meta.setdefault("reasoning", row["reasoning"])
            out.append(
                Edge(
                    source_id=row["source_id"],
                    target_id=row["target_id"],
                    graph_type="causal",
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
        visited: dict[str, float] = {memory_id: 0.0}
        frontier: list[tuple[str, float]] = [(memory_id, 1.0)]
        for _ in range(depth):
            next_frontier: list[tuple[str, float]] = []
            for node, cum in frontier:
                cursor = await conn.execute(
                    "SELECT target_id, weight FROM causal_edges WHERE source_id = ?",
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
        return sorted(visited.items(), key=lambda kv: kv[1], reverse=True)[:max_results]

    async def get_causal_chain(self, memory_id: str, max_depth: int = 5) -> list[str]:
        """Return the longest cause→effect chain ending at ``memory_id``."""
        conn = self.store._require_conn()
        chain = [memory_id]
        seen = {memory_id}
        current = memory_id
        for _ in range(max_depth):
            cursor = await conn.execute(
                "SELECT source_id FROM causal_edges "
                "WHERE target_id = ? ORDER BY weight DESC LIMIT 1",
                (current,),
            )
            row = await cursor.fetchone()
            if row is None or row["source_id"] in seen:
                break
            current = row["source_id"]
            chain.append(current)
            seen.add(current)
        return list(reversed(chain))
