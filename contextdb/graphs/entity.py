"""Entity graph — memories and named entities linked bidirectionally.

Entity extraction is delegated to the configured LLM. Responses are expected
to be JSON; we tolerate minor formatting noise and deduplicate on lowercased
entity names.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from contextdb.core.models import Edge, Entity
from contextdb.graphs.base import BaseGraph

if TYPE_CHECKING:
    from contextdb.store.sqlite_store import SQLiteStore
    from contextdb.utils.llm import LLMProvider

_SCHEMA = """
CREATE TABLE IF NOT EXISTS entities (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    attributes TEXT DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_entities_name_lc ON entities(LOWER(name));

CREATE TABLE IF NOT EXISTS memory_entity_edges (
    memory_id TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    relation TEXT DEFAULT 'MENTIONS',
    created_at TEXT NOT NULL,
    PRIMARY KEY (memory_id, entity_id)
);
CREATE INDEX IF NOT EXISTS idx_mee_memory ON memory_entity_edges(memory_id);
CREATE INDEX IF NOT EXISTS idx_mee_entity ON memory_entity_edges(entity_id);
"""

_EXTRACT_PROMPT = """Extract all named entities from the text. Return strict JSON.

Schema:
{"entities": [
  {"name": "string",
   "type": "PERSON|ORG|PRODUCT|LOCATION|EVENT|OTHER",
   "attributes": {}}
]}

Text:
"{text}"
"""


def _safe_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        # strip triple-backtick fences
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


class EntityGraph(BaseGraph):
    """LLM-extracted entity overlay over the memory store."""

    def __init__(self, store: SQLiteStore, llm: LLMProvider) -> None:
        super().__init__(store)
        self.llm = llm

    async def initialize(self) -> None:
        conn = self.store._require_conn()
        await conn.executescript(_SCHEMA)
        await conn.commit()

    async def extract_entities(self, content: str) -> list[Entity]:
        response = await self.llm.generate(_EXTRACT_PROMPT.replace("{text}", content))
        payload = _safe_json(response)
        out: list[Entity] = []
        for raw in payload.get("entities", []) or []:
            name = str(raw.get("name", "")).strip()
            if not name:
                continue
            out.append(
                Entity(
                    name=name,
                    entity_type=str(raw.get("type", "OTHER")),
                    attributes=raw.get("attributes") or {},
                )
            )
        return out

    async def _find_or_create(self, entity: Entity) -> str:
        conn = self.store._require_conn()
        cursor = await conn.execute(
            "SELECT id, attributes FROM entities WHERE LOWER(name) = ?",
            (entity.name.lower(),),
        )
        row = await cursor.fetchone()
        if row is not None:
            if entity.attributes:
                existing = json.loads(row["attributes"] or "{}")
                existing.update(entity.attributes)
                await conn.execute(
                    "UPDATE entities SET attributes = ?, updated_at = ? WHERE id = ?",
                    (
                        json.dumps(existing),
                        datetime.now(tz=timezone.utc).isoformat(),
                        row["id"],
                    ),
                )
                await conn.commit()
            return str(row["id"])
        eid = str(uuid4())
        now = datetime.now(tz=timezone.utc).isoformat()
        await conn.execute(
            "INSERT INTO entities (id, name, entity_type, attributes, created_at, updated_at) "
            "VALUES (?,?,?,?,?,?)",
            (eid, entity.name, entity.entity_type, json.dumps(entity.attributes), now, now),
        )
        await conn.commit()
        return eid

    async def _link_memory(self, memory_id: str, entity_id: str) -> None:
        conn = self.store._require_conn()
        await conn.execute(
            "INSERT OR IGNORE INTO memory_entity_edges "
            "(memory_id, entity_id, relation, created_at) VALUES (?,?,?,?)",
            (memory_id, entity_id, "MENTIONS", datetime.now(tz=timezone.utc).isoformat()),
        )
        await conn.commit()

    async def add_node(self, memory_id: str, data: dict[str, Any]) -> None:
        content = data.get("content", "")
        if not content:
            return
        extracted = await self.extract_entities(content)
        if not extracted:
            return
        for ent in extracted:
            eid = await self._find_or_create(ent)
            await self._link_memory(memory_id, eid)

    async def add_edge(self, edge: Edge) -> None:
        # Entity graph's primary edges are memory↔entity; generic Edge is a no-op.
        return None

    async def remove_node(self, memory_id: str) -> None:
        conn = self.store._require_conn()
        await conn.execute(
            "DELETE FROM memory_entity_edges WHERE memory_id = ?", (memory_id,)
        )
        await conn.commit()

    async def get_edges(self, memory_id: str) -> list[Edge]:
        conn = self.store._require_conn()
        cursor = await conn.execute(
            "SELECT entity_id, created_at FROM memory_entity_edges WHERE memory_id = ?",
            (memory_id,),
        )
        rows = await cursor.fetchall()
        return [
            Edge(
                source_id=memory_id,
                target_id=row["entity_id"],
                graph_type="entity",
                weight=1.0,
                created_at=datetime.fromisoformat(row["created_at"]),
            )
            for row in rows
        ]

    async def get_neighbors(
        self,
        memory_id: str,
        depth: int = 1,
        max_results: int = 20,
    ) -> list[tuple[str, float]]:
        conn = self.store._require_conn()
        cursor = await conn.execute(
            "SELECT entity_id FROM memory_entity_edges WHERE memory_id = ?", (memory_id,)
        )
        entity_ids = [row["entity_id"] for row in await cursor.fetchall()]
        if not entity_ids:
            return []
        placeholders = ",".join(["?"] * len(entity_ids))
        cursor = await conn.execute(
            f"SELECT memory_id, COUNT(*) as shared "
            f"FROM memory_entity_edges WHERE entity_id IN ({placeholders}) AND memory_id != ? "
            "GROUP BY memory_id ORDER BY shared DESC LIMIT ?",
            [*entity_ids, memory_id, max_results],
        )
        rows = await cursor.fetchall()
        return [(row["memory_id"], float(row["shared"])) for row in rows]

    async def get_entity_profile(self, name: str) -> dict[str, Any]:
        conn = self.store._require_conn()
        cursor = await conn.execute(
            "SELECT id, name, entity_type, attributes FROM entities WHERE LOWER(name) = ?",
            (name.lower(),),
        )
        row = await cursor.fetchone()
        if row is None:
            return {"name": name, "memories": [], "attributes": {}}
        eid = row["id"]
        mem_cursor = await conn.execute(
            "SELECT memory_id FROM memory_entity_edges WHERE entity_id = ?", (eid,)
        )
        memory_ids = [r["memory_id"] for r in await mem_cursor.fetchall()]
        return {
            "id": eid,
            "name": row["name"],
            "entity_type": row["entity_type"],
            "attributes": json.loads(row["attributes"] or "{}"),
            "memories": memory_ids,
        }

    async def get_entity_memories(self, name: str) -> list[str]:
        profile = await self.get_entity_profile(name)
        return list(profile.get("memories", []))

    async def list_entities(self) -> list[Entity]:
        conn = self.store._require_conn()
        cursor = await conn.execute(
            "SELECT id, name, entity_type, attributes FROM entities ORDER BY name"
        )
        rows = await cursor.fetchall()
        return [
            Entity(
                name=row["name"],
                entity_type=row["entity_type"],
                attributes=json.loads(row["attributes"] or "{}"),
                memory_ids=await self.get_entity_memories(row["name"]),
            )
            for row in rows
        ]
