"""Evolution engine — auto-linking, consolidation, and pruning.

* :class:`AutoLinker` mirrors a new memory into every configured graph's
  ``add_node`` hook so edges appear as memories are written.
* :class:`Consolidator` finds dense semantic clusters and replaces them with
  a single summary memory, pruning the originals (archive, not delete).
* :class:`Pruner` applies decay / age / redundancy strategies to drop stale
  memories — the operator chooses the strategy per call.

The consolidator is intentionally conservative: it only touches memories
that have multiple neighbors above the semantic threshold, so a single
outlier never triggers a merge.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

from contextdb.core.models import MemoryItem, MemoryStatus, MemoryType

if TYPE_CHECKING:
    from contextdb.graphs.base import BaseGraph
    from contextdb.graphs.semantic import SemanticGraph
    from contextdb.store.sqlite_store import SQLiteStore
    from contextdb.utils.llm import LLMProvider


class AutoLinker:
    """Forward newly-added nodes to each graph's ``add_node`` hook."""

    def __init__(self, graphs: dict[str, BaseGraph]) -> None:
        self.graphs = graphs

    async def link(self, memory_id: str, data: dict[str, Any]) -> None:
        for graph in self.graphs.values():
            try:
                await graph.add_node(memory_id, data)
            except Exception:  # noqa: BLE001
                # One graph failing must not block the others. Auto-linking
                # is best-effort; hard failures surface on explicit calls.
                continue


class Consolidator:
    """Merge dense semantic clusters into a single summary memory."""

    def __init__(
        self,
        store: SQLiteStore,
        semantic_graph: SemanticGraph,
        llm: LLMProvider,
        summary_prompt: str | None = None,
    ) -> None:
        self.store = store
        self.semantic = semantic_graph
        self.llm = llm
        self.summary_prompt = summary_prompt or (
            "Summarize these related memories into one coherent statement. "
            "Preserve every entity and date.\n\n{memories}"
        )

    async def consolidate(self, min_cluster_size: int = 5) -> list[MemoryItem]:
        """Walk active memories in 500-row pages, merging dense clusters."""
        visited: set[str] = set()
        summaries: list[MemoryItem] = []
        async for memory in self.store.iter_memories(batch_size=500):
            if memory.id in visited or memory.status != MemoryStatus.ACTIVE:
                continue
            neighbors = await self.semantic.get_neighbors(
                memory.id, depth=1, max_results=min_cluster_size * 2
            )
            cluster_ids = [memory.id] + [nid for nid, _ in neighbors]
            cluster_ids = [cid for cid in cluster_ids if cid not in visited]
            if len(cluster_ids) < min_cluster_size:
                continue
            cluster_items: list[MemoryItem] = []
            for cid in cluster_ids:
                item = await self.store.get_raw(cid)
                if item is None or item.status != MemoryStatus.ACTIVE:
                    continue
                cluster_items.append(item)
            if len(cluster_items) < min_cluster_size:
                continue

            summary = await self._summarize([m.content for m in cluster_items])
            if not summary:
                continue
            new_item = MemoryItem(
                content=summary,
                embedding=cluster_items[0].embedding,
                memory_type=MemoryType.FACTUAL,
                source="consolidator",
                metadata={"consolidated_from": [m.id for m in cluster_items]},
            )
            stored = await self.store.add(new_item)
            summaries.append(stored)
            for m in cluster_items:
                await self.store.update(m.id, status=MemoryStatus.ARCHIVED)
                visited.add(m.id)
            visited.add(stored.id)
        return summaries

    async def _summarize(self, contents: list[str]) -> str:
        joined = "\n".join(f"- {c}" for c in contents)
        prompt = self.summary_prompt.replace("{memories}", joined)
        response = await self.llm.generate(prompt, temperature=0.0, max_tokens=400)
        return response.strip()


class Pruner:
    """Drop stale memories by strategy.

    Strategies:
    * ``decay`` — access-count-weighted age; below ``threshold`` is archived.
    * ``age`` — hard cutoff by ``older_than`` timedelta.
    * ``redundancy`` — archive memories whose semantic neighbor count exceeds
      ``max_neighbors`` (i.e., already well-represented).
    """

    def __init__(self, store: SQLiteStore) -> None:
        self.store = store

    async def prune(self, strategy: str = "decay", **kwargs: Any) -> int:
        strategy = strategy.lower()
        if strategy == "decay":
            threshold = float(kwargs.get("threshold", 0.1))
            return await self._prune_decay(threshold)
        if strategy == "age":
            older_than = kwargs.get("older_than", timedelta(days=365))
            assert isinstance(older_than, timedelta)
            return await self._prune_age(older_than)
        if strategy == "redundancy":
            semantic = kwargs.get("semantic_graph")
            max_neighbors = int(kwargs.get("max_neighbors", 10))
            if semantic is None:
                return 0
            return await self._prune_redundancy(semantic, max_neighbors)
        raise ValueError(f"Unknown pruning strategy: {strategy}")

    async def _prune_decay(self, threshold: float) -> int:
        now = datetime.now(tz=timezone.utc)
        pruned = 0
        async for memory in self.store.iter_memories(batch_size=500):
            age_days = max(1.0, (now - memory.created_at).total_seconds() / 86400.0)
            score = (memory.access_count + 1) / age_days
            if score < threshold:
                await self.store.update(memory.id, status=MemoryStatus.ARCHIVED)
                pruned += 1
        return pruned

    async def _prune_age(self, older_than: timedelta) -> int:
        now = datetime.now(tz=timezone.utc)
        pruned = 0
        async for memory in self.store.iter_memories(batch_size=500):
            if now - memory.created_at > older_than:
                await self.store.update(memory.id, status=MemoryStatus.ARCHIVED)
                pruned += 1
        return pruned

    async def _prune_redundancy(self, semantic: SemanticGraph, max_neighbors: int) -> int:
        pruned = 0
        async for memory in self.store.iter_memories(batch_size=500):
            neighbors = await semantic.get_neighbors(memory.id, max_results=max_neighbors + 1)
            if len(neighbors) > max_neighbors:
                await self.store.update(memory.id, status=MemoryStatus.ARCHIVED)
                pruned += 1
        return pruned
