"""Multi-graph retrieval — query classification + reciprocal-rank fusion.

The engine runs the query against each configured graph (plus a raw vector
search against the store) and fuses results using Reciprocal Rank Fusion
(Cormack et al., 2009) with the classic ``k=60`` smoothing. RRF is
parameter-light, ignores raw score scales, and tends to beat linear
combination for heterogeneous retrievers — ideal when one retriever scores in
cosine space and another in edge-weight space.

The :class:`QueryClassifier` is intentionally rule-based. An LLM classifier
would be more accurate but would add latency to every query; the regex
heuristics below get ~80% of the signal for zero cost.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from contextdb.core.models import MemoryItem

if TYPE_CHECKING:
    from contextdb.graphs.base import BaseGraph
    from contextdb.store.sqlite_store import SQLiteStore


_TEMPORAL_MARKERS = re.compile(
    r"\b(when|before|after|yesterday|today|tomorrow|last|next|since|until|during|ago)\b",
    re.IGNORECASE,
)
_CAUSAL_MARKERS = re.compile(
    r"\b(why|because|caused|leads? to|due to|resulted? in|reason|so that)\b",
    re.IGNORECASE,
)
_ENTITY_MARKERS = re.compile(
    r"\b(who|whose|which person|what company|what product)\b",
    re.IGNORECASE,
)


class QueryClassifier:
    """Classify a natural-language query into graph-weighting hints.

    Returns a ``dict[graph_name, weight]`` that sums roughly to 1.0. Callers
    use these as mixing weights when fusing per-graph rankings.
    """

    def classify(self, query: str) -> dict[str, float]:
        weights: dict[str, float] = {"semantic": 1.0}
        if _TEMPORAL_MARKERS.search(query):
            weights["temporal"] = 1.2
        if _CAUSAL_MARKERS.search(query):
            weights["causal"] = 1.4
        if _ENTITY_MARKERS.search(query):
            weights["entity"] = 1.1
        total = sum(weights.values())
        return {k: v / total for k, v in weights.items()}


class RetrievalFuser:
    """Reciprocal Rank Fusion over per-graph candidate lists."""

    def __init__(self, k: int = 60) -> None:
        self.k = k

    def fuse(
        self,
        rankings: dict[str, list[tuple[str, float]]],
        weights: dict[str, float],
    ) -> list[tuple[str, float]]:
        scores: dict[str, float] = {}
        for graph_name, ranking in rankings.items():
            w = weights.get(graph_name, 0.0)
            if w == 0.0 or not ranking:
                continue
            for rank, (memory_id, _) in enumerate(ranking, start=1):
                scores[memory_id] = scores.get(memory_id, 0.0) + w * (1.0 / (self.k + rank))
        return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


class RetrievalEngine:
    """Coordinate vector + graph retrieval and return ranked memories."""

    def __init__(
        self,
        store: SQLiteStore,
        graphs: dict[str, BaseGraph],
        classifier: QueryClassifier,
        fuser: RetrievalFuser,
    ) -> None:
        self.store = store
        self.graphs = graphs
        self.classifier = classifier
        self.fuser = fuser

    async def search(
        self,
        query: str,
        query_embedding: list[float],
        top_k: int = 10,
    ) -> list[MemoryItem]:
        weights = self.classifier.classify(query)
        seed_items = await self.store.search_by_embedding(query_embedding, top_k=top_k * 2)
        semantic_ranking = [(item.id, 1.0 / (rank + 1)) for rank, item in enumerate(seed_items)]
        rankings: dict[str, list[tuple[str, float]]] = {"semantic": semantic_ranking}

        seed_ids = [item.id for item in seed_items[: max(1, top_k)]]
        for name, graph in self.graphs.items():
            if name == "semantic":
                continue
            if weights.get(name, 0.0) <= 0.0:
                continue
            expanded: dict[str, float] = {}
            for sid in seed_ids:
                neighbors = await graph.get_neighbors(sid, max_results=top_k)
                for nid, weight in neighbors:
                    expanded[nid] = max(expanded.get(nid, 0.0), weight)
            rankings[name] = sorted(expanded.items(), key=lambda kv: kv[1], reverse=True)

        fused = self.fuser.fuse(rankings, weights)
        ordered_ids = [mid for mid, _ in fused[:top_k]]
        if not ordered_ids:
            return seed_items[:top_k]

        items: list[MemoryItem] = []
        for mid in ordered_ids:
            item = await self.store.get_raw(mid)
            if item is not None:
                items.append(item)
        return items
