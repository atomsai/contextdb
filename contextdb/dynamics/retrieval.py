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
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np

from contextdb.core.clock import Clock, utc_now
from contextdb.core.models import MemoryItem
from contextdb.dynamics.salience import (
    DEFAULT_HALF_LIFE_DAYS,
    age_in_days,
    criticality_boost,
    recurrence,
    salience,
)

if TYPE_CHECKING:
    from contextdb.graphs.base import BaseGraph
    from contextdb.store.base import BaseStore


_TEMPORAL_MARKERS = re.compile(
    r"\b(when|before|after|yesterday|today|tomorrow|last|next|since|until|during|ago)\b",
    re.IGNORECASE,
)
_CAUSAL_MARKERS = re.compile(
    r"\b(why|because|caused|leads? to|due to|resulted? in|reason|so that)\b",
    re.IGNORECASE,
)
def _cosine(a: list[float], b: list[float]) -> float:
    va = np.asarray(a, dtype=np.float32)
    vb = np.asarray(b, dtype=np.float32)
    na = float(np.linalg.norm(va))
    nb = float(np.linalg.norm(vb))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


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


@dataclass
class ScoredMemory:
    """A retrieved memory with the full score decomposition (Epic 4/6).

    ``final_score`` = ``rrf_score`` x ``salience`` when salience is enabled.
    The components are what recall-side observability logs and
    ``db.explain`` surfaces — "why did the agent say that?" must be
    answerable from these numbers alone.
    """

    item: MemoryItem
    final_score: float
    rrf_score: float
    salience: float
    age_days: float
    recurrence: float
    criticality_boost: float
    cosine: float = 0.0
    per_graph: dict[str, int] = field(default_factory=dict)


class RetrievalEngine:
    """Coordinate vector + graph retrieval and return ranked memories."""

    def __init__(
        self,
        store: BaseStore,
        graphs: dict[str, BaseGraph],
        classifier: QueryClassifier,
        fuser: RetrievalFuser,
        enable_salience: bool = True,
        salience_half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
        clock: Clock = utc_now,
    ) -> None:
        self.store = store
        self.graphs = graphs
        self.classifier = classifier
        self.fuser = fuser
        self.enable_salience = enable_salience
        self.salience_half_life_days = salience_half_life_days
        self.clock = clock

    async def search(
        self,
        query: str,
        query_embedding: list[float],
        top_k: int = 10,
        user_id: str | None = None,
    ) -> list[MemoryItem]:
        return [
            s.item
            for s in await self.search_scored(
                query, query_embedding, top_k=top_k, user_id=user_id
            )
        ]

    async def search_scored(
        self,
        query: str,
        query_embedding: list[float],
        top_k: int = 10,
        user_id: str | None = None,
    ) -> list[ScoredMemory]:
        """RRF-fuse per-graph rankings, then multiply by salience."""
        weights = self.classifier.classify(query)
        seed_items = await self.store.search_by_embedding(
            query_embedding, top_k=top_k * 2, user_id=user_id
        )
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
        now = self.clock()
        seed_cosine = {
            item.id: _cosine(query_embedding, item.embedding) if item.embedding else 0.0
            for item in seed_items
        }
        seed_by_id = {item.id: item for item in seed_items}

        # Per-graph rank lookup for observability.
        per_graph_ranks: dict[str, dict[str, int]] = {
            name: {mid: rank for rank, (mid, _) in enumerate(ranking, start=1)}
            for name, ranking in rankings.items()
        }

        if not fused:
            # No fusion signal (empty store or zero weights): rank the seeds
            # directly, still applying salience.
            return [
                self._score(
                    item, 1.0 / (rank + 1), per_graph_ranks, now, seed_cosine.get(item.id, 0.0)
                )
                for rank, item in enumerate(seed_items[:top_k])
            ]

        scored: list[ScoredMemory] = []
        for mid, rrf_score in fused:
            item = seed_by_id.get(mid)
            if item is None:
                item = await self.store.get_raw(mid)
            if item is None:
                continue
            scored.append(
                self._score(item, rrf_score, per_graph_ranks, now, seed_cosine.get(mid, 0.0))
            )
        scored.sort(key=lambda s: s.final_score, reverse=True)
        return scored[:top_k]

    def _score(
        self,
        item: MemoryItem,
        rrf_score: float,
        per_graph_ranks: dict[str, dict[str, int]],
        now: datetime,
        cosine: float = 0.0,
    ) -> ScoredMemory:
        boost = criticality_boost(item)
        rec = recurrence(item)
        age = age_in_days(item, now)
        sal = (
            salience(item, now, self.salience_half_life_days)
            if self.enable_salience
            else 1.0
        )
        return ScoredMemory(
            item=item,
            final_score=rrf_score * sal,
            rrf_score=rrf_score,
            salience=sal,
            age_days=age,
            recurrence=rec,
            criticality_boost=boost,
            cosine=cosine,
            per_graph={
                name: ranks[item.id]
                for name, ranks in per_graph_ranks.items()
                if item.id in ranks
            },
        )
