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

import math
import re
from collections import Counter
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np

from contextdb.core.clock import Clock, utc_now
from contextdb.core.models import MemoryItem, _detached_memory_copy
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


# Tokens that look like identifiers: order numbers, SKUs, account references.
# Ordinary words never enter the lexical ranking, so an exact-identifier hit
# can bypass the relevance floor without making weak overlap a back door.
_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9#][A-Za-z0-9#_./-]*[A-Za-z0-9]")
_MAX_IDENTIFIERS = 3


def identifier_tokens(text: str) -> list[str]:
    """Up to three distinct case-folded tokens of 3+ characters that contain a digit."""
    tokens: list[str] = []
    for match in _IDENTIFIER_RE.finditer(text):
        token = match.group().casefold()
        if len(token) >= 3 and any(c.isdigit() for c in token) and token not in tokens:
            tokens.append(token)
        if len(tokens) == _MAX_IDENTIFIERS:
            break
    return tokens


_WORD_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_./'-]*[A-Za-z0-9]|[A-Za-z0-9]{2,}")
_QUERY_STOPWORDS = frozenset(
    {
        "the", "and", "for", "that", "this", "with", "from", "have", "has", "had",
        "what", "when", "where", "which", "who", "whose", "how", "why", "did", "does",
        "was", "were", "are", "you", "your", "yours", "our", "ours", "their", "theirs",
        "about", "into", "over", "after", "before", "during", "across", "between",
        "through", "can", "could", "would", "should", "will", "many", "much", "some",
        "any", "all", "both", "each", "other", "more", "most", "such", "only", "own",
        "same", "so", "than", "too", "very", "just", "now", "then", "also", "been",
        "being", "there", "here", "they", "them", "his", "her", "its", "not", "but",
        "tell", "give", "list", "show", "please", "based", "according", "mentioned",
    }
)
_MAX_KEYWORD_TOKENS = 8


def keyword_tokens(text: str) -> list[str]:
    """Up to eight salient non-stopword query tokens for BM25 lexical fusion."""
    tokens: list[str] = []
    for match in _WORD_TOKEN_RE.finditer(text):
        token = match.group().strip("._'-/").casefold()
        if len(token) < 2 or token in _QUERY_STOPWORDS:
            continue
        if len(token) == 2 and not any(c.isdigit() for c in token):
            continue
        if token not in tokens:
            tokens.append(token)
    tokens.sort(
        key=lambda t: (
            any(c.isdigit() for c in t),
            any(c in "-_./" for c in t),
            len(t),
        ),
        reverse=True,
    )
    return tokens[:_MAX_KEYWORD_TOKENS]


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
        *,
        as_of: datetime | None = None,
    ) -> list[MemoryItem]:
        return [
            s.item
            for s in await self.search_scored(
                query, query_embedding, top_k=top_k, user_id=user_id, as_of=as_of
            )
        ]

    async def search_scored(
        self,
        query: str,
        query_embedding: list[float],
        top_k: int = 10,
        user_id: str | None = None,
        *,
        as_of: datetime | None = None,
    ) -> list[ScoredMemory]:
        """RRF-fuse per-graph rankings, then multiply by salience."""
        weights = self.classifier.classify(query)
        seed_items = await self.store.search_by_embedding(
            query_embedding,
            top_k=top_k * 2,
            filters=None if as_of is None else {"as_of": as_of},
            user_id=user_id,
            copy_items=False,
        )
        if as_of is not None:
            seed_items = [item for item in seed_items if item.is_valid_at(as_of)]
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

        lexical_items: dict[str, MemoryItem] = {}
        tokens = identifier_tokens(query)
        if tokens:
            lexical_ranking, lexical_items = await self._lexical_ranking(
                tokens, user_id, limit=max(top_k * 2, 10), as_of=as_of
            )
            if lexical_ranking:
                rankings["lexical"] = lexical_ranking
                weights = {**weights, "lexical": max(weights.values(), default=1.0)}

        bm25_items: dict[str, MemoryItem] = {}
        kw_tokens = keyword_tokens(query)
        if kw_tokens and hasattr(self.store, "search_text"):
            bm25_ranking, bm25_items = await self._bm25_ranking(
                kw_tokens,
                seed_items,
                user_id,
                limit=max(top_k * 2, 20),
                as_of=as_of,
            )
            if bm25_ranking:
                rankings["bm25"] = bm25_ranking
                weights = {**weights, "bm25": weights.get("semantic", 1.0) * 0.75}

        fused = self.fuser.fuse(rankings, weights)
        now = as_of or self.clock()
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
            scored = [
                self._score(
                    item, 1.0 / (rank + 1), per_graph_ranks, now, seed_cosine.get(item.id, 0.0)
                )
                for rank, item in enumerate(seed_items[:top_k])
            ]
        else:
            scored = []
            for mid, rrf_score in fused:
                item = seed_by_id.get(mid) or lexical_items.get(mid) or bm25_items.get(mid)
                if item is None:
                    item = await self.store.get_raw(mid)
                if item is None:
                    continue
                if as_of is not None and not item.is_valid_at(as_of):
                    continue
                cos = seed_cosine.get(mid)
                if cos is None:
                    cos = (
                        _cosine(query_embedding, item.embedding)
                        if item.embedding and query_embedding
                        else 0.0
                    )
                scored.append(
                    self._score(
                        item,
                        rrf_score,
                        per_graph_ranks,
                        now,
                        cos,
                    )
                )
            scored.sort(key=lambda s: s.final_score, reverse=True)
        return [
            replace(score, item=_detached_memory_copy(score.item))
            for score in scored[:top_k]
        ]

    async def _lexical_ranking(
        self,
        tokens: list[str],
        user_id: str | None,
        limit: int,
        *,
        as_of: datetime | None = None,
    ) -> tuple[list[tuple[str, float]], dict[str, MemoryItem]]:
        """Memories containing each identifier as a whole token, most matches first."""
        matches: dict[str, int] = {}
        items: dict[str, MemoryItem] = {}
        for token in tokens:
            whole = re.compile(
                rf"(?<![A-Za-z0-9]){re.escape(token)}(?![A-Za-z0-9])", re.IGNORECASE
            )
            for item in await self.store.search_text(token, user_id=user_id, limit=limit):
                if as_of is not None and not item.is_valid_at(as_of):
                    continue
                if whole.search(item.content):
                    matches[item.id] = matches.get(item.id, 0) + 1
                    items[item.id] = item
        ranked = sorted(matches.items(), key=lambda kv: kv[1], reverse=True)
        return [(mid, float(count)) for mid, count in ranked], items

    async def _bm25_ranking(
        self,
        tokens: list[str],
        seed_items: list[MemoryItem],
        user_id: str | None,
        limit: int,
        *,
        as_of: datetime | None = None,
    ) -> tuple[list[tuple[str, float]], dict[str, MemoryItem]]:
        """BM25 keyword ranking over store text matches and semantic seeds."""
        items: dict[str, MemoryItem] = {}
        for item in seed_items:
            if as_of is None or item.is_valid_at(as_of):
                items[item.id] = item
        per_token_limit = max(limit, 25)
        for token in tokens:
            for item in await self.store.search_text(token, user_id=user_id, limit=per_token_limit):
                if as_of is not None and not item.is_valid_at(as_of):
                    continue
                items[item.id] = item
        if not items:
            return [], {}
        docs: dict[str, Counter[str]] = {}
        doc_lens: dict[str, int] = {}
        df: Counter[str] = Counter()
        for mid, item in items.items():
            words = [
                m.group().strip("._'-/").casefold()
                for m in _WORD_TOKEN_RE.finditer(item.content)
            ]
            counts = Counter(words)
            docs[mid] = counts
            doc_lens[mid] = max(1, len(words))
            for token in tokens:
                if counts[token] > 0 or token in item.content.casefold():
                    df[token] += 1
        n_docs = max(len(items) * 3, 25)
        avgdl = sum(doc_lens.values()) / max(1, len(doc_lens))
        scored: list[tuple[float, datetime, str]] = []
        for mid, item in items.items():
            counts = docs[mid]
            dl = doc_lens[mid]
            lowered = item.content.casefold()
            score = 0.0
            for token in tokens:
                tf = counts[token]
                if tf == 0 and token in lowered:
                    tf = 1
                if tf == 0:
                    continue
                idf = math.log(1.0 + (n_docs - df[token] + 0.5) / (df[token] + 0.5))
                score += idf * (tf * 2.2) / (tf + 1.2 * (1.0 - 0.75 + 0.75 * dl / avgdl))
            if score > 0.0:
                basis = item.event_time or item.valid_from or item.created_at
                scored.append((score, basis, mid))
        scored.sort(key=lambda entry: (entry[0], entry[1]), reverse=True)
        top = [(mid, s) for s, _basis, mid in scored[:limit]]
        return top, {mid: items[mid] for mid, _ in top}

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
