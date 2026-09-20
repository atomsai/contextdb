"""Tests for multi-graph retrieval."""

from __future__ import annotations

import pytest

from contextdb.core.models import MemoryItem
from contextdb.dynamics.retrieval import (
    QueryClassifier,
    RetrievalEngine,
    RetrievalFuser,
)


def test_query_classifier_weights() -> None:
    qc = QueryClassifier()
    weights_temporal = qc.classify("when did we last talk?")
    assert "temporal" in weights_temporal
    weights_causal = qc.classify("why did the build fail?")
    assert "causal" in weights_causal
    weights_generic = qc.classify("tell me about python")
    assert weights_generic["semantic"] == 1.0


def test_rrf_fusion() -> None:
    fuser = RetrievalFuser(k=60)
    rankings = {
        "semantic": [("a", 0.9), ("b", 0.7), ("c", 0.5)],
        "temporal": [("b", 1.0), ("c", 0.8)],
    }
    weights = {"semantic": 0.5, "temporal": 0.5}
    fused = fuser.fuse(rankings, weights)
    ranked_ids = [mid for mid, _ in fused]
    # "b" appears higher than "a" on aggregate (rank 2 semantic + rank 1 temporal
    # vs. rank 1 semantic only).
    assert ranked_ids[0] == "b"
    assert set(ranked_ids) == {"a", "b", "c"}


@pytest.mark.asyncio
async def test_retrieval_ranks_store_references_but_returns_detached_items() -> None:
    source = MemoryItem(
        content="Thursday is confirmed",
        embedding=[1.0, 0.0],
        metadata={"nested": ["source"]},
    )

    class ReferenceStore:
        async def search_by_embedding(
            self,
            _embedding: list[float],
            top_k: int = 10,
            filters: dict[str, object] | None = None,
            user_id: str | None = None,
            *,
            copy_items: bool = True,
        ) -> list[MemoryItem]:
            assert top_k == 2
            assert filters is None
            assert user_id == "user-1"
            assert copy_items is False
            return [source]

        async def get_raw(self, _memory_id: str) -> MemoryItem | None:
            return None

    engine = RetrievalEngine(
        ReferenceStore(),  # type: ignore[arg-type]
        {},
        QueryClassifier(),
        RetrievalFuser(),
    )
    results = await engine.search_scored(
        "visit day",
        [1.0, 0.0],
        top_k=1,
        user_id="user-1",
    )

    assert len(results) == 1
    assert results[0].item == source
    assert results[0].item is not source
    results[0].item.metadata["nested"].append("caller")
    assert source.metadata == {"nested": ["source"]}
