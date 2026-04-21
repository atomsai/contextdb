"""Tests for multi-graph retrieval."""

from __future__ import annotations

from contextdb.dynamics.retrieval import QueryClassifier, RetrievalFuser


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
