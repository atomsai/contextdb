"""Tests for multi-graph retrieval."""

from __future__ import annotations

from pathlib import Path

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


def test_identifier_tokens_keep_codes_and_drop_words() -> None:
    from contextdb.dynamics.retrieval import identifier_tokens

    assert identifier_tokens("Refund ORD-48213, SKU AB-100, and order 4419 today") == [
        "ord-48213",
        "ab-100",
        "4419",
    ]
    assert identifier_tokens("which order was refunded for the [EMAIL] customer") == []
    assert identifier_tokens("party of 4 at 7") == []


def test_keyword_tokens_extract_salient_terms_without_stopwords() -> None:
    from contextdb.dynamics.retrieval import keyword_tokens

    tokens = keyword_tokens(
        "How did I integrate Flask-Login for session management before March 29?"
    )
    assert "29" in tokens
    assert "flask-login" in tokens
    assert "how" not in tokens
    assert "for" not in tokens


@pytest.mark.asyncio
async def test_backdated_event_time_and_pre_truncation_as_of(tmp_path: Path) -> None:
    from datetime import datetime, timezone

    import contextdb
    from contextdb.core.config import ContextDBConfig

    db = contextdb.init(
        user_id="u-temporal",
        config=ContextDBConfig(
            storage_url=f"sqlite:///{tmp_path / 'temporal.db'}",
            embedding_model="mock",
            llm_model="mock",
            llm_api_key="mock",
            enable_auto_link=False,
        ),
    )
    async with db:
        t1 = datetime(2024, 3, 15, 12, 0, tzinfo=timezone.utc)
        t2 = datetime(2024, 4, 25, 12, 0, tzinfo=timezone.utc)
        await db.factual.add_many(
            [
                {
                    "content": "First sprint ends on March 29 with 150 commits.",
                    "source": "user_stated",
                    "event_time": t1.isoformat(),
                },
                *[
                    {
                        "content": f"Later April update {i}: 165 commits merged into main.",
                        "source": "user_stated",
                        "event_time": t2.isoformat(),
                    }
                    for i in range(30)
                ],
            ]
        )
        early = await db.factual.recall(
            "How many commits and when does the first sprint end?",
            top_k=3,
            as_of=datetime(2024, 4, 1, 12, 0, tzinfo=timezone.utc),
        )
        assert len(early) == 1
        assert "March 29" in early[0].content
        assert "150 commits" in early[0].content
