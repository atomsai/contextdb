"""Tests for the memory formation pipeline."""

from __future__ import annotations

import pytest

from contextdb.dynamics.formation import (
    FormationPipeline,
    MemoryCompressor,
    MemoryExtractor,
    Segmenter,
)
from contextdb.privacy.pii_detector import PIIDetector
from contextdb.utils.embeddings import MockEmbedding
from contextdb.utils.llm import MockLLM


def test_segmenter_splits_paragraphs() -> None:
    seg = Segmenter(min_chars=5)
    segments = seg.segment("First paragraph.\n\nSecond paragraph here.")
    assert len(segments) == 2


def test_segmenter_merges_short_chunks() -> None:
    seg = Segmenter(min_chars=20)
    segments = seg.segment("Hi.\n\nThis is a slightly longer paragraph to keep.")
    assert all(len(s) >= 1 for s in segments)


@pytest.mark.asyncio
async def test_extractor_returns_facts() -> None:
    llm = MockLLM(
        responses={
            "Extract": (
                '{"facts": [{"content": "User likes coffee", "type": "FACTUAL", '
                '"entities": ["coffee"]}]}'
            )
        }
    )
    extractor = MemoryExtractor(llm)
    facts = await extractor.extract("I love coffee in the morning.")
    assert facts and facts[0]["entities"] == ["coffee"]


@pytest.mark.asyncio
async def test_compressor_single_item_passthrough() -> None:
    llm = MockLLM()
    comp = MemoryCompressor(llm)
    out = await comp.compress(["only one"])
    assert out == "only one"


@pytest.mark.asyncio
async def test_pipeline_end_to_end() -> None:
    llm = MockLLM(
        responses={
            "Extract": (
                '{"facts": [{"content": "A fact", "type": "FACTUAL", "entities": []}]}'
            )
        }
    )
    pipeline = FormationPipeline(
        Segmenter(min_chars=1),
        MemoryExtractor(llm),
        MemoryCompressor(llm),
        PIIDetector(),
        MockEmbedding(dimension=16),
    )
    items = await pipeline.process("Hello world.\n\nSecond thing.")
    assert items
    assert all(m.embedding is not None for m in items)
