"""Memory formation pipeline.

Turns a raw conversation or document into a list of ready-to-store
:class:`~contextdb.core.models.MemoryItem` objects. The steps are:

1. :class:`Segmenter` — split into coherent conversational turns.
2. :class:`MemoryExtractor` — LLM pulls atomic facts + entities from a turn.
3. :class:`MemoryCompressor` — LLM compresses a cluster into a single summary.
4. PII detection + embedding generation happen on the output items.

Every step is optional at the call site — the pipeline short-circuits to
the raw text if the LLM returns nothing usable.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

from contextdb.core.models import MemoryItem, MemoryType

if TYPE_CHECKING:
    from contextdb.privacy.pii_detector import PIIDetector
    from contextdb.utils.embeddings import EmbeddingProvider
    from contextdb.utils.llm import LLMProvider


_EXTRACT_PROMPT = """Extract atomic facts and named entities from the text.
Return strict JSON.

Schema:
{"facts": [{"content": "string", "type": "FACTUAL|EXPERIENTIAL", "entities": ["string"]}]}

Rules:
- Each fact must be self-contained (understandable without context).
- Skip small talk; keep substantive information only.
- Aim for 1-5 facts per turn.

Text: "{text}"
"""

_COMPRESS_PROMPT = """Summarize the following related memories into one concise
statement. Preserve all named entities and dates. Return plain text, no JSON.

Memories:
{memories}
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


class Segmenter:
    """Split raw text into turns / coherent chunks.

    The baseline rule: newlines separate turns, and speaker prefixes
    (``User:``, ``Agent:``) are preserved. Anything shorter than ``min_chars``
    is merged into the next chunk so we don't emit fragments.
    """

    def __init__(self, min_chars: int = 20) -> None:
        self.min_chars = min_chars

    def segment(self, text: str) -> list[str]:
        raw = [
            chunk.strip()
            for chunk in re.split(r"\n{2,}|(?<=[.!?])\s{2,}", text)
            if chunk.strip()
        ]
        merged: list[str] = []
        buffer = ""
        for chunk in raw:
            candidate = f"{buffer} {chunk}".strip() if buffer else chunk
            if len(candidate) < self.min_chars:
                buffer = candidate
                continue
            merged.append(candidate)
            buffer = ""
        if buffer:
            if merged:
                merged[-1] = f"{merged[-1]} {buffer}".strip()
            else:
                merged.append(buffer)
        return merged


class MemoryExtractor:
    """LLM-driven fact + entity extraction per turn."""

    def __init__(self, llm: LLMProvider) -> None:
        self.llm = llm

    async def extract(self, turn: str) -> list[dict[str, Any]]:
        response = await self.llm.generate(_EXTRACT_PROMPT.replace("{text}", turn))
        payload = _safe_json(response)
        out: list[dict[str, Any]] = []
        for raw in payload.get("facts", []) or []:
            content = str(raw.get("content", "")).strip()
            if not content:
                continue
            mem_type = str(raw.get("type", "FACTUAL")).upper()
            if mem_type not in {"FACTUAL", "EXPERIENTIAL", "WORKING"}:
                mem_type = "FACTUAL"
            entities = [str(e).strip() for e in raw.get("entities", []) or [] if e]
            out.append({"content": content, "memory_type": mem_type, "entities": entities})
        return out


class MemoryCompressor:
    """LLM-driven cluster summarization.

    Given a list of memory contents, produce a single condensed statement
    that preserves entities and temporal markers. If the LLM returns an empty
    string, we fall back to naïve concatenation so the caller never loses
    data.
    """

    def __init__(self, llm: LLMProvider) -> None:
        self.llm = llm

    async def compress(self, memories: list[str]) -> str:
        if not memories:
            return ""
        if len(memories) == 1:
            return memories[0]
        joined = "\n".join(f"- {m}" for m in memories)
        response = await self.llm.generate(_COMPRESS_PROMPT.replace("{memories}", joined))
        summary = response.strip()
        return summary or " | ".join(memories)


class FormationPipeline:
    """Glue the formation steps into a single async entry point."""

    def __init__(
        self,
        segmenter: Segmenter,
        extractor: MemoryExtractor,
        compressor: MemoryCompressor,
        pii: PIIDetector,
        embedder: EmbeddingProvider,
    ) -> None:
        self.segmenter = segmenter
        self.extractor = extractor
        self.compressor = compressor
        self.pii = pii
        self.embedder = embedder

    async def process(self, text: str, source: str = "") -> list[MemoryItem]:
        turns = self.segmenter.segment(text)
        all_facts: list[dict[str, Any]] = []
        for turn in turns:
            facts = await self.extractor.extract(turn)
            if not facts:
                # Fallback: store the turn verbatim as a FACTUAL memory.
                facts = [{"content": turn, "memory_type": "FACTUAL", "entities": []}]
            all_facts.extend(facts)

        items: list[MemoryItem] = []
        contents = [fact["content"] for fact in all_facts]
        embeddings = await self.embedder.embed(contents) if contents else []
        for fact, embedding in zip(all_facts, embeddings, strict=False):
            content = fact["content"]
            processed, pii_annotations = self.pii.process(content)
            items.append(
                MemoryItem(
                    content=processed,
                    embedding=embedding,
                    memory_type=MemoryType(fact["memory_type"]),
                    source=source,
                    pii_annotations=pii_annotations,
                    entity_mentions=list(fact.get("entities", [])),
                )
            )
        return items
