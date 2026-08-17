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
from contextdb.dynamics.trust import infer_action_relevant

if TYPE_CHECKING:
    from contextdb.privacy.pii_detector import PIIDetector
    from contextdb.utils.embeddings import EmbeddingProvider
    from contextdb.utils.llm import LLMProvider


_EXTRACT_PROMPT = """Extract atomic facts and named entities from the text.
Return strict JSON.

Schema:
{"facts": [{"content": "string", "type": "FACTUAL|EXPERIENTIAL",
  "entities": ["string"],
  "source": "user_stated|agent_inferred|third_party",
  "confidence": 0.0-1.0,
  "action_relevant": true|false,
  "entity": "string|null", "attribute": "string|null"}]}

Rules:
- Each fact must be self-contained (understandable without context).
- Skip small talk; keep substantive information only.
- Aim for 1-5 facts per turn.
- source: "user_stated" when the speaker asserts it about themselves;
  "third_party" for hearsay ("my colleague said...") or content of
  uncertain origin; "agent_inferred" for conclusions you draw rather than
  statements made.
- Wishes, hypotheticals, and plans are NOT facts: "I'd like to come in
  Thursday" is a desire, not a booking. Extract them with confidence <= 0.5
  so they cannot be acted on without confirmation.
- action_relevant=true for anything that gates an action: bookings, prices,
  schedules, contact details, identity, health/finance/legal attributes.
- entity/attribute name the stable slot the fact fills (e.g. entity
  "meeting", attribute "time") so repeats and corrections match. Use
  snake_case; null when no clear slot exists.

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


_EPISTEMIC_SOURCES = {"user_stated", "agent_inferred", "third_party"}


class MemoryExtractor:
    """LLM-driven fact + entity extraction per turn, with epistemic typing."""

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

            source = str(raw.get("source", "user_stated")).lower()
            if source not in _EPISTEMIC_SOURCES:
                source = "user_stated"
            try:
                confidence = float(raw.get("confidence", 0.8))
            except (TypeError, ValueError):
                confidence = 0.8
            confidence = min(1.0, max(0.0, confidence))
            action_relevant_raw = raw.get("action_relevant")
            action_relevant = (
                bool(action_relevant_raw)
                if action_relevant_raw is not None
                else infer_action_relevant(content)
            )
            entity = raw.get("entity")
            attribute = raw.get("attribute")
            out.append(
                {
                    "content": content,
                    "memory_type": mem_type,
                    "entities": entities,
                    "epistemic_source": source,
                    "confidence": confidence,
                    "action_relevant": action_relevant,
                    "entity_key": str(entity).strip() if entity else None,
                    "attribute_key": str(attribute).strip() if attribute else None,
                }
            )
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
        # PII-before-embedder: redact first, embed only the processed text.
        processed_facts: list[tuple[dict[str, Any], str, list[Any]]] = []
        for fact in all_facts:
            processed, pii_annotations = self.pii.process(fact["content"])
            processed_facts.append((fact, processed, pii_annotations))
        contents = [processed for _, processed, _ in processed_facts]
        embeddings = await self.embedder.embed(contents) if contents else []
        for (fact, processed, pii_annotations), embedding in zip(
            processed_facts, embeddings, strict=False
        ):
            items.append(
                MemoryItem(
                    content=processed,
                    embedding=embedding,
                    memory_type=MemoryType(fact["memory_type"]),
                    source=source,
                    pii_annotations=pii_annotations,
                    entity_mentions=list(fact.get("entities", [])),
                    epistemic_source=fact.get("epistemic_source", "user_stated"),
                    confidence=float(fact.get("confidence", 0.8)),
                    action_relevant=bool(
                        fact.get("action_relevant", infer_action_relevant(processed))
                    ),
                    entity_key=fact.get("entity_key"),
                    attribute_key=fact.get("attribute_key"),
                )
            )
        return items
