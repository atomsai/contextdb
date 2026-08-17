"""Acceptance evals for the ContextDB Trust Model PRD (EVAL-0.1 … EVAL-8.1).

These evals are the spec for the trust-model epics. Where the PRD prose and
these evals disagree, the evals win.

Conventions:
* Every eval uses the mock embedder (deterministic, network-free) and either
  the mock LLM or a local OpenAI-compatible stub server.
* No eval ever loosens the PII-before-embedder rule or the hash-chained
  audit log to pass.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
import warnings
from collections.abc import AsyncIterator, Iterator
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio

import contextdb
from contextdb import ContextDB, ContextDBConfig
from contextdb.core.exceptions import ConfigError
from contextdb.utils.llm import MockLLM

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def make_config(tmp_path: Path, name: str = "eval.db", **overrides: Any) -> ContextDBConfig:
    """Build a fully-mocked config rooted at ``tmp_path``."""
    base: dict[str, Any] = {
        "storage_url": f"sqlite:///{tmp_path}/{name}",
        "embedding_model": "mock",
        "embedding_dim": 32,
        "llm_model": "mock",
        "llm_api_key": "mock",
        "enable_entity_graph": False,
        "enable_multi_graph": False,
        "enable_rl_manager": False,
        "enable_audit": True,
        "enable_auto_link": True,
        "pii_action": "redact",
    }
    base.update(overrides)
    return ContextDBConfig(**base)


@pytest_asyncio.fixture
async def db(tmp_path: Path) -> AsyncIterator[ContextDB]:
    client = contextdb.init(user_id="eval-user", config=make_config(tmp_path))
    try:
        yield client
    finally:
        await client.close()


# ---------------------------------------------------------------------------
# Epic 0 — Provider-agnostic LLM and embedder
# ---------------------------------------------------------------------------

_EXTRACTION_PAYLOAD = json.dumps(
    {
        "facts": [
            {
                "content": "The meeting with Acme is scheduled for 3pm on Thursday.",
                "type": "FACTUAL",
                "entities": ["Acme", "meeting"],
                "source": "user_stated",
                "confidence": 0.9,
                "action_relevant": True,
                "entity": "meeting",
                "attribute": "time",
            }
        ],
        "entities": ["Acme", "meeting"],
    }
)


class _OpenAICompatibleHandler(BaseHTTPRequestHandler):
    """Minimal OpenAI-compatible stub: chat completions + embeddings."""

    def do_POST(self) -> None:  # noqa: N802 (stdlib handler name)
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            request = json.loads(raw)
        except json.JSONDecodeError:
            request = {}

        if self.path.endswith("/chat/completions"):
            body: dict[str, Any] = {
                "id": "chatcmpl-local",
                "object": "chat.completion",
                "created": 0,
                "model": request.get("model", "local"),
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": _EXTRACTION_PAYLOAD},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        elif self.path.endswith("/embeddings"):
            inputs = request.get("input", [])
            if isinstance(inputs, str):
                inputs = [inputs]
            body = {
                "object": "list",
                "data": [
                    {
                        "object": "embedding",
                        "index": i,
                        "embedding": [0.01 * (i + 1)] * 32,
                    }
                    for i, _ in enumerate(inputs)
                ],
                "model": request.get("model", "local"),
                "usage": {"prompt_tokens": 1, "total_tokens": 1},
            }
        else:
            self.send_response(404)
            self.end_headers()
            return

        payload = json.dumps(body).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *_args: Any) -> None:
        return


@pytest.fixture
def openai_compatible_base_url() -> Iterator[str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _OpenAICompatibleHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/v1"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


async def test_eval_0_1_non_openai_model_via_base_url_extracts_end_to_end(
    tmp_path: Path, openai_compatible_base_url: str
) -> None:
    """EVAL-0.1: llm_base_url + a non-OpenAI model name performs extraction."""
    config = make_config(
        tmp_path,
        llm_model="llama-3.3-70b-versatile",  # not gpt-*/o1-* — rejected without base_url
        llm_api_key=None,
        llm_base_url=openai_compatible_base_url,
    )
    db = contextdb.init(user_id="eval-user", config=config)
    try:
        items = await db.add_conversation("The meeting with Acme is at 3pm on Thursday.")
        assert items, "extraction produced no memories"
        contents = [m.content for m in items]
        # The stored content comes from the stub server's extraction payload,
        # proving the non-OpenAI model was invoked through the custom base_url.
        assert any("Acme" in c and "3pm" in c for c in contents)
        assert any("meeting" in (m.entity_mentions or []) for m in items)
    finally:
        await db.close()


async def test_eval_0_1b_embedding_base_url_routes_any_model(
    tmp_path: Path, openai_compatible_base_url: str
) -> None:
    """EVAL-0.1 (embedder half): embedding_base_url accepts any model name."""
    from contextdb.utils.embeddings import OpenAIEmbedding, get_embedding_provider

    provider = get_embedding_provider(
        "bge-large-en-v1.5",  # unknown to the OpenAI dim table
        api_key=None,
        base_url=openai_compatible_base_url,
        dimension=32,
    )
    assert isinstance(provider, OpenAIEmbedding)
    vectors = await provider.embed(["hello", "world"])
    assert len(vectors) == 2
    assert all(len(v) == 32 for v in vectors)


async def test_eval_0_2_init_warns_loudly_without_key(tmp_path: Path, monkeypatch: Any) -> None:
    """EVAL-0.2: no key + default endpoint => loud warning at init, ConfigError
    at first extraction use — never a silent degrade to raw-text storage."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.warns(UserWarning, match="(?i)api key"):
        db = contextdb.init(
            user_id="eval-user",
            config=make_config(
                tmp_path,
                name="warn.db",
                llm_model="gpt-4o-mini",
                llm_api_key=None,
                llm_base_url=None,
            ),
        )
    try:
        with pytest.raises(ConfigError, match="(?i)api key"):
            await db.add_conversation("hello there, this turn needs extraction")
    finally:
        await db.close()


def test_eval_0_2b_base_url_without_key_does_not_warn(tmp_path: Path, monkeypatch: Any) -> None:
    """Keyless local servers (Ollama/vLLM) must not trigger the warning."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        db = contextdb.init(
            user_id="eval-user",
            config=make_config(
                tmp_path,
                name="nowarn.db",
                llm_model="llama-3.3-70b-versatile",
                llm_api_key=None,
                llm_base_url="http://127.0.0.1:9/v1",  # unreachable is fine; lazy
            ),
        )
    assert isinstance(db, ContextDB)


# ---------------------------------------------------------------------------
# Epic 1 — Epistemic typing + verify-before-act
# ---------------------------------------------------------------------------


async def test_eval_1_1_wish_requires_confirmation_and_is_not_actionable(
    db: ContextDB,
) -> None:
    """EVAL-1.1: a stored wish is recalled with action_relevant=True,
    requires_confirmation=True, and never appears in recall_for_action()
    uncorroborated."""
    wish = await db.factual.add(
        "I'd like to come in Thursday",
        source="user_stated",
        confidence=0.5,  # a stated desire, not an asserted fact
        action_relevant=True,
        entity="caller",
        attribute="preferred_visit_day",
    )
    assert wish.action_relevant is True
    assert wish.requires_confirmation is True
    assert wish.action_trusted is False

    recalled = await db.factual.recall("come in Thursday")
    hit = next(m for m in recalled if m.id == wish.id)
    assert hit.action_relevant is True
    assert hit.requires_confirmation is True

    trusted = await db.factual.recall_for_action("come in Thursday")
    assert all(m.id != wish.id for m in trusted)

    # Corroboration is what graduates a wish into an actionable fact.
    again = await db.factual.add(
        "I'd like to come in Thursday",
        source="user_stated",
        confidence=0.5,
        action_relevant=True,
        entity="caller",
        attribute="preferred_visit_day",
    )
    assert again.id == wish.id, "same slot value must not duplicate"
    assert again.corroboration_count == 2
    trusted_after = await db.factual.recall_for_action("come in Thursday")
    assert any(m.id == wish.id for m in trusted_after)


async def test_eval_1_2_hearsay_needs_corroboration(db: ContextDB) -> None:
    """EVAL-1.2: third_party claims stay out of recall_for_action until
    corroboration_count reaches 2 (dedupe by entity+attribute, not string)."""
    first = await db.factual.add(
        "A colleague said the office is moving to Denver",
        source="third_party",
        confidence=0.6,
        action_relevant=True,
        entity="office",
        attribute="location",
    )
    assert first.requires_confirmation is True
    assert all(
        m.id != first.id for m in await db.factual.recall_for_action("office move Denver")
    )

    # Same slot, same value, different speaker: corroborates the occupant
    # instead of duplicating or superseding it.
    second = await db.factual.add(
        "A colleague said the office is moving to Denver",
        source="third_party",
        confidence=0.6,
        action_relevant=True,
        entity="office",
        attribute="location",
    )
    assert second.id == first.id, "same slot value must corroborate, not duplicate"
    assert second.corroboration_count == 2
    assert second.requires_confirmation is False
    trusted = await db.factual.recall_for_action("office move Denver")
    assert any(m.id == first.id for m in trusted)


async def test_eval_1_3_user_stated_confident_fact_is_immediately_actionable(
    db: ContextDB,
) -> None:
    """EVAL-1.3: source=user_stated at high confidence passes the action bar."""
    fact = await db.factual.add(
        "My account number ends in 1234",
        source="user_stated",
        confidence=0.95,
        action_relevant=True,
        entity="account",
        attribute="number",
    )
    assert fact.requires_confirmation is False
    trusted = await db.factual.recall_for_action("account number")
    assert any(m.id == fact.id for m in trusted)


async def test_eval_1_4_extraction_sets_epistemic_fields(tmp_path: Path) -> None:
    """EVAL-1.4: the extraction prompt types source/confidence/action_relevant
    per fact, and the pipeline persists them."""
    db = contextdb.init(user_id="eval-user", config=make_config(tmp_path))
    scripted = MockLLM(
        responses={
            "I'd like to come in Thursday": json.dumps(
                {
                    "facts": [
                        {
                            "content": "The caller wishes to visit on Thursday",
                            "type": "FACTUAL",
                            "entities": ["caller"],
                            "source": "user_stated",
                            "confidence": 0.5,
                            "action_relevant": True,
                            "entity": "caller",
                            "attribute": "preferred_visit_day",
                        }
                    ],
                    "entities": ["caller"],
                }
            )
        }
    )
    db._llm = scripted  # injected before first await; _ensure_init keeps it
    try:
        items = await db.add_conversation("I'd like to come in Thursday")
        assert items, "extraction produced nothing"
        fact = items[0]
        assert fact.epistemic_source == "user_stated"
        assert fact.confidence == 0.5
        assert fact.action_relevant is True
        assert fact.entity_key == "caller"
        assert fact.attribute_key == "preferred_visit_day"
        assert fact.requires_confirmation is True
    finally:
        await db.close()


# ---------------------------------------------------------------------------
# Epic 2 — Temporal validity + supersede
# ---------------------------------------------------------------------------


async def test_eval_2_1_contradiction_supersedes(db: ContextDB) -> None:
    """EVAL-2.1: '3pm' then 'actually 4pm' => recall returns only 4pm."""
    first = await db.factual.add(
        "The meeting is at 3pm",
        source="user_stated",
        confidence=0.9,
        action_relevant=True,
        entity="meeting",
        attribute="time",
    )
    second = await db.factual.add(
        "Actually, the meeting is at 4pm",
        source="user_stated",
        confidence=0.9,
        action_relevant=True,
        entity="meeting",
        attribute="time",
    )
    assert second.id != first.id

    recalled = await db.factual.recall("when is the meeting")
    contents = [m.content for m in recalled]
    assert any("4pm" in c for c in contents), contents
    assert not any("3pm" in c for c in contents), contents

    # The superseded memory is retained (audit) but closed out.
    old = await db.get(first.id)
    assert old is not None
    assert old.valid_until is not None
    assert old.superseded_by == second.id


async def test_eval_2_2_as_of_returns_historical_value(db: ContextDB) -> None:
    """EVAL-2.2: as_of before the correction returns 3pm."""
    await db.factual.add(
        "The meeting is at 3pm",
        source="user_stated",
        confidence=0.9,
        action_relevant=True,
        entity="meeting",
        attribute="time",
    )
    between = datetime.now(tz=timezone.utc)
    await db.factual.add(
        "Actually, the meeting is at 4pm",
        source="user_stated",
        confidence=0.9,
        action_relevant=True,
        entity="meeting",
        attribute="time",
    )
    historical = await db.factual.recall("when is the meeting", as_of=between)
    contents = [m.content for m in historical]
    assert any("3pm" in c for c in contents), contents
    assert not any("4pm" in c for c in contents), contents


async def test_eval_2_3_audit_log_shows_supersede_edge(db: ContextDB) -> None:
    """EVAL-2.3: the audit log records the supersede edge."""
    first = await db.factual.add(
        "The meeting is at 3pm",
        source="user_stated",
        confidence=0.9,
        action_relevant=True,
        entity="meeting",
        attribute="time",
    )
    second = await db.factual.add(
        "Actually, the meeting is at 4pm",
        source="user_stated",
        confidence=0.9,
        action_relevant=True,
        entity="meeting",
        attribute="time",
    )
    assert db.audit is not None
    history = await db.audit.get_history(memory_id=first.id)
    supersede = [e for e in history if e.operation == "SUPERSEDE"]
    assert supersede, "no SUPERSEDE audit entry"
    assert supersede[0].details["superseded_by"] == second.id
    assert await db.audit.verify_chain()


# ---------------------------------------------------------------------------
# Epic 3 — Latency-tiered write path
# ---------------------------------------------------------------------------


async def test_eval_3_1_write_p95_under_10ms_while_consolidation_runs(
    tmp_path: Path,
) -> None:
    """EVAL-3.1: 1K add_fast writes with p95 < 10ms while the consolidator
    churns through the pending queue behind the write."""
    db = contextdb.init(user_id="eval-user", config=make_config(tmp_path))
    slow_llm = MockLLM()
    original_generate = slow_llm.generate

    async def slow_generate(*args: Any, **kwargs: Any) -> str:
        # Simulate a real extraction call occupying the event loop.
        await asyncio.sleep(0.0005)
        return await original_generate(*args, **kwargs)

    slow_llm.generate = slow_generate  # type: ignore[method-assign]
    db._llm = slow_llm
    try:
        await db._ensure_init()
        stop = asyncio.Event()

        async def churn() -> None:
            while not stop.is_set():
                await db.consolidate_pending(batch_size=25)
                await asyncio.sleep(0.001)

        task = asyncio.create_task(churn())
        latencies_ms: list[float] = []
        for i in range(1000):
            start = time.perf_counter()
            await db.factual.add_fast(f"caller utterance number {i} about ordering pizza")
            latencies_ms.append((time.perf_counter() - start) * 1000.0)
        stop.set()
        await task

        latencies_ms.sort()
        p95 = latencies_ms[int(0.95 * len(latencies_ms)) - 1]
        p50 = latencies_ms[len(latencies_ms) // 2]
        assert p95 < 10.0, f"write p95 {p95:.2f}ms (p50 {p50:.2f}ms) exceeds 10ms"
    finally:
        await db.close()


async def test_eval_3_2_fast_write_recallable_before_consolidation(
    tmp_path: Path,
) -> None:
    """EVAL-3.2: a fast-written fact is recallable immediately, marked
    pending_consolidation, and the write never touched the LLM."""
    db = contextdb.init(user_id="eval-user", config=make_config(tmp_path))
    spy_llm = MockLLM()
    db._llm = spy_llm
    try:
        item = await db.factual.add_fast("The caller prefers email over phone")
        assert item.pending_consolidation is True
        assert item.confidence == 0.5
        assert item.epistemic_source == "user_stated"

        recalled = await db.factual.recall("prefers email")
        assert any(m.id == item.id for m in recalled)

        assert spy_llm.calls == [], "add_fast must never call the LLM"

        # Consolidation drains the queue behind the write.
        drained = await db.consolidate_pending()
        assert drained >= 1
        assert db._store is not None
        after = await db._store.get_raw(item.id)
        assert after is not None
        assert after.pending_consolidation is False
    finally:
        await db.close()
