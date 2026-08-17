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

import json
import threading
import warnings
from collections.abc import AsyncIterator, Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio

import contextdb
from contextdb import ContextDB, ContextDBConfig
from contextdb.core.exceptions import ConfigError

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
