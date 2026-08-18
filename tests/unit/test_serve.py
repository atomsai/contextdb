from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import contextdb
from contextdb.core.config import ContextDBConfig
from contextdb.core.exceptions import UnauthorizedError
from contextdb.serve import create_app

httpx = pytest.importorskip("httpx")
pytest.importorskip("starlette")


def _cfg(tmp_path: Path, name: str = "serve.db") -> ContextDBConfig:
    return ContextDBConfig(
        storage_url=f"sqlite:///{tmp_path}/{name}",
        embedding_model="mock",
        embedding_dim=32,
        llm_model="mock",
        llm_api_key="mock",
        enable_entity_graph=False,
        enable_multi_graph=False,
        enable_auto_link=False,
    )


def _http_client(app: Any) -> Any:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://t"
    )


@pytest.mark.asyncio
async def test_http_remember_requires_source_and_scopes_user(tmp_path: Path) -> None:
    db = contextdb.init(config=_cfg(tmp_path))
    app = create_app(db, token="secret")
    try:
        async with _http_client(app) as client:
            denied = await client.post("/v1/remember", json={"content": "x"})
            assert denied.status_code == 401
            missing = await client.post(
                "/v1/remember",
                headers={"Authorization": "Bearer secret", "X-ContextDB-User": "u1"},
                json={"content": "x"},
            )
            assert missing.status_code == 400
            ok = await client.post(
                "/v1/remember",
                headers={"Authorization": "Bearer secret", "X-ContextDB-User": "u1"},
                json={"content": "Thursday works", "source": "user_stated"},
            )
            assert ok.status_code == 200
            recall = await client.post(
                "/v1/recall",
                headers={"Authorization": "Bearer secret", "X-ContextDB-User": "u2"},
                json={"query": "Thursday"},
            )
            assert recall.status_code == 200
            assert recall.json()["memories"] == []
            pending = await client.get(
                "/v1/pending_confirmations",
                headers={"Authorization": "Bearer secret", "X-ContextDB-User": "u1"},
            )
            assert pending.status_code == 200
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_non_health_routes_require_configured_auth(tmp_path: Path) -> None:
    """Every route except /health runs authentication when it is configured —
    including /v1/trust_policy and /mcp."""
    db = contextdb.init(config=_cfg(tmp_path))
    app = create_app(db, token="secret")
    try:
        async with _http_client(app) as client:
            assert (await client.get("/health")).status_code == 200
            assert (await client.get("/v1/health")).status_code == 200

            denied_policy = await client.get("/v1/trust_policy")
            assert denied_policy.status_code == 401
            denied_mcp = await client.post(
                "/mcp", json={"name": "recall", "arguments": {"query": "x"}}
            )
            assert denied_mcp.status_code == 401

            auth = {"Authorization": "Bearer secret"}
            ok_policy = await client.get("/v1/trust_policy", headers=auth)
            assert ok_policy.status_code == 200
            assert "relevance_floor" in ok_policy.json()
            ok_mcp = await client.post(
                "/mcp",
                headers=auth,
                json={"name": "recall", "arguments": {"query": "x"}},
            )
            assert ok_mcp.status_code == 200
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_trust_policy_requires_auth_with_auth_hook(tmp_path: Path) -> None:
    def hook(headers: dict[str, str], body: dict[str, Any]) -> dict[str, Any]:
        user = headers.get("x-user-id")
        if not user:
            raise UnauthorizedError("unauthorized")
        return {"user_id": user}

    db = contextdb.init(config=_cfg(tmp_path))
    app = create_app(db, auth_hook=hook)
    try:
        async with _http_client(app) as client:
            assert (await client.get("/v1/trust_policy")).status_code == 401
            ok = await client.get("/v1/trust_policy", headers={"X-User-Id": "alice"})
            assert ok.status_code == 200
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_auth_hook_user_cannot_be_overridden(tmp_path: Path) -> None:
    """JSON body, headers, and query string must not override the
    authenticated user; a conflict is a clear 400, not a silent pick."""

    def hook(headers: dict[str, str], body: dict[str, Any]) -> dict[str, Any]:
        return {"user_id": "alice"}

    db = contextdb.init(config=_cfg(tmp_path))
    app = create_app(db, auth_hook=hook)
    try:
        async with _http_client(app) as client:
            by_body = await client.post(
                "/v1/remember",
                json={"content": "x", "source": "user_stated", "user_id": "bob"},
            )
            assert by_body.status_code == 400
            assert "conflict" in by_body.json()["error"].lower()

            by_header = await client.post(
                "/v1/remember",
                headers={"X-ContextDB-User": "bob"},
                json={"content": "x", "source": "user_stated"},
            )
            assert by_header.status_code == 400

            by_query = await client.post(
                "/v1/recall?user_id=bob", json={"query": "x"}
            )
            assert by_query.status_code == 400

            by_tool_args = await client.post(
                "/mcp",
                json={"name": "recall", "arguments": {"query": "x", "user_id": "bob"}},
            )
            assert by_tool_args.status_code == 400

            # Whole-user erasure cannot be retargeted by the body either.
            retarget = await client.post("/v1/forget", json={"user_id": "bob"})
            assert retarget.status_code == 400

            # No override: the request runs as the authenticated user.
            ok = await client.post(
                "/v1/remember",
                json={"content": "alice fact", "source": "user_stated"},
            )
            assert ok.status_code == 200
            assert ok.json()["memory"]["user_id"] == "alice"

            # A matching user_id is not a conflict.
            matching = await client.post(
                "/v1/remember",
                json={
                    "content": "alice fact 2",
                    "source": "user_stated",
                    "user_id": "alice",
                },
            )
            assert matching.status_code == 200
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_conflicting_request_scopes_rejected_without_auth_user(
    tmp_path: Path,
) -> None:
    """Token auth authenticates the service, not a user: two disagreeing
    request-supplied scopes are still a 400, never a silent pick."""
    db = contextdb.init(config=_cfg(tmp_path))
    app = create_app(db, token="secret")
    try:
        async with _http_client(app) as client:
            conflict = await client.post(
                "/v1/remember",
                headers={"Authorization": "Bearer secret", "X-ContextDB-User": "u1"},
                json={"content": "x", "source": "user_stated", "user_id": "u2"},
            )
            assert conflict.status_code == 400
            agree = await client.post(
                "/v1/remember",
                headers={"Authorization": "Bearer secret", "X-ContextDB-User": "u1"},
                json={"content": "x", "source": "user_stated", "user_id": "u1"},
            )
            assert agree.status_code == 200
            assert agree.json()["memory"]["user_id"] == "u1"
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_mcp_cannot_cross_user_scope(tmp_path: Path) -> None:
    """/mcp propagates the authenticated scope into every tool call: an
    authenticated caller cannot recall, confirm, or forget another user's
    memory through tool arguments."""

    def hook(headers: dict[str, str], body: dict[str, Any]) -> dict[str, Any]:
        return {"user_id": "alice"}

    db = contextdb.init(config=_cfg(tmp_path))
    app = create_app(db, auth_hook=hook)
    try:
        bob_item = await db.factual.add(
            "Bob likes green", source="user_stated", user_id="bob"
        )
        async with _http_client(app) as client:
            remembered = await client.post(
                "/mcp",
                json={
                    "name": "remember",
                    "arguments": {
                        "content": "Alice PIN is 7788",
                        "source": "user_stated",
                    },
                },
            )
            assert remembered.status_code == 200
            assert remembered.json()["memory"]["user_id"] == "alice"

            # Recall runs as alice: bob's memory is never returned, even
            # though the query targets it.
            recalled = await client.post(
                "/mcp", json={"name": "recall", "arguments": {"query": "green"}}
            )
            assert recalled.status_code == 200
            hits = recalled.json()["memories"]
            assert all(m["user_id"] == "alice" for m in hits)
            assert all("green" not in m["content"] for m in hits)

            # Tool arguments cannot override the authenticated scope.
            override = await client.post(
                "/mcp",
                json={
                    "name": "recall",
                    "arguments": {"query": "green", "user_id": "bob"},
                },
            )
            assert override.status_code == 400

            # ID-based tools reject bob's memory id when running as alice.
            foreign_confirm = await client.post(
                "/mcp",
                json={"name": "confirm", "arguments": {"memory_id": bob_item.id}},
            )
            assert foreign_confirm.status_code == 400
            foreign_forget = await client.post(
                "/mcp",
                json={"name": "forget", "arguments": {"memory_id": bob_item.id}},
            )
            assert foreign_forget.status_code == 400

        still = await db.get(bob_item.id)
        assert still is not None
        assert still.confirmed is False
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_confirm_and_forget_reject_foreign_memory_id(tmp_path: Path) -> None:
    """A request must never confirm or delete a known foreign memory ID."""
    db = contextdb.init(config=_cfg(tmp_path))
    app = create_app(db, token="secret")
    try:
        alice_item = await db.factual.add(
            "Alice PIN is 7788",
            source="user_stated",
            confidence=0.5,
            action_relevant=True,
            entity="account",
            attribute="pin",
            user_id="alice",
        )
        as_bob = {"Authorization": "Bearer secret", "X-ContextDB-User": "bob"}
        as_alice = {"Authorization": "Bearer secret", "X-ContextDB-User": "alice"}
        async with _http_client(app) as client:
            confirm = await client.post(
                "/v1/confirm", headers=as_bob, json={"memory_id": alice_item.id}
            )
            assert confirm.status_code == 400
            still = await db.get(alice_item.id)
            assert still is not None
            assert still.confirmed is False

            forget = await client.post(
                "/v1/forget", headers=as_bob, json={"memory_id": alice_item.id}
            )
            assert forget.status_code == 400
            assert await db.get(alice_item.id) is not None

            ok_confirm = await client.post(
                "/v1/confirm", headers=as_alice, json={"memory_id": alice_item.id}
            )
            assert ok_confirm.status_code == 200
            assert ok_confirm.json()["memory"]["confirmed"] is True

            ok_forget = await client.post(
                "/v1/forget", headers=as_alice, json={"memory_id": alice_item.id}
            )
            assert ok_forget.status_code == 200
            assert ok_forget.json()["deleted"] == 1
            assert await db.get(alice_item.id) is None
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_anonymous_loopback_behavior_unchanged(tmp_path: Path) -> None:
    """No configured auth + allow_anonymous: unscoped local usage still works.
    This is a convenience for loopback deployments, not an authorization
    boundary (see docs/multi_tenant.md)."""
    db = contextdb.init(config=_cfg(tmp_path))
    app = create_app(db, allow_anonymous=True)
    try:
        async with _http_client(app) as client:
            remembered = await client.post(
                "/v1/remember",
                json={"content": "local loopback fact", "source": "user_stated"},
            )
            assert remembered.status_code == 200
            assert remembered.json()["memory"]["user_id"] is None
            mid = remembered.json()["memory"]["id"]

            recalled = await client.post("/v1/recall", json={"query": "loopback"})
            assert recalled.status_code == 200
            assert any(m["id"] == mid for m in recalled.json()["memories"])

            confirmed = await client.post("/v1/confirm", json={"memory_id": mid})
            assert confirmed.status_code == 200

            forgotten = await client.post("/v1/forget", json={"memory_id": mid})
            assert forgotten.status_code == 200
            assert forgotten.json()["deleted"] == 1

            # Per-call user_id remains available without auth (documented
            # convenience — anyone who can reach the port can claim it).
            scoped = await client.post(
                "/v1/remember",
                headers={"X-ContextDB-User": "u1"},
                json={"content": "u1 fact", "source": "user_stated"},
            )
            assert scoped.status_code == 200
            assert scoped.json()["memory"]["user_id"] == "u1"

            assert (await client.get("/v1/trust_policy")).status_code == 200
    finally:
        await db.close()
