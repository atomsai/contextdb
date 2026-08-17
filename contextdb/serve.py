"""Network server — HTTP JSON plus optional MCP streamable HTTP.

Node, Go, and Cloud Run hosts cannot speak stdio MCP. This module is the
replacement for the FastAPI wrapper every integrator otherwise writes.

Install: ``pip install 'pycontextdb[serve]'``.

Auth: set ``CONTEXTDB_SERVE_TOKEN`` or pass ``token=``. Requests must send
``Authorization: Bearer <token>``. Bind to a loopback address if you run
without a token. Pass ``auth_hook`` to mint user/tenant from your own
gateway headers.
"""

from __future__ import annotations

import inspect
import os
from collections.abc import Awaitable, Callable
from typing import Any

from contextdb.client import ContextDB
from contextdb.core.exceptions import ConfigError, ContextDBError, SourceRequiredError
from contextdb.core.models import MemoryItem
from contextdb.mcp import ContextDBMCPServer

AuthHook = Callable[[dict[str, str], dict[str, Any]], Awaitable[dict[str, Any]] | dict[str, Any]]


def _serialize(item: MemoryItem) -> dict[str, Any]:
    return {
        "id": item.id,
        "content": item.content,
        "user_id": item.user_id,
        "epistemic_source": item.epistemic_source,
        "confidence": item.confidence,
        "corroboration_count": item.corroboration_count,
        "action_relevant": item.action_relevant,
        "requires_confirmation": item.requires_confirmation,
        "confirmed": item.confirmed,
        "independent_corroboration": item.independent_corroboration,
        "injection_suspect": item.injection_suspect,
        "entity_key": item.entity_key,
        "attribute_key": item.attribute_key,
        "valid_until": item.valid_until.isoformat() if item.valid_until else None,
    }


def _json_response(status: int, payload: dict[str, Any]) -> Any:
    from starlette.responses import JSONResponse

    return JSONResponse(payload, status_code=status)


async def _read_json(request: Any) -> dict[str, Any]:
    try:
        data = await request.json()
    except Exception:  # noqa: BLE001
        return {}
    return data if isinstance(data, dict) else {}


def create_app(
    client: ContextDB,
    *,
    token: str | None = None,
    auth_hook: AuthHook | None = None,
    allow_anonymous: bool = False,
) -> Any:
    """Build a Starlette app around one :class:`ContextDB` client.

    The client should be unscoped (``init()`` without ``user_id``) so each
    request can pass ``user_id``. A scoped client still works; it will reject
    a request for a different user.
    """
    try:
        from starlette.applications import Starlette
        from starlette.requests import Request
        from starlette.routing import Route
    except ImportError as exc:
        raise ConfigError(
            "HTTP serve requires Starlette. Install with `pip install 'pycontextdb[serve]'`."
        ) from exc

    resolved_token = token if token is not None else os.environ.get("CONTEXTDB_SERVE_TOKEN")
    mcp = ContextDBMCPServer(client)

    async def _authorize(request: Request, body: dict[str, Any]) -> dict[str, Any]:
        headers = {k.lower(): v for k, v in request.headers.items()}
        if auth_hook is not None:
            extra = auth_hook(headers, body)
            if inspect.isawaitable(extra):
                extra = await extra
            if not isinstance(extra, dict):
                raise ConfigError("auth_hook must return a dict")
            return extra
        if resolved_token:
            auth = headers.get("authorization", "")
            if auth != f"Bearer {resolved_token}":
                raise ConfigError("unauthorized")
        elif not allow_anonymous:
            raise ConfigError(
                "Refusing anonymous HTTP. Set CONTEXTDB_SERVE_TOKEN, pass "
                "token=, or allow_anonymous=True on loopback only."
            )
        return {}

    def _user_id(request: Request, body: dict[str, Any], auth: dict[str, Any]) -> str | None:
        return (
            (auth.get("user_id") if isinstance(auth.get("user_id"), str) else None)
            or body.get("user_id")
            or request.headers.get("x-contextdb-user")
            or client.user_id
        )

    async def health(_request: Request) -> Any:
        return _json_response(200, {"ok": True, "service": "contextdb"})

    async def trust_policy(_request: Request) -> Any:
        return _json_response(200, client.trust_policy.model_dump())

    async def remember(request: Request) -> Any:
        body = await _read_json(request)
        try:
            auth = await _authorize(request, body)
            if not body.get("source"):
                raise SourceRequiredError(
                    "remember requires source: user_stated | agent_inferred | third_party"
                )
            uid = _user_id(request, body, auth)
            item = await client.factual.add(
                str(body["content"]),
                source=body.get("source"),
                confidence=float(body.get("confidence", 1.0)),
                action_relevant=body.get("action_relevant"),
                entity=body.get("entity"),
                attribute=body.get("attribute"),
                user_id=uid,
            )
            return _json_response(200, {"memory": _serialize(item)})
        except SourceRequiredError as exc:
            return _json_response(400, {"error": str(exc)})
        except ContextDBError as exc:
            code = 401 if str(exc) == "unauthorized" else 400
            return _json_response(code, {"error": str(exc)})

    async def remember_many(request: Request) -> Any:
        body = await _read_json(request)
        try:
            auth = await _authorize(request, body)
            uid = _user_id(request, body, auth)
            items_in = body.get("items") or body.get("contents") or []
            if not isinstance(items_in, list):
                raise ConfigError("items must be a list")
            stored = await client.factual.add_many(items_in, user_id=uid)
            return _json_response(200, {"memories": [_serialize(m) for m in stored]})
        except ContextDBError as exc:
            code = 401 if str(exc) == "unauthorized" else 400
            return _json_response(code, {"error": str(exc)})

    async def recall(request: Request) -> Any:
        body = await _read_json(request)
        try:
            auth = await _authorize(request, body)
            uid = _user_id(request, body, auth)
            items = await client.factual.recall(
                str(body["query"]),
                top_k=int(body.get("top_k", 5)),
                user_id=uid,
                entity=body.get("entity"),
                min_confidence=body.get("min_confidence"),
                include_third_party=bool(body.get("include_third_party", True)),
            )
            from contextdb.integrations.prompting import render_recalled_context

            return _json_response(
                200,
                {
                    "context": render_recalled_context(
                        items, max_tokens=int(body.get("max_tokens", 512))
                    ),
                    "memories": [_serialize(m) for m in items],
                },
            )
        except ContextDBError as exc:
            code = 401 if str(exc) == "unauthorized" else 400
            return _json_response(code, {"error": str(exc)})

    async def recall_for_action(request: Request) -> Any:
        body = await _read_json(request)
        try:
            auth = await _authorize(request, body)
            uid = _user_id(request, body, auth)
            items = await client.factual.recall_for_action(
                str(body["query"]),
                top_k=int(body.get("top_k", 5)),
                user_id=uid,
            )
            return _json_response(200, {"memories": [_serialize(m) for m in items]})
        except ContextDBError as exc:
            code = 401 if str(exc) == "unauthorized" else 400
            return _json_response(code, {"error": str(exc)})

    async def confirm(request: Request) -> Any:
        body = await _read_json(request)
        try:
            auth = await _authorize(request, body)
            uid = _user_id(request, body, auth)
            item = await client.factual.confirm(str(body["memory_id"]), user_id=uid)
            return _json_response(200, {"memory": _serialize(item)})
        except ContextDBError as exc:
            code = 401 if str(exc) == "unauthorized" else 400
            return _json_response(code, {"error": str(exc)})

    async def pending(request: Request) -> Any:
        body = await _read_json(request) if request.method == "POST" else {}
        try:
            auth = await _authorize(request, body)
            uid = _user_id(request, body, auth) or request.query_params.get("user_id")
            items = await client.factual.pending_confirmations(user_id=uid)
            return _json_response(200, {"memories": [_serialize(m) for m in items]})
        except ContextDBError as exc:
            code = 401 if str(exc) == "unauthorized" else 400
            return _json_response(code, {"error": str(exc)})

    async def forget(request: Request) -> Any:
        body = await _read_json(request)
        try:
            auth = await _authorize(request, body)
            uid = _user_id(request, body, auth)
            if body.get("memory_id"):
                deleted = await client.forget(
                    user_id=uid,
                    memory_id=str(body["memory_id"]),
                )
            elif body.get("entity") and body.get("attribute"):
                deleted = await client.forget(
                    user_id=uid,
                    entity=str(body["entity"]),
                    attribute=str(body["attribute"]),
                )
            elif body.get("user_id") or uid:
                target = str(body.get("user_id") or uid)
                deleted = await client.forget_user(target)
                verified = await client.verify_forgotten(target)
                return _json_response(200, {"deleted": deleted, "verified": verified})
            else:
                raise ConfigError("forget requires memory_id, entity+attribute, or user_id")
            return _json_response(200, {"deleted": deleted})
        except ContextDBError as exc:
            code = 401 if str(exc) == "unauthorized" else 400
            return _json_response(code, {"error": str(exc)})

    async def mcp_call(request: Request) -> Any:
        """JSON-RPC-ish MCP tools/call for hosts that already speak MCP names."""
        body = await _read_json(request)
        try:
            await _authorize(request, body)
            name = str(body.get("name") or body.get("method") or "")
            arguments = body.get("arguments") or body.get("params") or {}
            if not isinstance(arguments, dict):
                raise ConfigError("arguments must be an object")
            result = await mcp.call_tool(name, arguments)
            return _json_response(200, result)
        except ContextDBError as exc:
            code = 401 if str(exc) == "unauthorized" else 400
            return _json_response(code, {"error": str(exc)})

    routes = [
        Route("/health", health, methods=["GET"]),
        Route("/v1/health", health, methods=["GET"]),
        Route("/v1/trust_policy", trust_policy, methods=["GET"]),
        Route("/v1/remember", remember, methods=["POST"]),
        Route("/v1/remember_many", remember_many, methods=["POST"]),
        Route("/v1/recall", recall, methods=["POST"]),
        Route("/v1/recall_for_action", recall_for_action, methods=["POST"]),
        Route("/v1/confirm", confirm, methods=["POST"]),
        Route("/v1/pending_confirmations", pending, methods=["GET", "POST"]),
        Route("/v1/forget", forget, methods=["POST"]),
        Route("/mcp", mcp_call, methods=["POST"]),
    ]
    return Starlette(routes=routes)


async def serve_http(
    client: ContextDB,
    *,
    host: str = "127.0.0.1",
    port: int = 8080,
    token: str | None = None,
    auth_hook: AuthHook | None = None,
    allow_anonymous: bool = False,
) -> None:
    """Run uvicorn until cancelled."""
    try:
        import uvicorn
    except ImportError as exc:
        raise ConfigError(
            "HTTP serve requires uvicorn. Install with `pip install 'pycontextdb[serve]'`."
        ) from exc
    loopback = host in {"127.0.0.1", "localhost", "::1"}
    if not token and not os.environ.get("CONTEXTDB_SERVE_TOKEN") and not allow_anonymous:
        if not loopback:
            raise ConfigError(
                "Refusing to bind a non-loopback address without a token. "
                "Set CONTEXTDB_SERVE_TOKEN or pass --token."
            )
        allow_anonymous = True
    app = create_app(
        client,
        token=token,
        auth_hook=auth_hook,
        allow_anonymous=allow_anonymous,
    )
    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()


def serve_http_sync(
    client: ContextDB,
    *,
    host: str = "127.0.0.1",
    port: int = 8080,
    token: str | None = None,
    allow_anonymous: bool = False,
) -> None:
    import asyncio

    asyncio.run(
        serve_http(
            client,
            host=host,
            port=port,
            token=token,
            allow_anonymous=allow_anonymous,
        )
    )
