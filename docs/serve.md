# HTTP server

stdio MCP only works for a local Python process. Node, Go, and serverless
hosts need a port.

```bash
pip install 'pycontextdb[serve]'
contextdb serve --http --host 127.0.0.1 --port 8080 --token "$CONTEXTDB_SERVE_TOKEN"
```

Or:

```python
import contextdb
from contextdb.serve import serve_http

db = contextdb.init()  # no user_id — each request supplies one
await serve_http(db, host="0.0.0.0", port=8080, token="...")
```

Without a token the server only binds loopback. Non-loopback requires
`--token` / `CONTEXTDB_SERVE_TOKEN` or an `auth_hook`.

## Routes

| Method | Path | Notes |
|---|---|---|
| GET | `/health` | liveness — the only unauthenticated route |
| GET | `/v1/trust_policy` | active policy constants; **requires auth when auth is configured** |
| POST | `/v1/remember` | **requires** `source` (400 if omitted) |
| POST | `/v1/remember_many` | LLM-free batch |
| POST | `/v1/recall` | demoted context + memories |
| POST | `/v1/recall_for_action` | empty list if nothing is trusted |
| POST | `/v1/confirm` | user said yes; foreign `memory_id` → 400 |
| GET/POST | `/v1/pending_confirmations` | facts waiting on a yes |
| POST | `/v1/forget` | `memory_id` / `entity`+`attribute` / whole resolved user |
| POST | `/mcp` | same tool names as stdio MCP; runs in the authenticated scope |

Every route except `/health` (and its alias `/v1/health`) runs the
configured authentication — token and/or `auth_hook` — including
`/v1/trust_policy` and `/mcp`.

## Request scoping

Send `X-ContextDB-User` or `"user_id"` in the JSON body. `Authorization: Bearer <token>` when a token is configured.

When `auth_hook` returns a `user_id`, that authenticated scope **wins**:
a conflicting `user_id` in the JSON body, the `X-ContextDB-User` header,
the query string, or MCP tool arguments is rejected with `400`
(`ScopeConflictError`) — the server never silently picks one scope over
another. A matching `user_id` is accepted; an absent one means the
request runs as the authenticated user.

`/mcp` propagates the resolved scope into every tool call, so an
authenticated caller cannot recall, confirm, or forget another user's
memories by passing their `user_id` in tool arguments.

ID-based `confirm` and `forget` verify the target belongs to the resolved
scope. A `memory_id` owned by a different user fails with `400`, exactly
like an unknown id — a request can never read, confirm, or delete a known
foreign memory.

```python
from contextdb.serve import create_app

async def from_gateway(headers, body):
    return {"user_id": headers["x-user-id"], "tenant_id": headers.get("x-tenant")}

app = create_app(db, token="...", auth_hook=from_gateway)
```

`/mcp` is JSON `{ "name": "remember", "arguments": {...} }`, not a full
MCP session. Use it from any HTTP client. A streamable-HTTP MCP transport
can sit in front of the same `ContextDBMCPServer.call_tool` surface.

## Anonymous loopback is not an authorization boundary

`allow_anonymous=True` (or `serve_http` on a loopback address with no
token) exists for local, single-tenant development: requests may pass any
`user_id` per call, or none at all. **Nothing authenticates or isolates
those callers** — any process that can reach the port can read, confirm,
or delete any user's memories. Never expose an anonymous server beyond
loopback; put a token or an `auth_hook` in front of anything else.
