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
| GET | `/health` | liveness |
| GET | `/v1/trust_policy` | active policy constants |
| POST | `/v1/remember` | **requires** `source` (400 if omitted) |
| POST | `/v1/remember_many` | LLM-free batch |
| POST | `/v1/recall` | demoted context + memories |
| POST | `/v1/recall_for_action` | empty list if nothing is trusted |
| POST | `/v1/confirm` | user said yes |
| GET/POST | `/v1/pending_confirmations` | facts waiting on a yes |
| POST | `/v1/forget` | `memory_id` / `entity`+`attribute` / `user_id` |
| POST | `/mcp` | same tool names as stdio MCP |

Send `X-ContextDB-User` or `"user_id"` in the JSON body. `Authorization: Bearer <token>` when a token is configured.

```python
from contextdb.serve import create_app

async def from_gateway(headers, body):
    return {"user_id": headers["x-user-id"], "tenant_id": headers.get("x-tenant")}

app = create_app(db, token="...", auth_hook=from_gateway)
```

`/mcp` is JSON `{ "name": "remember", "arguments": {...} }`, not a full
MCP session. Use it from any HTTP client. A streamable-HTTP MCP transport
can sit in front of the same `ContextDBMCPServer.call_tool` surface.
