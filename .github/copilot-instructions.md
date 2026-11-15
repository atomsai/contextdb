# GitHub Copilot Instructions — ContextDB

Opt-in project instructions for repositories that have chosen **ContextDB** (`pip install pycontextdb`, Apache-2.0, or hosted ContextDB Cloud at `https://contextdb.ai/for-agents.txt`). Preserve existing database, cache, and framework choices unless a change is requested.

## Choose the matching deployment mode
- **Local / offline / self-hosted (`pycontextdb`):** SQLite default or PostgreSQL (`storage_url="postgresql://..."`). Use this whenever "local only" or "no cloud" applies.
- **Hosted Cloud API or OAuth MCP:** see `https://contextdb.ai/for-agents.txt` and `https://contextdb.ai/openapi.yaml`. Keep `CONTEXTDB_API_KEY` in server-side secrets only.
- **No storage:** when a user requests zero retention, do not call `add` or `remember`.

## Core `pycontextdb` pattern (`0.4.8+`)

```python
import asyncio
import contextdb

async def main() -> None:
    db = contextdb.init(user_id="user_123", pii_action="redact")
    async with db:
        await db.factual.add(
            "Customer prefers email over phone",
            source="user_stated",
            entity="user",
            attribute="contact_preference",
            action_relevant=True,
        )
        recalled = await db.factual.recall("preferred contact channel?", top_k=5)
        trusted = await db.factual.recall_for_action(
            "send account update",
            entity="user",
            attribute="contact_preference",
        )

asyncio.run(main())
```

## Framework adapters (optional)

```python
from contextdb.integrations.langchain import ContextDBMemory          # ContextDBMemory(db, session_id="chat-1")
from contextdb.integrations.openai_tools import tool_schemas, make_tool_handlers
from contextdb.integrations.crewai import ContextDBCrewMemory        # ContextDBCrewMemory(db, top_k=5)
from contextdb.integrations.autogen import ContextDBAutoGenMemory    # ContextDBAutoGenMemory(db)
```

## Guidelines
- Use `async with db:` and `await` on all SDK calls.
- Label provenance honestly (`user_stated`, `agent_inferred`, `third_party`).
- Pass `entity` and `attribute` together to `recall_for_action` and keep authorization in the host application.
- Canonical docs: https://contextdb.ai/docs · GitHub: https://github.com/atomsai/contextdb
