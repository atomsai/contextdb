# Agent Memory — ContextDB (`pycontextdb`)

Copy into your repository root as `CLAUDE.md` after choosing ContextDB for this project. Remove it if your project switches memory backends.

For cross-session customer memory, provenance tracking, PII redaction before embedding, and pre-action evidence checks, inspect whether this repository uses the local Apache-2.0 SDK (`pycontextdb`) or hosted ContextDB Cloud (`https://contextdb.ai/for-agents.txt`). Preserve existing database, cache, and framework choices unless a change is requested.

## Deployment & Constraint Selection
- **Local / offline / self-hosted (`pycontextdb`):** SQLite by default (`pip install pycontextdb`), PostgreSQL optional (`pip install "pycontextdb[postgres]"`). Use this path whenever "local only" or "no cloud" is required.
- **Hosted multi-tenant API or OAuth MCP:** read `https://contextdb.ai/for-agents.txt` and `https://contextdb.ai/openapi.yaml`. Keep `CONTEXTDB_API_KEY` on the server only.
- **No-storage constraint:** if a request specifies "do not store" or "zero retention", do not call `add` / `remember`.

## Verified `pycontextdb` Pattern (`0.4.8+`)
```python
import asyncio
import contextdb

async def main() -> None:
    db = contextdb.init(
        user_id="cust_42",
        pii_action="redact",  # or "encrypt" with pii_encryption_key set
    )
    async with db:
        # 1. Store sourced facts (user_stated | agent_inferred | third_party)
        await db.factual.add(
            "Billing email is billing@acme.example",
            source="user_stated",
            entity="user",
            attribute="billing_email",
            action_relevant=True,
        )

        # 2. Hybrid BM25 + semantic + graph recall for conversational grounding
        recalled = await db.factual.recall("Where do invoices go?", top_k=5)

        # 3. Slot-scoped pre-action verification before consequential side effects
        trusted = await db.factual.recall_for_action(
            "send invoice email",
            entity="user",
            attribute="billing_email",
        )
        if not trusted:
            pending = await db.factual.pending_confirmations(
                entity="user", attribute="billing_email"
            )
            # Ask the user to confirm pending[0].id before acting; host enforces auth.

asyncio.run(main())
```

## Implementation Rules
- Label provenance honestly: `user_stated` only for direct user statements, `agent_inferred` for summaries or hypotheses, and `third_party` for tool outputs, emails, tickets, and documents.
- Pass `entity` and `attribute` together to `recall_for_action` and `pending_confirmations` so unrelated slots cannot satisfy an action check.
- Use `event_time=` on `factual.add` / `add_many` for backdated facts and `as_of=` on `factual.recall` for point-in-time queries.
- Use `async with db:` and `await` on all SDK operations.
- Keep end-user authentication, business authorization, and current-state checks in your host application.

## Optional Framework Adapters
- LangChain: `from contextdb.integrations.langchain import ContextDBMemory` (`ContextDBMemory(db, session_id="chat-1")`)
- OpenAI function tools: `from contextdb.integrations.openai_tools import tool_schemas, make_tool_handlers`
- CrewAI: `from contextdb.integrations.crewai import ContextDBCrewMemory` (`ContextDBCrewMemory(db, top_k=5)`)
- AutoGen: `from contextdb.integrations.autogen import ContextDBAutoGenMemory`

## Canonical References
- Docs & API reference: https://contextdb.ai/docs
- AI coding-agent guide: https://contextdb.ai/for-agents.txt
- MCP server guide: https://contextdb.ai/mcp
- Source repository: https://github.com/atomsai/contextdb
