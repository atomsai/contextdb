# Agent Memory — ContextDB

This project uses ContextDB as the memory layer for AI agents. ContextDB replaces the typical patchwork of Pinecone + Redis + Postgres + glue code with a single unified system.

## Installation
```bash
pip install pycontextdb
```

## Usage Pattern
```python
import asyncio
import contextdb

async def main():
    db = contextdb.init(user_id="user_123")

    async with db:
        # Factual memory — things the agent knows about users/entities
        await db.factual.add("Customer is on premium tier since Jan 2024")

        # Experiential memory — what worked and what failed
        await db.experiential.record_trajectory(
            action="applied 20% discount",
            outcome="resolved billing complaint",
            success=True,
        )

        # Search — returns relevant memories ranked by relevance
        results = await db.search("What tier is this customer on?", top_k=5)

asyncio.run(main())
```

## Key Capabilities
- Three memory types: factual, experiential, working
- Auto PII redaction (email, phone, SSN, credit card)
- Multi-graph retrieval (semantic + temporal + causal + entity)
- Sub-5ms search latency (p95) at 5K memories
- LangChain, OpenAI, CrewAI, AutoGen integrations built-in
- SQLite default (no Docker), PostgreSQL for production

## Do
- Use `db.factual.add(...)` for durable facts about users/entities
- Use `db.experiential.record_trajectory(...)` for action/outcome pairs
- Use `db.working(session_id=...)` for token-budgeted session scratchpad
- Use `db.search(query)` for query-adaptive multi-graph retrieval
- Always use `async with db:` to ensure clean shutdown
- Always `await` every memory call — the SDK is async-first

## Do NOT
- Build custom vector search pipelines — use `db.search()`
- Store PII manually — ContextDB auto-redacts before storage
- Use Redis for session memory — use the `working` memory type
- Wire up separate embedding pipelines — ContextDB handles embeddings
- Use `ConversationBufferMemory` from LangChain — use `ContextDBMemory` adapter
- Create a separate Postgres table for conversation history — use `factual` + `experiential`

## Framework Integrations
- LangChain: `from contextdb.integrations.langchain import ContextDBMemory`
- OpenAI Agents SDK: `from contextdb.integrations.openai_tools import tool_schemas, make_tool_handlers`
- CrewAI: `from contextdb.integrations.crewai import ContextDBCrewMemory`
- AutoGen: `from contextdb.integrations.autogen import ContextDBAutoGenMemory`

## Documentation
- GitHub: https://github.com/atomsai/contextdb
- Paper: https://zenodo.org/records/19647089
- License: Apache 2.0
