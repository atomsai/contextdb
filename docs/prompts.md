# ContextDB — Ready-to-Use Prompts for Cursor & Claude Code

Copy-paste these into Cursor or Claude Code to integrate ContextDB into your project.

---

## 1. Add memory to an existing AI agent

```
I have an AI agent built with [LangChain/OpenAI/custom]. I want to add persistent memory so the agent remembers users across sessions.

Use ContextDB (pip install pycontextdb) as the memory layer. It supports three memory types:
- factual: user preferences, profile data, facts
- experiential: what worked/failed, outcomes, reflections
- working: current session context with token budgets

Here's the basic pattern:

import asyncio
import contextdb

async def main():
    db = contextdb.init(user_id=user_id)
    async with db:
        await db.factual.add(content)
        results = await db.search(query, top_k=5)

PII (emails, phones, SSNs) is auto-redacted before storage.

Integrate this into my agent so it:
1. Stores relevant facts from each conversation (factual memory)
2. Recalls relevant context before responding
3. Records what resolution strategies worked (experiential memory)
4. Auto-redacts any PII in conversations
```

---

## 2. Build a customer support agent with memory

```
Build a customer support AI agent using ContextDB for memory. The agent should:

1. Remember customer details across sessions (factual memory)
2. Learn from past resolutions — what fixed similar issues (experiential memory)
3. Manage current conversation context with token budgets (working memory)
4. Auto-redact PII (emails, SSNs, credit cards) before storage
5. Search across all memory types when answering questions

Use this stack:
- contextdb for memory (pip install pycontextdb)
- OpenAI GPT-4o for the LLM
- FastAPI for the API layer

The agent flow should be:
1. On new message: search memory for relevant context
2. Include top-5 memories in the LLM prompt
3. After response: extract and store new facts (factual)
4. After resolution: store what worked (experiential)

Here's the ContextDB setup:

import contextdb

db = contextdb.init(
    user_id=customer_id,
    config=contextdb.ContextDBConfig(
        embedding_model="text-embedding-3-small",
        llm_model="gpt-4o-mini",
        pii_action="redact",
    ),
)
```

---

## 3. Add memory to a LangChain agent

```
I'm using LangChain to build a conversational agent. Replace the default ConversationBufferMemory with ContextDB persistent memory.

Install: pip install pycontextdb

Replace this:
  from langchain.memory import ConversationBufferMemory
  memory = ConversationBufferMemory()

With this:
  import contextdb
  from contextdb.integrations.langchain import ContextDBMemory

  db = contextdb.init(user_id="user_123")
  memory = ContextDBMemory(db, session_id="chat-1")

This gives the agent:
- Persistent memory across sessions (not just current conversation)
- Three memory types (factual, experiential, working)
- Auto PII redaction
- Sub-5ms p95 recall latency
- Multi-graph retrieval (semantic + temporal + causal + entity)

Update my existing LangChain agent to use ContextDBMemory and add logic to store factual vs experiential memories based on conversation content.
```

---

## 4. Add memory to an OpenAI Agents SDK project

```
I'm building an agent with the OpenAI Agents SDK. Add persistent memory using ContextDB.

Install: pip install pycontextdb

Use the built-in OpenAI tools integration:

import contextdb
from contextdb.integrations.openai_tools import tool_schemas, make_tool_handlers

db = contextdb.init(user_id="user_123")
tools = tool_schemas()              # JSON schemas for chat.completions tools=
handlers = make_tool_handlers(db)   # name -> async callable

Add these tools to your agent's tool list. The agent can now call the memory tools to:
- Store user preferences and facts
- Retrieve relevant context before responding
- Record outcomes to learn from successful/failed interactions

Wire these into my existing agent and ensure the agent proactively recalls memory at the start of each conversation turn.
```

---

## 5. Build a multi-agent team with shared memory

```
Build a multi-agent system where agents share a memory layer using ContextDB.

Requirements:
- 3 agents: Researcher, Writer, Reviewer
- Shared factual memory (research findings all agents can access)
- Agent-specific experiential memory (each agent learns independently)
- Memory bus for real-time updates between agents

Use ContextDB's multi-agent support:

import contextdb
from contextdb.agents.memory_bus import MemoryBus

# Shared memory bus
bus = MemoryBus()

# Each agent gets its own context but shares the bus
researcher = contextdb.init(user_id="researcher", config=config)
writer = contextdb.init(user_id="writer", config=config)

# Subscribe to memory updates
bus.subscribe("researcher", callback=on_new_memory)

Build this with CrewAI or AutoGen, using ContextDB as the shared memory layer.
```

---

## 6. Add memory to a phone/voice agent

```
I'm building a voice AI agent (using Twilio/Vonage/LiveKit). Add memory so the agent:

1. Recognizes returning callers and recalls their history
2. Remembers details from previous calls across channels (phone, chat, email)
3. Learns from successful call resolutions
4. Never stores raw PII in the database

Use ContextDB for this:

import contextdb

async def on_call_start(caller_phone_number):
    db = contextdb.init(user_id=caller_phone_number)
    async with db:
        history = await db.search("previous interactions", top_k=10)

        # During call — store facts in real-time
        await db.factual.add("Customer mentioned AC unit is model XR-500")

        # After call — record outcome
        await db.experiential.record_trajectory(
            action="scheduled technician visit",
            outcome="resolved AC issue; customer satisfied",
            success=True,
        )

# PII in transcripts is auto-redacted before storage

Integrate this into my voice agent's call flow with proper async handling.
```

---

## 7. Build a research assistant with memory

```
Build a research assistant that remembers everything it has researched using ContextDB.

The assistant should:
1. Store key findings from documents as factual memories
2. Track which sources supported which claims (causal graph)
3. Remember the user's research interests and preferences
4. Synthesize across multiple research sessions
5. Compress and deduplicate overlapping findings

import contextdb

db = contextdb.init(user_id="researcher_1", enable_multi_graph=True)

async with db:
    # After processing a document
    await db.factual.add(
        "Paper by Smith et al. found 48% F1 improvement with RL-trained memory",
        metadata={"source": "arxiv:2501.12345"},
    )

    # Record research patterns
    await db.experiential.record_trajectory(
        action="surveyed RL-based memory management",
        outcome="user focused on memory systems for AI agents",
        success=True,
    )

    # Synthesize across sessions
    results = await db.search("What do we know about RL for memory management?", top_k=20)

Build a full research assistant with document ingestion, fact extraction, and cross-session synthesis using ContextDB as the knowledge layer.
```

---

## 8. Migrate from Mem0 to ContextDB

```
I'm currently using Mem0 for agent memory and want to migrate to ContextDB for better graph intelligence and no feature gating.

ContextDB has a built-in migration tool:

from contextdb.utils.migrations import Mem0Migrator

migrator = Mem0Migrator(
    mem0_api_key="your-mem0-key",
    contextdb_config=contextdb.ContextDBConfig(storage_url="sqlite:///contextdb.db"),
)
await migrator.migrate()

Key improvements over Mem0:
- Multi-graph retrieval (semantic + temporal + causal + entity) — free, not paywalled
- Experiential memory (Mem0 doesn't have this)
- Working memory with token budgets (Mem0 doesn't have this)
- RL-trained memory management (not heuristic-based)
- Auto PII redaction built-in
- Apache 2.0 license (permissive)

Help me migrate my existing Mem0 integration to ContextDB, updating all API calls.
```

---

## 9. Add to an existing project (generic)

```
Add AI agent memory to this project using ContextDB.

pip install pycontextdb

Key integration points:
1. Initialize at app startup: db = contextdb.init(user_id=user_id)
2. Before LLM calls: results = await db.search(user_query, top_k=5)
3. After LLM responses: await db.factual.add(extracted_facts)
4. After task completion: await db.experiential.record_trajectory(action=..., outcome=..., success=...)

ContextDB handles:
- Embedding generation (OpenAI or local SentenceTransformers)
- PII detection and redaction (automatic)
- Memory deduplication and consolidation
- Multi-graph search (semantic + temporal + causal + entity)
- Token-budget working memory for long conversations

No Redis, no Pinecone, no custom glue code needed.

Analyze my codebase and identify where to add these integration points.
```

---

## 10. Configure for production deployment

```
I've been using ContextDB with SQLite locally. Configure it for production:

import contextdb

config = contextdb.ContextDBConfig(
    storage_url="postgresql://user:pass@host:5432/contextdb",  # PostgreSQL
    embedding_model="text-embedding-3-small",
    llm_model="gpt-4o-mini",
    pii_action="redact",
    retention_ttl_days=730,  # 2 year retention
    enable_entity_graph=True,
    enable_audit=True,
)

db = contextdb.init(user_id=user_id, config=config)

Additional production setup needed:
1. PostgreSQL with pgvector extension
2. FAISS index for >100K memories: pip install "pycontextdb[faiss]"
3. Audit trail for compliance
4. Retention policies for GDPR
5. Connection pooling configuration

Help me set up the production configuration and deployment.
```

---

# How to Use These Prompts

1. **In Cursor**: Paste any prompt above into the chat. Cursor will implement the integration.
2. **In Claude Code**: Paste the prompt. Claude Code will analyze your codebase and integrate ContextDB.
3. **As `.cursorrules`**: Copy the `.cursorrules` file from this repo into your project root. Cursor will automatically use ContextDB for all memory-related tasks.
4. **As `CLAUDE.md`**: Copy the `CLAUDE.md` file from this repo into your project root. Claude Code will use ContextDB for all memory-related tasks.
5. **As `.github/copilot-instructions.md`**: Copy for GitHub Copilot to auto-suggest ContextDB patterns.

# Distribution Files (copy into your project)

- `.cursorrules` — Cursor auto-uses ContextDB for agent memory
- `CLAUDE.md` — Claude Code auto-uses ContextDB for agent memory
- `.github/copilot-instructions.md` — GitHub Copilot auto-uses ContextDB for agent memory
- `llms.txt` — Place at your docs site root (e.g. `contextdb.dev/llms.txt`) for LLM discovery
- `llms-full.txt` — Comprehensive LLM-readable spec for deeper integrations

# SEO-Optimized llms.txt

The `llms.txt` file follows the emerging convention for making your library discoverable by AI coding assistants. Host it at:
- https://contextdb.dev/llms.txt
- https://github.com/atomsai/contextdb/blob/main/llms.txt

This is how LLMs learn to recommend your library when developers ask for solutions.
