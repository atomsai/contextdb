# ContextDB — Claude Code Instructions

## Project Vision
ContextDB is the **unified context layer for AI agents** — replacing the patchwork of Pinecone + Redis + Postgres + glue code with one system that understands memory. Think "Supabase for agent memory."

**Positioning:** Databricks Lakebase gives agents a hard drive. ContextDB gives agents a brain.

## Architecture Overview
ContextDB is a Python library (`pip install contextdb`) with 5 layers:

1. **Storage Layer** — SQLite (default), PostgreSQL (production), FAISS vectors
2. **Memory Layer** — Factual, Experiential, Working memory types
3. **Graph Layer** — 4 orthogonal graphs: semantic, temporal, causal, entity
4. **Dynamics Layer** — Formation (segmentation + extraction + compression), Evolution (linking + consolidation + pruning), Retrieval (query-adaptive multi-graph fusion)
5. **Privacy Layer** — PII detection, configurable retention, audit trails

## Tech Stack
- Python 3.10+, async-first (asyncio/aiosqlite/asyncpg)
- Pydantic v2 for all data models
- FAISS for vector search (with NumpyIndex fallback)
- OpenAI API for embeddings + LLM (with local model support)
- PyTorch + TRL for RL memory manager (Phase 4)
- hatch as build backend
- pytest + pytest-asyncio for tests
- ruff for linting, mypy (strict) for type checking

## Project Structure
```
contextdb/
├── contextdb/
│   ├── __init__.py           # Exports: init, __version__
│   ├── client.py             # Main ContextDB class
│   ├── py.typed              # PEP 561 marker
│   ├── core/
│   │   ├── config.py         # ContextDBConfig (Pydantic BaseSettings)
│   │   ├── exceptions.py     # ContextDBError hierarchy
│   │   └── models.py         # MemoryItem, Edge, Entity, RetentionPolicy
│   ├── store/
│   │   ├── base.py           # BaseStore ABC
│   │   ├── sqlite_store.py   # SQLiteStore (default)
│   │   └── vector_index.py   # FAISSIndex, NumpyIndex
│   ├── graphs/
│   │   ├── base.py           # BaseGraph ABC
│   │   ├── semantic.py       # SemanticGraph (embedding similarity)
│   │   ├── temporal.py       # TemporalGraph (time-based)
│   │   ├── causal.py         # CausalGraph (LLM-inferred cause/effect)
│   │   └── entity.py         # EntityGraph (NER + coreference)
│   ├── dynamics/
│   │   ├── formation.py      # Segmenter + Extractor + Compressor
│   │   ├── evolution.py      # AutoLinker + Consolidator + Pruner
│   │   └── retrieval.py      # QueryClassifier + RetrievalFuser + RetrievalEngine
│   ├── memory/
│   │   ├── factual.py        # FactualMemory API
│   │   ├── experiential.py   # ExperientialMemory (trajectories, reflections)
│   │   └── working.py        # WorkingMemory (token-budget paging)
│   ├── privacy/
│   │   ├── pii_detector.py   # Regex-based PII detection + redaction
│   │   ├── retention.py      # RetentionPolicyEnforcer
│   │   └── audit.py          # Hash-chained audit trail
│   ├── agents/
│   │   ├── memory_bus.py     # Multi-agent pub/sub memory sharing
│   │   └── rl_manager.py     # RL-trained ADD/UPDATE/DELETE/NOOP
│   ├── integrations/
│   │   ├── langchain.py      # LangChain memory adapter
│   │   ├── openai_tools.py   # OpenAI function calling tools
│   │   ├── crewai.py         # CrewAI memory adapter
│   │   └── autogen.py        # AutoGen memory adapter
│   └── utils/
│       ├── embeddings.py     # EmbeddingProvider (OpenAI, SentenceTransformer, Mock)
│       └── llm.py            # LLMProvider (OpenAI, Mock)
├── tests/
│   ├── conftest.py           # Shared fixtures (mock_embedder, mock_llm, tmp_store)
│   ├── unit/                 # Fast tests, no API calls
│   └── integration/          # Tests with real storage
├── examples/
│   ├── customer_support.py
│   ├── research_assistant.py
│   ├── phone_agent.py
│   └── multi_agent_team.py
├── docs/
├── pyproject.toml
├── LICENSE                   # Apache 2.0
└── README.md
```

## Code Style & Conventions
- All public functions/methods have docstrings (Google style)
- All modules use `from __future__ import annotations`
- Type hints on everything — mypy strict must pass
- Async-first: all I/O methods are async
- Line length: 100 characters (ruff)
- Use `str` Enums for serialization (e.g., `class MemoryType(str, Enum)`)
- All datetime values in UTC, ISO format for storage
- Parameterized SQL queries only — never string formatting
- Tests use `pytest-asyncio` with `@pytest.mark.asyncio`
- Use `MockEmbedding` and `MockLLM` for tests — never require API keys

## Build Phases
See `TASKS.md` for the full 32-task build plan. Execute tasks in order:

- **Phase 1 (Tasks 1-8):** Foundation — scaffold, models, SQLite store, embeddings, FAISS, LLM, PII, client
- **Phase 2 (Tasks 9-16):** Intelligence — 4 graphs, multi-graph retrieval, formation pipeline, evolution engine
- **Phase 3 (Tasks 17-20):** Memory APIs — factual, experiential, working memory, framework integrations
- **Phase 4 (Tasks 21-26):** Production — multi-agent, audit trails, retention, RL manager, migration tools, CLI
- **Phase 5 (Tasks 27-32):** Polish — example apps, docs, README, PyPI release

## Key Design Decisions
1. **SQLite default, Postgres production** — No Docker needed for development
2. **FAISS with NumpyIndex fallback** — Works without FAISS installed for <10K memories
3. **LLM-agnostic** — OpenAI default, but any provider via ABC
4. **RL as enhancement, not requirement** — System works without RL; RL makes it better
5. **Privacy at memory layer, not application layer** — PII never reaches storage unprocessed
6. **Open-core model** — Free: single-graph, all memory types, basic PII, SDK. Paid: multi-graph, RL, managed cloud, enterprise compliance

## Critical Quality Gates
Before marking any task complete:
1. `ruff check .` passes
2. `mypy . --strict` passes (or `--strict` on changed files)
3. `pytest` passes with >90% coverage on new code
4. No hardcoded API keys or secrets
5. All new public APIs have docstrings
