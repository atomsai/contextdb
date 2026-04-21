# ContextDB — Product Requirements Document (PRD)

**Version:** 2.0
**Author:** Gaurav Sharma (gaurav@saaslabs.co)
**Date:** April 2, 2026
**Status:** Draft
**License:** Apache 2.0 (open-source core)

---

## 1. Executive Summary

ContextDB is the **unified context layer for AI agents** — an open-source system that replaces the fragile patchwork of vector databases, session stores, SQL databases, and custom glue code that every team building AI agents assembles today.

Where Databricks Lakebase gives agents a hard drive, ContextDB gives agents a brain. Where Mem0 offers a key-value memory API, ContextDB offers a complete memory operating system with graph intelligence, learned memory policies, and privacy by design.

The core library is free and open-source. Advanced features (multi-graph intelligence, RL-trained memory management, managed cloud, enterprise compliance) are gated behind a paid tier — invite-only at launch.

**One-liner:** ContextDB replaces your agent's patchwork of Pinecone + Redis + Postgres + glue code with one system that understands memory.

**Positioning:** ContextDB is to AI agent memory what Snowflake was to data warehousing and Supabase was to the backend — the unified layer that replaces the duct-tape architecture.

---

## 2. Problem Statement

### 2.1 The Patchwork Problem
Every team building production AI agents assembles the same fragile architecture:

| What the agent needs | What teams use today | What breaks |
|---------------------|---------------------|-------------|
| Semantic recall | Pinecone / Qdrant / Weaviate | Semantic-only; no temporal, causal, or entity awareness |
| Session state | Redis / Memcached | Ephemeral; lost between sessions; no compression |
| User profiles | PostgreSQL / MongoDB | Static rows; no memory lifecycle; no graph links |
| Conversation logs | S3 / flat files | Write-only archive; not queryable by meaning or time |
| Cross-service linking | Custom glue code | Brittle; no consistency; duplicated across every team |
| Learning from outcomes | *(nothing)* | Agents never learn what worked or failed |
| PII protection | *(nothing)* | Raw PII stored indefinitely; compliance risk |

This patchwork is rebuilt by every organization independently. It costs 2-4 engineering months to build, breaks at every seam, and never improves because there's no memory lifecycle — just storage.

### 2.2 Why Incumbents Are Approaching This Wrong

**Databricks Lakebase** launched a managed PostgreSQL with pgvector and LangGraph checkpointing for agent state. **Snowflake Project SnowWork** offers autonomous agents backed by Snowflake's data cloud. Both approach agent memory as a *storage problem* — "here's a database where agents can persist state."

But agent memory is a *cognitive problem*. A database stores conversation turns as rows. A memory system understands that the person emailing today is the same person who called yesterday, knows the issue was caused by a proration bug, remembers that the last three customers with this problem were resolved by clearing the cache, and redacts the credit card number before storing anything. The gap between "storage for agents" and "memory for agents" is the entire ContextDB value proposition.

### 2.3 Why Existing Memory Solutions Fall Short

| System | What it does well | What it lacks |
|--------|------------------|---------------|
| **Mem0** | Simple key-value memory, production-ready SDK | No graph intelligence (free tier), no experiential memory, no RL, no working memory |
| **Zep** | Temporal knowledge graphs, bitemporal tracking | No experiential memory, no working memory management, no RL-trained policies |
| **MemGPT/Letta** | OS-inspired working memory paging | No persistent factual memory, no graph structures, no multi-agent |
| **Databricks Lakebase** | Managed Postgres, pgvector, LangGraph checkpointing | No memory intelligence — just storage. No formation, evolution, or learned policies |
| **Snowflake SnowWork** | Enterprise data platform, agent framework | Analytics-first; not purpose-built for memory; vendor lock-in |
| **LangChain Memory** | Easy to integrate | Toy-level: flat chat history, no structure, no evolution |
| **Custom patchwork** | Flexible | Every team rebuilds the same plumbing; 2-4 months; breaks at every seam |

### 2.4 The Opportunity
The AI agents market is projected to grow from $7.6B (2025) to $183B by 2033. Every one of those agents needs memory. No system today covers the full taxonomy: Forms (token, parametric, latent) × Functions (factual, experiential, working) × Dynamics (formation, evolution, retrieval). No system replaces the patchwork with a unified layer. ContextDB does.

**The Supabase parallel:** Supabase became a $5B company by unifying the backend patchwork (auth + database + storage + real-time) into one open-source platform. ContextDB unifies the agent memory patchwork (vectors + sessions + profiles + logs + glue) into one open-source platform. The pattern is identical.

---

## 3. Target Users

### 3.1 Primary: AI Engineers & Agent Developers
- Building production AI agents (chatbots, copilots, phone agents, research assistants)
- Currently using LangChain/LlamaIndex/custom RAG and hitting memory limitations
- Need: `pip install contextdb` → 3 lines of code → persistent memory

### 3.2 Secondary: AI-Native SaaS Companies
- Companies embedding AI into their products (customer support, sales, HR)
- Need: managed memory infrastructure with compliance guarantees
- Examples: companies like SaaSLabs (Helpwise, JustCall, ServiceAgent.ai)

### 3.3 Tertiary: Enterprise AI Teams
- Large organizations deploying multi-agent systems
- Need: on-prem deployment, SOC2/HIPAA compliance, RBAC, audit trails

---

## 4. Product Architecture

### 4.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Application Layer                      │
│  (Your AI Agent / Chatbot / Phone Agent / Copilot)       │
└──────────────────────┬──────────────────────────────────┘
                       │ contextdb SDK
┌──────────────────────▼──────────────────────────────────┐
│                   ContextDB Core                         │
│  ┌──────────┐  ┌──────────┐  ┌───────────┐             │
│  │ Memory   │  │ Dynamics │  │ Privacy   │             │
│  │ Store    │  │ Engine   │  │ Layer     │             │
│  │          │  │          │  │           │             │
│  │ •Token   │  │ •Form    │  │ •PII Det  │             │
│  │ •Param   │  │ •Evolve  │  │ •Retention│             │
│  │ •Latent  │  │ •Retrieve│  │ •Audit    │             │
│  └──────────┘  └──────────┘  └───────────┘             │
│  ┌──────────┐  ┌──────────┐                             │
│  │ Function │  │ Multi-   │                             │
│  │ Mapping  │  │ Agent    │                             │
│  │          │  │ Layer    │                             │
│  │ •Factual │  │ •Bus     │                             │
│  │ •Exper.  │  │ •Routing │                             │
│  │ •Working │  │ •Conflict│                             │
│  └──────────┘  └──────────┘                             │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│                 Storage Backends                         │
│  SQLite (default) │ PostgreSQL │ Neo4j │ Redis │ S3     │
└─────────────────────────────────────────────────────────┘
```

### 4.2 Core Modules (in build order)

#### Module 1: Memory Store (`contextdb.store`)
**What it does:** Stores and indexes memory items across multiple representations.

**Components:**
- `MemoryItem` — Core data model: content, embedding, metadata, timestamps, graph positions
- `TokenStore` — Primary store for explicit text memories
  - SQLite backend (default, zero-config)
  - PostgreSQL backend (production)
  - Vector index via FAISS/Qdrant for embeddings
- `GraphStore` — Multi-graph representation engine
  - `SemanticGraph` — Embedding similarity edges (threshold-based)
  - `TemporalGraph` — Time-ordered edges with proximity weights
  - `CausalGraph` — LLM-inferred causal/logical dependency edges
  - `EntityGraph` — Named entity nodes + entity-memory association edges
- `ParametricStore` — Optional LoRA adapter management (paid tier)
- `LatentStore` — KV-cache state management (paid tier)

**Data Model:**
```python
@dataclass
class MemoryItem:
    id: str                          # UUID
    content: str                     # Natural language content
    embedding: np.ndarray            # Dense vector (configurable dim)
    memory_type: MemoryType          # FACTUAL | EXPERIENTIAL | WORKING
    source: str                      # Origin identifier
    metadata: dict                   # Arbitrary key-value pairs

    # Bitemporal timestamps
    event_time: datetime             # When the fact/event occurred
    ingestion_time: datetime         # When the system observed it

    # Graph positions (populated by GraphStore)
    semantic_edges: List[Edge]
    temporal_edges: List[Edge]
    causal_edges: List[Edge]
    entity_mentions: List[Entity]

    # Privacy
    pii_detected: List[PIIAnnotation]
    retention_policy: RetentionPolicy

    # Lifecycle
    created_at: datetime
    updated_at: datetime
    access_count: int
    last_accessed: datetime
    confidence: float                # 0.0 - 1.0
    status: Status                   # ACTIVE | ARCHIVED | DELETED
```

**Key Design Decisions:**
- SQLite as default backend (zero-config, single-file, surprisingly fast for <1M memories)
- Embedding model is pluggable: default to `text-embedding-3-small`, support any SentenceTransformer
- Graph edges are stored in the same DB as memories (no separate graph DB required for free tier)
- Neo4j integration available as optional backend (paid tier optimization)

---

#### Module 2: Dynamics Engine (`contextdb.dynamics`)
**What it does:** Handles the lifecycle of memories — creation, evolution, and retrieval.

**Components:**

##### 2a. Formation Pipeline (`contextdb.dynamics.formation`)
```
Raw Input → Segmenter → Extractor → Compressor → MemoryItem
```

- `Segmenter` — Splits raw input into topically coherent segments
  - Default: LLM-based topic segmentation (works with any model)
  - Fast mode: sentence-level sliding window with embedding similarity drop detection
  - Configurable: turn-level, session-level, or custom boundaries

- `Extractor` — Extracts structured information from each segment
  - Factual extractor: entities, relationships, facts, states
  - Experiential extractor: action-outcome pairs, workflows, reflections
  - Working memory extractor: key claims, open questions, active goals
  - All extractors use a single LLM call with structured output (JSON mode)

- `Compressor` — Compression-as-denoising
  - Default: LLM-based summarization with configurable compression ratio
  - Fast mode: extractive compression (key sentence selection)
  - Domain-adaptive: auto-tunes compression ratio per domain via validation set
  - Target: 60-80% compression with <2% retrieval precision loss

##### 2b. Evolution Engine (`contextdb.dynamics.evolution`)

- `AutoLinker` — When new memory is added, discovers connections across all 4 graphs
  - Semantic: cosine similarity > threshold → edge
  - Temporal: timestamp proximity + sequence detection → edge
  - Causal: LLM inference for causal/logical deps → edge (batch, async)
  - Entity: shared entity mention → edge

- `Consolidator` — Periodic background process
  - Clusters semantically similar memories
  - Creates hierarchical summaries (GraphRAG-style community detection)
  - Merges near-duplicate memories
  - Runs on configurable schedule (default: every 100 new memories or daily)

- `Pruner` — Removes low-value memories
  - Heuristic mode (free): access count, recency, confidence decay
  - RL mode (paid): trained policy decides DELETE/KEEP/MERGE

##### 2c. Retrieval Engine (`contextdb.dynamics.retrieval`)

- `QueryClassifier` — Determines query type and graph weights
  - Input: query string + optional context
  - Output: `{semantic: 0.3, temporal: 0.5, causal: 0.1, entity: 0.1}`
  - Implementation: lightweight fine-tuned classifier or LLM-based

- `GraphTraverser` — Traverses each graph independently
  - Semantic: embedding similarity top-k
  - Temporal: recency-weighted neighborhood expansion
  - Causal: dependency chain following (forward/backward)
  - Entity: Personalized PageRank from query entities

- `Fuser` — Combines results from all graphs
  - Weighted reciprocal rank fusion
  - Deduplication across graphs
  - Optional LLM re-ranking for top candidates

- `ContextAssembler` — Formats retrieved memories for LLM consumption
  - Chronological ordering
  - Relevance-first ordering
  - Hierarchical (summary + details on demand)
  - Token-budget-aware truncation

---

#### Module 3: Function Mapping (`contextdb.functions`)
**What it does:** Provides high-level APIs organized by memory function.

- `FactualMemory` — Declarative knowledge store
  ```python
  factual = contextdb.factual(user_id="user_123")
  factual.remember("Alex prefers email over phone")
  factual.recall("How does Alex prefer to be contacted?")
  factual.update("Alex", {"preferred_contact": "email"})
  factual.forget(entity="Alex", fact_type="contact_preference")
  ```

- `ExperientialMemory` — Procedural knowledge store
  ```python
  exp = contextdb.experiential(agent_id="support_agent")
  exp.record_trajectory(steps=[...], outcome="resolved", score=0.9)
  exp.record_reflection("Billing sync errors: clearing cache first saves 5 min")
  exp.recall_workflow("customer reports billing sync error")
  exp.get_similar_experiences(situation="user can't login")
  ```

- `WorkingMemory` — Active context manager
  ```python
  wm = contextdb.working(session_id="call_456")
  wm.push("Customer mentioned AC noise issue")       # Add to active context
  wm.compress(target_tokens=2000)                      # Fit within budget
  wm.get_context(max_tokens=4000)                      # Retrieve active context
  wm.page_out(strategy="least_recent")                 # Move to long-term
  ```

---

#### Module 4: Privacy Layer (`contextdb.privacy`)
**What it does:** Ensures all memory operations comply with privacy policies.

**Components:**

- `PIIDetector` — Identifies PII before storage
  - Built-in NER for: names, emails, phone numbers, addresses, SSNs, credit cards
  - Configurable: add custom PII patterns (regex or model-based)
  - Actions: REDACT (replace with placeholder), ENCRYPT (store encrypted), FLAG (mark for review), ALLOW (pass through)
  - Default: REDACT for financial identifiers, ENCRYPT for names/emails

- `RetentionManager` — Enforces data lifecycle policies
  ```python
  retention = RetentionPolicy(
      default_ttl=timedelta(days=730),          # 2 years default
      factual_ttl=timedelta(days=365*5),        # 5 years for facts
      experiential_ttl=None,                     # Never expire workflows
      working_ttl=timedelta(hours=24),           # 24h for working memory
      right_to_erasure=True,                     # GDPR compliance
  )
  ```

- `AuditLogger` — Append-only operation log
  - Every CREATE, READ, UPDATE, DELETE logged
  - Includes: agent identity, user identity, timestamp, operation type, memory ID
  - Export formats: JSON, CSV, SIEM-compatible
  - Tamper-resistant: hash-chained log entries

- `AccessControl` — Role-based memory access (paid tier)
  - Roles: OWNER, ADMIN, AGENT, VIEWER
  - Scopes: per-user, per-team, per-organization
  - Cross-tenant isolation in multi-tenant deployments

---

#### Module 5: Multi-Agent Layer (`contextdb.multiagent`)
**What it does:** Enables multiple agents to share, coordinate, and avoid conflicting memories.

- `MemoryBus` — Pub/sub event system
  - Events: MEMORY_CREATED, MEMORY_UPDATED, MEMORY_CONFLICT, WORKFLOW_DISCOVERED
  - Agents subscribe to relevant event types based on role
  - In-process (default) or Redis-backed (production)

- `RoleRouter` — Routes memories to relevant agents
  - Each agent has a role profile (e.g., "billing specialist", "tech support")
  - Router selectively feeds role-relevant memories
  - Prevents information overload in multi-agent systems

- `ConflictResolver` — Handles contradicting memories
  - Detection: entity graph analysis for conflicting attribute values
  - Resolution strategies: most-recent-event-time wins, confidence-weighted merge, escalate to orchestrator
  - Audit trail for all conflict resolutions

---

#### Module 6: RL Memory Manager (`contextdb.rl`) — PAID TIER
**What it does:** Learns optimal memory operations from downstream task performance.

- `MemoryManager` — RL-trained agent that decides ADD/UPDATE/DELETE/NOOP
  - Pre-trained checkpoint included (trained on general conversation data)
  - Fine-tuning pipeline for domain-specific training
  - Training requires: 150+ examples with downstream quality labels
  - Algorithms: PPO (default), GRPO (for ranking-based rewards)

- `RetrievalPolicy` — Learned query-to-graph routing
  - Replaces the heuristic QueryClassifier with a trained policy
  - Learns which graph combinations work best for which query patterns

- `TrainingPipeline` — End-to-end RL training
  - Data collection: automatic logging of memory operations + downstream outcomes
  - Reward computation: F1, BLEU, LLM-as-Judge, or custom reward function
  - Training loop: PPO/GRPO with configurable hyperparameters
  - Evaluation: automatic benchmark suite

---

## 5. SDK Design

### 5.1 Installation
```bash
# Open source core
pip install contextdb

# With optional backends
pip install contextdb[postgres]
pip install contextdb[neo4j]
pip install contextdb[redis]

# Paid tier (invite-only)
pip install contextdb[pro]
```

### 5.2 Quick Start (3 lines)
```python
import contextdb

db = contextdb.init(user_id="user_123")
db.add("Alex prefers morning calls and uses a Carrier AC unit")
result = db.search("What kind of AC does Alex have?")
# → "Alex uses a Carrier AC unit"
```

### 5.3 Full API Surface

```python
import contextdb

# Initialize with configuration
db = contextdb.init(
    storage="sqlite:///memory.db",          # or "postgresql://...", "neo4j://..."
    embedding_model="text-embedding-3-small",
    llm_model="gpt-4o-mini",                # for extraction, compression, linking
    privacy=contextdb.PrivacyConfig(
        pii_action="redact",
        retention_ttl_days=730,
    ),
)

# --- Core Operations ---
db.add(content, memory_type=None, metadata=None, event_time=None)
db.search(query, top_k=10, memory_type=None, time_range=None)
db.update(memory_id, new_content=None, new_metadata=None)
db.delete(memory_id)
db.forget(user_id=None, entity=None, older_than=None)

# --- Function-Specific APIs ---
db.factual.remember(content, entity=None)
db.factual.recall(query, entity=None)
db.factual.get_entity(entity_name)
db.factual.list_entities()

db.experiential.record_trajectory(steps, outcome, score)
db.experiential.record_reflection(content, context)
db.experiential.recall_workflow(situation)
db.experiential.get_similar_experiences(situation, top_k=5)

db.working.push(content)
db.working.get_context(max_tokens=4000)
db.working.compress(target_tokens=2000)
db.working.page_out(strategy="least_recent")
db.working.clear()

# --- Graph Operations ---
db.graph.get_neighbors(memory_id, graph_type="semantic", depth=2)
db.graph.get_entity_graph(entity_name)
db.graph.get_timeline(entity=None, start=None, end=None)
db.graph.get_causal_chain(memory_id, direction="forward")
db.graph.communities(level=1)  # GraphRAG-style hierarchy

# --- Multi-Agent ---
db.agents.register(agent_id, role, permissions)
db.agents.share(memory_id, target_agent_id)
db.agents.broadcast(event_type, payload)
db.agents.get_conflicts(entity=None)

# --- Privacy ---
db.privacy.detect_pii(text)
db.privacy.erase_user(user_id)
db.privacy.export_audit_log(format="json", since=None)
db.privacy.set_retention(memory_type, ttl_days)

# --- Admin ---
db.admin.stats()                # Memory count, storage size, graph stats
db.admin.consolidate()          # Trigger manual consolidation
db.admin.export(format="json")  # Full memory export
db.admin.import_from(source)    # Import from Mem0, Zep, or JSON
```

### 5.4 Framework Integrations

```python
# LangChain
from contextdb.integrations import LangChainMemory
memory = LangChainMemory(contextdb_instance=db)
chain = ConversationChain(memory=memory)

# LlamaIndex
from contextdb.integrations import LlamaIndexMemory
memory = LlamaIndexMemory(contextdb_instance=db)

# CrewAI
from contextdb.integrations import CrewAIMemory
crew = Crew(memory=CrewAIMemory(db))

# OpenAI Agents SDK
from contextdb.integrations import OpenAIAgentMemory

# AutoGen
from contextdb.integrations import AutoGenMemory
```

---

## 6. Implementation Phases

### Phase 1: Foundation (Weeks 1-4) — OPEN SOURCE
**Goal:** A working `pip install contextdb` with basic memory that's better than LangChain.

| Week | Deliverable | Details |
|------|------------|---------|
| 1 | Project scaffold + MemoryItem + SQLite store | Repo setup, CI/CD, data model, basic CRUD |
| 2 | Embedding index + basic retrieval | FAISS integration, cosine similarity search, top-k retrieval |
| 3 | Formation pipeline (segmenter + extractor) | LLM-based extraction, turn-level segmentation |
| 4 | Basic privacy (PII detection + redaction) + SDK polish | NER-based PII, `contextdb.init()` API, README, docs |

**Exit criteria:** `pip install contextdb` works. 3-line quick start demo works. Basic add/search/update/delete with PII redaction.

### Phase 2: Graph Intelligence (Weeks 5-8) — OPEN SOURCE (single graph) + PAID (multi-graph)
**Goal:** Graph-based memory that's demonstrably better than flat retrieval.

| Week | Deliverable | Details |
|------|------------|---------|
| 5 | Semantic graph + entity graph | Embedding similarity edges, entity extraction + linking |
| 6 | Temporal graph + causal graph | Timestamp-based edges, LLM-inferred causal links |
| 7 | Multi-graph retrieval + query classifier | Graph traversal, weighted fusion, query routing |
| 8 | Consolidation + auto-linking | Community detection, hierarchical summaries, background linker |

**Free tier:** Single graph (semantic OR entity — user chooses).
**Paid tier:** All 4 graphs + multi-graph fusion + consolidation.

### Phase 3: Memory Functions (Weeks 9-12) — OPEN SOURCE
**Goal:** High-level APIs for factual, experiential, and working memory.

| Week | Deliverable | Details |
|------|------------|---------|
| 9 | Factual memory API | Entity-centric recall, bitemporal tracking, profile aggregation |
| 10 | Experiential memory API | Trajectory recording, workflow induction, reflection storage |
| 11 | Working memory API | Context paging, compression, token-budget management |
| 12 | Function mapping + integration tests | Unified API surface, cross-function queries, benchmarks |

### Phase 4: Production Features (Weeks 13-16) — MOSTLY PAID
**Goal:** Production-grade features for real deployments.

| Week | Deliverable | Details |
|------|------------|---------|
| 13 | Multi-agent memory sharing | Memory bus, role router, conflict resolution |
| 14 | RL Memory Manager | Training pipeline, pre-trained checkpoint, fine-tuning API |
| 15 | Advanced privacy + compliance | RBAC, audit trails, retention automation, encryption at rest |
| 16 | Framework integrations + migration tools | LangChain, LlamaIndex, CrewAI adapters; Mem0/Zep importers |

**Free tier:** Basic multi-agent (shared memory pool).
**Paid tier:** RL manager, RBAC, audit trails, role-aware routing, migration tools.

### Phase 5: Cloud & Enterprise (Weeks 17-24) — PAID ONLY
**Goal:** Managed cloud service + enterprise features.

| Week | Deliverable | Details |
|------|------------|---------|
| 17-18 | Managed API service | Hosted ContextDB with REST API, dashboard, usage metering |
| 19-20 | Dashboard + observability | Memory explorer, graph visualizer, query analytics, health monitoring |
| 21-22 | Enterprise features | SSO/SAML, SOC2 controls, HIPAA mode, cross-region replication |
| 23-24 | Benchmarks + paper results | Run full benchmark suite, publish results, update arXiv paper |

---

## 7. Feature Gating (Open Source vs Paid)

### 7.1 Free / Open Source (Apache 2.0)
Everything a solo developer or startup needs:
- All 3 memory functions (factual, experiential, working)
- Single-graph memory (semantic OR entity — user's choice)
- SQLite + PostgreSQL backends
- Basic PII detection and redaction
- Retention policies
- Basic multi-agent (shared pool, no routing)
- LangChain/LlamaIndex integrations
- Compression (LLM-based)
- Full Python SDK
- Community support (GitHub Issues, Discord)

### 7.2 Cloud Pro ($49-499/month, invite-only at launch)
For teams building production AI products:
- Everything in Free, plus:
- **Multi-graph memory** (all 4 graphs + fusion retrieval)
- **RL-trained Memory Manager** (pre-trained + fine-tuning)
- **Hierarchical consolidation** (GraphRAG-style)
- **Memory dashboard** (explorer, visualizer, analytics)
- **Advanced compression** (domain-adaptive ratios)
- **Role-aware multi-agent routing**
- **Neo4j backend support**
- **Hosted REST API** (no infra management)
- **Email support + 99.9% SLA**

### 7.3 Enterprise ($75K-150K/year)
For large organizations with compliance requirements:
- Everything in Cloud Pro, plus:
- **On-premises deployment** (Docker, Kubernetes, air-gapped)
- **SOC2 / HIPAA compliance mode**
- **RBAC with SSO/SAML**
- **Full audit trail with SIEM integration**
- **Cross-region replication**
- **Encrypted memory at rest + in transit**
- **Dedicated support + custom SLA**
- **Custom RL training on your data**

### 7.4 Gating Rationale
| Feature | Why it's paid |
|---------|--------------|
| Multi-graph | 4x storage + compute; the key differentiator from competitors |
| RL Manager | GPU-intensive training; highest accuracy gains |
| Dashboard | Ongoing hosting cost; high enterprise demand |
| RBAC/Audit | Enterprise requirement; negligible OSS demand |
| Neo4j backend | Enterprise graph DB; licensing implications |
| Managed API | Infrastructure cost; convenience premium |

---

## 8. Technical Specifications

### 8.1 Performance Targets
| Metric | Target | Notes |
|--------|--------|-------|
| Add latency (p50) | <50ms | Single memory insertion |
| Add latency (p95) | <200ms | Including graph linking |
| Search latency (p50) | <30ms | Single-graph retrieval |
| Search latency (p95) | <100ms | Multi-graph fusion |
| Memory capacity | 1M+ items | Per user/tenant |
| Embedding dimensions | 256-3072 | Configurable |
| Compression ratio | 60-80% | Token savings vs raw |
| PII detection accuracy | >95% | For standard PII types |

### 8.2 Technology Stack
| Component | Technology | Reason |
|-----------|-----------|--------|
| Language | Python 3.10+ | AI ecosystem standard |
| Default DB | SQLite (via aiosqlite) | Zero-config, fast, portable |
| Production DB | PostgreSQL + pgvector | Battle-tested, vector support |
| Graph DB (optional) | Neo4j | Enterprise graph queries |
| Vector Index | FAISS (default), Qdrant (optional) | Fast similarity search |
| Embeddings | OpenAI, Sentence Transformers, Cohere | Pluggable |
| LLM | Any OpenAI-compatible API | Extraction, compression, linking |
| Cache | Redis (optional) | Multi-agent bus, hot cache |
| RL Training | PyTorch + TRL | PPO/GRPO training |
| Testing | pytest, hypothesis | Unit + property-based |
| Docs | MkDocs Material | Developer docs |
| CI/CD | GitHub Actions | Automated testing + release |
| Package | PyPI | `pip install contextdb` |

### 8.3 Repository Structure
```
contextdb/
├── README.md
├── pyproject.toml
├── LICENSE                          # Apache 2.0
├── contextdb/
│   ├── __init__.py                  # Public API: init(), version
│   ├── core/
│   │   ├── __init__.py
│   │   ├── models.py                # MemoryItem, Edge, Entity, etc.
│   │   ├── config.py                # Configuration dataclasses
│   │   └── exceptions.py            # Custom exceptions
│   ├── store/
│   │   ├── __init__.py
│   │   ├── base.py                  # Abstract store interface
│   │   ├── sqlite_store.py          # SQLite backend
│   │   ├── postgres_store.py        # PostgreSQL backend
│   │   ├── vector_index.py          # FAISS/Qdrant wrapper
│   │   └── graph/
│   │       ├── __init__.py
│   │       ├── base_graph.py        # Abstract graph interface
│   │       ├── semantic_graph.py
│   │       ├── temporal_graph.py
│   │       ├── causal_graph.py
│   │       └── entity_graph.py
│   ├── dynamics/
│   │   ├── __init__.py
│   │   ├── formation/
│   │   │   ├── __init__.py
│   │   │   ├── segmenter.py         # Topic segmentation
│   │   │   ├── extractor.py         # Fact/experience/working extraction
│   │   │   └── compressor.py        # Compression-as-denoising
│   │   ├── evolution/
│   │   │   ├── __init__.py
│   │   │   ├── auto_linker.py       # Cross-graph linking
│   │   │   ├── consolidator.py      # Hierarchical consolidation
│   │   │   └── pruner.py            # Memory pruning
│   │   └── retrieval/
│   │       ├── __init__.py
│   │       ├── query_classifier.py  # Query type detection
│   │       ├── graph_traverser.py   # Per-graph traversal
│   │       ├── fuser.py             # Multi-graph fusion
│   │       └── context_assembler.py # Format for LLM consumption
│   ├── functions/
│   │   ├── __init__.py
│   │   ├── factual.py               # Factual memory API
│   │   ├── experiential.py          # Experiential memory API
│   │   └── working.py               # Working memory API
│   ├── privacy/
│   │   ├── __init__.py
│   │   ├── pii_detector.py          # PII detection + redaction
│   │   ├── retention.py             # Retention policy enforcement
│   │   ├── audit.py                 # Audit trail logging
│   │   └── access_control.py        # RBAC (paid tier)
│   ├── multiagent/
│   │   ├── __init__.py
│   │   ├── memory_bus.py            # Pub/sub event system
│   │   ├── role_router.py           # Role-aware memory routing
│   │   └── conflict_resolver.py     # Conflict detection + resolution
│   ├── rl/                          # PAID TIER
│   │   ├── __init__.py
│   │   ├── memory_manager.py        # RL-trained memory agent
│   │   ├── retrieval_policy.py      # Learned query routing
│   │   ├── training.py              # PPO/GRPO training pipeline
│   │   └── checkpoints/             # Pre-trained model weights
│   ├── integrations/
│   │   ├── __init__.py
│   │   ├── langchain.py
│   │   ├── llamaindex.py
│   │   ├── crewai.py
│   │   ├── openai_agents.py
│   │   └── autogen.py
│   └── utils/
│       ├── __init__.py
│       ├── embeddings.py            # Embedding model wrapper
│       ├── llm.py                   # LLM call wrapper
│       ├── tokenizer.py             # Token counting
│       └── migrations.py            # Mem0/Zep import tools
├── tests/
│   ├── unit/
│   │   ├── test_models.py
│   │   ├── test_sqlite_store.py
│   │   ├── test_formation.py
│   │   ├── test_retrieval.py
│   │   ├── test_privacy.py
│   │   └── ...
│   ├── integration/
│   │   ├── test_end_to_end.py
│   │   ├── test_multi_agent.py
│   │   └── test_frameworks.py
│   └── benchmarks/
│       ├── bench_latency.py
│       ├── bench_accuracy.py
│       └── bench_memory.py
├── docs/
│   ├── index.md
│   ├── quickstart.md
│   ├── architecture.md
│   ├── api-reference.md
│   ├── guides/
│   │   ├── customer-support.md
│   │   ├── research-assistant.md
│   │   └── phone-agent.md
│   └── migration/
│       ├── from-mem0.md
│       ├── from-zep.md
│       └── from-langchain.md
├── examples/
│   ├── quickstart.py
│   ├── customer_support_agent.py
│   ├── research_assistant.py
│   ├── phone_agent.py
│   ├── multi_agent_team.py
│   └── rl_training.py
└── benchmarks/
    ├── locomo/
    ├── longmemeval/
    ├── contextdb_support/
    ├── contextdb_research/
    └── contextdb_service/
```

---

## 9. Benchmarks & Success Metrics

### 9.1 Academic Benchmarks
| Benchmark | Metric | Target | Baseline (best existing) |
|-----------|--------|--------|-------------------------|
| LoCoMo | F1 | >0.72 | 0.65 (Mem0 + Graph) |
| LongMemEval | Accuracy | >0.78 | 0.71 (Zep) |
| MSC | BLEU-1 | >0.35 | 0.29 (MemGPT) |

### 9.2 Product Metrics (for SaaSLabs integration)
| Metric | Target | Measurement |
|--------|--------|------------|
| Customer support handle time | -30% | Helpwise A/B test |
| First-contact resolution | +20% | ServiceAgent.ai pilot |
| Cross-document connection F1 | >0.65 | ReadingNotes.ai eval |
| Developer time-to-integrate | <30 min | New user onboarding |
| GitHub stars (6 months) | 5,000+ | Community adoption |
| PyPI monthly downloads | 10,000+ | Usage growth |

### 9.3 Technical Quality Gates
- Test coverage: >85% for core modules
- All CI checks pass on every PR
- No P0 bugs in released versions
- API response time SLA: p95 <100ms
- Documentation coverage: every public method documented

---

## 10. Competitive Positioning

### 10.0 The Real Competition: The Patchwork
Our primary competitor is not another product — it's the DIY patchwork. Most teams today build their own agent memory from Pinecone + Redis + Postgres + glue code. ContextDB wins by being faster to integrate (3 lines vs 2-4 months), more capable (graph intelligence, learned policies, experiential memory), and cheaper to maintain (one dependency vs five).

### 10.1 vs Databricks Lakebase
Databricks offers managed Postgres + pgvector + LangGraph checkpointing. This is storage infrastructure, not memory intelligence.
- **Lakebase stores rows; ContextDB understands memory.** Lakebase doesn't segment conversations, doesn't extract experiences, doesn't build multi-graph representations, doesn't learn what to remember.
- **ContextDB can run ON Lakebase.** PostgreSQL is a pluggable backend for ContextDB. We complement Databricks at the infrastructure level while owning the memory semantics above it.
- **ContextDB is open-source and portable.** Lakebase locks you into the Databricks ecosystem. ContextDB runs anywhere: SQLite for local dev, any Postgres for production, Neo4j for power users.

### 10.2 vs Snowflake SnowWork
Snowflake is an analytics platform adding agent features. It's approaching from the data side, not the agent side. ContextDB is purpose-built for agent memory patterns that Snowflake's architecture was never designed for.

### 10.3 vs Mem0
Mem0 is the closest point competitor. Key differentiators:
- **ContextDB is truly open-source** — Mem0 gates graph memory at $249/month. ContextDB includes single-graph free.
- **ContextDB replaces the patchwork** — Mem0 is one piece of it (factual memory only). ContextDB replaces all of it.
- **Experiential memory** — Mem0 has no workflow/trajectory storage. ContextDB does.
- **Working memory** — Mem0 has no context paging. ContextDB does.
- **RL-trained management** — Mem0 uses heuristics. ContextDB learns optimal policies.
- **Multi-agent native** — Mem0 is single-agent. ContextDB has bus + routing + conflict resolution.

### 10.4 vs Zep
- **Broader scope** — Zep does temporal KG well but nothing else. ContextDB covers all forms × functions × dynamics.
- **Open source** — Zep's cloud is the primary product. ContextDB's OSS is genuinely useful standalone.

### 10.5 vs MemGPT/Letta
- **Persistent memory** — MemGPT focuses on working memory paging. ContextDB adds long-term factual + experiential.
- **Graph intelligence** — MemGPT has no graph structures. ContextDB has 4 orthogonal graphs.

### 10.6 Positioning Summary
| | Storage | Single memory type | Full memory OS |
|---|---|---|---|
| **Databricks Lakebase** | ✅ | | |
| **Pinecone + Redis + PG** | ✅ | | |
| **Mem0** | | ✅ (factual) | |
| **Zep** | | ✅ (temporal KG) | |
| **MemGPT** | | ✅ (working) | |
| **ContextDB** | | | ✅ |

---

## 11. Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Mem0 adds multi-graph to free tier | Medium | High | Ship faster; differentiate on RL + experiential |
| LLM costs for extraction/linking | High | Medium | Offer fast mode (no LLM); batch operations; caching |
| RL training too complex for users | Medium | Medium | Pre-trained checkpoints; fine-tuning wizard; managed option |
| Graph queries too slow at scale | Low | High | Index optimization; caching; Neo4j backend for power users |
| Adoption too slow for cloud revenue | Medium | High | Focus on OSS community first; cloud is Phase 5 |

---

## 12. Open Questions

1. **Should we support TypeScript/JavaScript SDK?** Many agent frameworks are JS-first. Potentially Phase 3+.
2. **Should the RL checkpoint be open-sourced or kept paid?** Open-sourcing would boost adoption; keeping it paid protects moat.
3. **Redis vs Kafka for multi-agent memory bus?** Redis is simpler; Kafka is more robust for enterprise. Start Redis, offer Kafka later.
4. **How to handle multimodal memory (images, audio)?** Not in v1. Design the data model to be extensible for v2.
5. **Should we build a CLI tool?** `contextdb inspect`, `contextdb migrate`, etc. Nice-to-have for Phase 3.
