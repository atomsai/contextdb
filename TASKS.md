# ContextDB — Cursor Build Prompt

> **Vision:** ContextDB is the unified context layer for AI agents — replacing the patchwork of Pinecone + Redis + Postgres + glue code with one system that understands memory. Think "Supabase for agent memory."
>
> **Positioning:** Databricks Lakebase gives agents a hard drive. ContextDB gives agents a brain.
>
> **How to use this document:**
> Feed each TASK block to Cursor one at a time, in order. Each task is self-contained with clear inputs, outputs, and acceptance criteria. Wait for each task to pass its acceptance criteria before moving to the next.
>
> **Total tasks:** 32 (across 5 phases)
> **Estimated time:** 4-6 weeks with focused Cursor sessions

---

## PHASE 1: PROJECT FOUNDATION (Tasks 1-8)

---

### TASK 1: Project Scaffold

```
Create a new Python project called "contextdb" with the following structure:

contextdb/
├── README.md                    # "ContextDB — The Unified Context Layer for AI Agents", badges, quickstart
├── pyproject.toml               # Use hatch as build backend
├── LICENSE                      # Apache 2.0
├── .github/
│   └── workflows/
│       └── ci.yml               # pytest + ruff + mypy on push/PR
├── contextdb/
│   ├── __init__.py              # Exports: init, __version__
│   ├── py.typed                 # PEP 561 marker
│   └── core/
│       ├── __init__.py
│       ├── config.py
│       └── exceptions.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   └── test_version.py
└── docs/
    └── index.md

Requirements in pyproject.toml:
- Python >=3.10
- Dependencies: pydantic>=2.0, numpy, aiosqlite
- Optional deps groups: [postgres] = asyncpg,pgvector; [neo4j] = neo4j; [redis] = redis; [all]
- Dev deps: pytest, pytest-asyncio, ruff, mypy, hypothesis
- Set version = "0.1.0"
- Set ruff config: line-length=100, target python 3.10
- Set mypy config: strict=true

config.py should define:
@dataclass or Pydantic BaseSettings class ContextDBConfig with fields:
  storage_url: str = "sqlite:///contextdb.db"
  embedding_model: str = "text-embedding-3-small"
  embedding_dim: int = 1536
  llm_model: str = "gpt-4o-mini"
  llm_api_key: str | None = None     # Falls back to OPENAI_API_KEY env var
  pii_action: Literal["redact", "encrypt", "flag", "allow"] = "redact"
  retention_ttl_days: int | None = 730
  log_level: str = "INFO"

exceptions.py should define:
  ContextDBError(Exception), MemoryNotFoundError, StorageError, PrivacyError, ConfigError

__init__.py should export:
  from contextdb.core.config import ContextDBConfig
  __version__ = "0.1.0"
  def init(user_id=None, config=None, **kwargs) -> ContextDB: ...  # stub for now

ACCEPTANCE CRITERIA:
- `pip install -e .` works
- `python -c "import contextdb; print(contextdb.__version__)"` prints "0.1.0"
- `ruff check .` passes
- `pytest` passes (test_version.py checks __version__)
- GitHub Actions CI runs on push
```

---

### TASK 2: Core Data Models

```
Create contextdb/core/models.py with the following Pydantic v2 models:

1. MemoryType (str Enum): FACTUAL, EXPERIENTIAL, WORKING

2. MemoryStatus (str Enum): ACTIVE, ARCHIVED, DELETED

3. PIIType (str Enum): NAME, EMAIL, PHONE, ADDRESS, SSN, CREDIT_CARD, CUSTOM

4. PIIAnnotation(BaseModel):
   pii_type: PIIType
   start: int           # Character offset start
   end: int             # Character offset end
   original: str        # Original text
   redacted: str        # Replacement text (e.g., "[NAME]")

5. Edge(BaseModel):
   source_id: str
   target_id: str
   graph_type: Literal["semantic", "temporal", "causal", "entity"]
   weight: float = 1.0
   metadata: dict = {}
   created_at: datetime = Field(default_factory=datetime.utcnow)

6. Entity(BaseModel):
   name: str
   entity_type: str     # PERSON, ORG, PRODUCT, LOCATION, etc.
   attributes: dict = {}
   memory_ids: list[str] = []  # Memories that mention this entity

7. RetentionPolicy(BaseModel):
   default_ttl: timedelta | None = timedelta(days=730)
   factual_ttl: timedelta | None = timedelta(days=1825)  # 5 years
   experiential_ttl: timedelta | None = None              # Never
   working_ttl: timedelta | None = timedelta(hours=24)
   right_to_erasure: bool = True

8. MemoryItem(BaseModel):
   id: str = Field(default_factory=lambda: str(uuid4()))
   content: str
   embedding: list[float] | None = None
   memory_type: MemoryType = MemoryType.FACTUAL
   source: str = ""
   metadata: dict = {}

   # Bitemporal
   event_time: datetime | None = None
   ingestion_time: datetime = Field(default_factory=datetime.utcnow)

   # Privacy
   pii_annotations: list[PIIAnnotation] = []
   retention_policy: RetentionPolicy | None = None

   # Lifecycle
   created_at: datetime = Field(default_factory=datetime.utcnow)
   updated_at: datetime = Field(default_factory=datetime.utcnow)
   access_count: int = 0
   last_accessed: datetime | None = None
   confidence: float = 1.0
   status: MemoryStatus = MemoryStatus.ACTIVE

   # Relationships (populated by graph operations)
   entity_mentions: list[str] = []       # Entity names
   tags: list[str] = []

Write thorough tests in tests/unit/test_models.py:
- Test MemoryItem creation with defaults
- Test MemoryItem creation with all fields
- Test serialization/deserialization (model_dump / model_validate)
- Test Edge creation for each graph type
- Test RetentionPolicy defaults
- Test MemoryType enum values
- Test PIIAnnotation with real examples

ACCEPTANCE CRITERIA:
- All models import correctly
- All tests pass
- mypy passes with strict mode
- ruff passes
```

---

### TASK 3: SQLite Storage Backend

```
Create contextdb/store/base.py with an abstract base class:

class BaseStore(ABC):
    @abstractmethod
    async def add(self, item: MemoryItem) -> MemoryItem: ...
    @abstractmethod
    async def get(self, memory_id: str) -> MemoryItem | None: ...
    @abstractmethod
    async def update(self, memory_id: str, **kwargs) -> MemoryItem: ...
    @abstractmethod
    async def delete(self, memory_id: str, hard: bool = False) -> None: ...
    @abstractmethod
    async def search_by_embedding(self, embedding: list[float], top_k: int = 10,
                                   filters: dict | None = None) -> list[MemoryItem]: ...
    @abstractmethod
    async def list_memories(self, user_id: str | None = None,
                            memory_type: MemoryType | None = None,
                            status: MemoryStatus = MemoryStatus.ACTIVE,
                            limit: int = 100, offset: int = 0) -> list[MemoryItem]: ...
    @abstractmethod
    async def count(self, user_id: str | None = None) -> int: ...
    @abstractmethod
    async def close(self) -> None: ...

Create contextdb/store/sqlite_store.py implementing BaseStore:

class SQLiteStore(BaseStore):
    Uses aiosqlite for async SQLite operations.

    Schema (create on init if not exists):
    CREATE TABLE IF NOT EXISTS memories (
        id TEXT PRIMARY KEY,
        content TEXT NOT NULL,
        embedding BLOB,                    -- numpy array serialized with numpy.tobytes()
        memory_type TEXT NOT NULL DEFAULT 'FACTUAL',
        source TEXT DEFAULT '',
        metadata TEXT DEFAULT '{}',         -- JSON string
        user_id TEXT,
        event_time TEXT,                    -- ISO format
        ingestion_time TEXT NOT NULL,
        pii_annotations TEXT DEFAULT '[]',  -- JSON string
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        access_count INTEGER DEFAULT 0,
        last_accessed TEXT,
        confidence REAL DEFAULT 1.0,
        status TEXT DEFAULT 'ACTIVE',
        entity_mentions TEXT DEFAULT '[]',  -- JSON array of strings
        tags TEXT DEFAULT '[]'              -- JSON array of strings
    );

    CREATE INDEX IF NOT EXISTS idx_memories_user_id ON memories(user_id);
    CREATE INDEX IF NOT EXISTS idx_memories_status ON memories(status);
    CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type);
    CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at);

    For search_by_embedding:
    - Load all active embeddings into memory (cached, invalidated on add/update/delete)
    - Compute cosine similarity with numpy
    - Return top-k results
    - This is intentionally simple. FAISS integration comes in Task 5.

    Implement soft delete (set status=DELETED) by default, hard delete optional.
    On get(), increment access_count and set last_accessed.

Write tests in tests/unit/test_sqlite_store.py:
- Test add and get
- Test update (content, metadata, status)
- Test soft delete and hard delete
- Test search_by_embedding returns correct ordering
- Test list_memories with filters
- Test count
- Test persistence (close and reopen)
- Use pytest-asyncio with async tests
- Use tmp_path fixture for test databases

ACCEPTANCE CRITERIA:
- All CRUD operations work correctly
- Embedding search returns results ordered by similarity
- Soft delete hides from list but allows get
- All tests pass
- No SQL injection vulnerabilities (parameterized queries only)
```

---

### TASK 4: Embedding Wrapper

```
Create contextdb/utils/embeddings.py:

class EmbeddingProvider(ABC):
    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]: ...
    @abstractmethod
    def dimension(self) -> int: ...

class OpenAIEmbedding(EmbeddingProvider):
    def __init__(self, model: str = "text-embedding-3-small", api_key: str | None = None):
        # Uses openai async client
        # Batches requests (max 2048 texts per call)
        # Handles rate limiting with exponential backoff

    async def embed(self, texts: list[str]) -> list[list[float]]:
        # Call OpenAI API
        # Return list of embedding vectors

    def dimension(self) -> int:
        # Return dimension based on model name
        # text-embedding-3-small: 1536
        # text-embedding-3-large: 3072
        # text-embedding-ada-002: 1536

class SentenceTransformerEmbedding(EmbeddingProvider):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        # Uses sentence-transformers library (optional dep)
        # Falls back gracefully if not installed

    async def embed(self, texts: list[str]) -> list[list[float]]:
        # Run model inference (CPU by default)
        # Return embeddings

class MockEmbedding(EmbeddingProvider):
    """For testing without API calls."""
    def __init__(self, dimension: int = 384):
        self.dim = dimension
    async def embed(self, texts: list[str]) -> list[list[float]]:
        # Return deterministic pseudo-random embeddings based on text hash
        # Same text always produces same embedding (for test reproducibility)

def get_embedding_provider(model: str, api_key: str | None = None) -> EmbeddingProvider:
    # Factory function
    # "text-embedding-*" → OpenAIEmbedding
    # "mock" or "test" → MockEmbedding
    # Anything else → SentenceTransformerEmbedding

Add openai>=1.0 to project dependencies.
Add sentence-transformers to optional [local] deps group.

Write tests using MockEmbedding:
- Test same text produces same embedding
- Test different texts produce different embeddings
- Test batch embedding
- Test dimension reporting
- Test factory function routing

ACCEPTANCE CRITERIA:
- MockEmbedding works without any API keys
- OpenAIEmbedding works with OPENAI_API_KEY set
- Factory function routes correctly
- All tests pass without API keys (using MockEmbedding)
```

---

### TASK 5: Vector Index (FAISS)

```
Create contextdb/store/vector_index.py:

class VectorIndex(ABC):
    @abstractmethod
    def add(self, ids: list[str], embeddings: np.ndarray) -> None: ...
    @abstractmethod
    def search(self, query: np.ndarray, top_k: int = 10) -> list[tuple[str, float]]: ...
    @abstractmethod
    def remove(self, ids: list[str]) -> None: ...
    @abstractmethod
    def save(self, path: str) -> None: ...
    @abstractmethod
    def load(self, path: str) -> None: ...
    @abstractmethod
    def __len__(self) -> int: ...

class FAISSIndex(VectorIndex):
    def __init__(self, dimension: int, index_type: str = "flat"):
        # index_type: "flat" (exact, small datasets), "ivf" (approximate, large datasets)
        # For "flat": faiss.IndexFlatIP (inner product, assumes normalized vectors)
        # For "ivf": faiss.IndexIVFFlat with nlist=min(4096, n//39 + 1)
        # Maintain an id_map: dict[int, str] mapping FAISS internal IDs to memory IDs

    def add(self, ids: list[str], embeddings: np.ndarray):
        # Normalize embeddings (L2 norm)
        # Add to FAISS index
        # Update id_map

    def search(self, query: np.ndarray, top_k: int = 10) -> list[tuple[str, float]]:
        # Normalize query
        # FAISS search
        # Map internal IDs back to memory IDs via id_map
        # Return list of (memory_id, similarity_score) sorted descending

    def remove(self, ids: list[str]):
        # FAISS doesn't support direct removal for all index types
        # Strategy: mark as removed in id_map, rebuild periodically
        # For flat index: rebuild from scratch (fast enough for <100K)

    def save(self, path: str):
        # faiss.write_index() + pickle id_map

    def load(self, path: str):
        # faiss.read_index() + unpickle id_map

class NumpyIndex(VectorIndex):
    """Fallback when FAISS is not installed. Pure numpy brute-force."""
    # Store embeddings as numpy array
    # Search via np.dot for cosine similarity
    # Simple but works for <10K memories

Add faiss-cpu to optional [faiss] deps. Add it to [all] group too.

Now update SQLiteStore to use VectorIndex:
- Accept optional vector_index parameter
- On add(): also add embedding to vector_index
- On search_by_embedding(): use vector_index.search() instead of brute-force numpy
- On delete(): also remove from vector_index
- Lazy init: create vector_index on first embedding operation

Write tests:
- Test FAISSIndex add/search/remove/save/load
- Test NumpyIndex same operations
- Test that both return same results for small datasets
- Test search accuracy: nearest neighbor is correct for known embeddings

ACCEPTANCE CRITERIA:
- FAISS index works for add/search/save/load
- NumpyIndex works as fallback
- SQLiteStore uses vector index when available
- Search returns correct nearest neighbors
- All tests pass
```

---

### TASK 6: LLM Wrapper

```
Create contextdb/utils/llm.py:

class LLMProvider(ABC):
    @abstractmethod
    async def generate(self, prompt: str, system: str = "",
                       temperature: float = 0.0,
                       max_tokens: int = 1000,
                       response_format: type | None = None) -> str: ...

class OpenAILLM(LLMProvider):
    def __init__(self, model: str = "gpt-4o-mini", api_key: str | None = None):
        # Uses openai async client
        # Supports structured output (response_format=SomePydanticModel)

    async def generate(self, prompt, system="", temperature=0.0,
                       max_tokens=1000, response_format=None) -> str:
        # If response_format is a Pydantic model, use OpenAI structured outputs
        # Otherwise plain text completion
        # Handle rate limits with exponential backoff (3 retries)

class MockLLM(LLMProvider):
    """For testing. Returns configurable responses."""
    def __init__(self, responses: dict[str, str] | None = None):
        self.responses = responses or {}
        self.calls: list[dict] = []  # Record all calls for assertions

    async def generate(self, prompt, **kwargs) -> str:
        self.calls.append({"prompt": prompt, **kwargs})
        # Check if any key in self.responses matches a substring of prompt
        for key, response in self.responses.items():
            if key in prompt:
                return response
        return '{"facts": [], "entities": []}'  # Safe default

def get_llm_provider(model: str, api_key: str | None = None) -> LLMProvider:
    # "mock" → MockLLM
    # "gpt-*" or "o1-*" → OpenAILLM
    # Others: raise ConfigError with helpful message

Write tests using MockLLM:
- Test call recording
- Test response matching
- Test factory routing

ACCEPTANCE CRITERIA:
- MockLLM allows fully offline testing
- OpenAILLM handles structured output
- Factory function works
- All tests pass without API keys
```

---

### TASK 7: PII Detection

```
Create contextdb/privacy/pii_detector.py:

class PIIDetector:
    def __init__(self, action: Literal["redact", "encrypt", "flag", "allow"] = "redact"):
        self.action = action
        # Built-in regex patterns for common PII:
        self._patterns = {
            PIIType.EMAIL: r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            PIIType.PHONE: r'\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',
            PIIType.SSN: r'\b\d{3}-\d{2}-\d{4}\b',
            PIIType.CREDIT_CARD: r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
            PIIType.ADDRESS: None,  # Requires NER, handled by spacy if available
        }
        # Name detection: use a simple heuristic (capitalized word pairs) + optional spacy

    def detect(self, text: str) -> list[PIIAnnotation]:
        """Find all PII in text. Returns list of annotations with character offsets."""
        annotations = []
        for pii_type, pattern in self._patterns.items():
            if pattern:
                for match in re.finditer(pattern, text):
                    annotations.append(PIIAnnotation(
                        pii_type=pii_type,
                        start=match.start(),
                        end=match.end(),
                        original=match.group(),
                        redacted=f"[{pii_type.value}]"
                    ))
        return sorted(annotations, key=lambda a: a.start)

    def redact(self, text: str, annotations: list[PIIAnnotation] | None = None) -> str:
        """Replace PII with typed placeholders. e.g., 'Call John at 555-1234' → 'Call [NAME] at [PHONE]'"""
        if annotations is None:
            annotations = self.detect(text)
        # Replace from end to start to preserve offsets
        result = text
        for ann in sorted(annotations, key=lambda a: a.start, reverse=True):
            result = result[:ann.start] + ann.redacted + result[ann.end:]
        return result

    def process(self, text: str) -> tuple[str, list[PIIAnnotation]]:
        """Detect PII and apply configured action. Returns (processed_text, annotations)."""
        annotations = self.detect(text)
        if self.action == "redact":
            return self.redact(text, annotations), annotations
        elif self.action == "allow":
            return text, annotations
        elif self.action == "flag":
            return text, annotations  # Annotations serve as flags
        elif self.action == "encrypt":
            # For MVP: same as redact but store originals encrypted
            # TODO: implement actual encryption in Phase 4
            return self.redact(text, annotations), annotations

Write thorough tests in tests/unit/test_pii.py:
- Test email detection (various formats: user@domain.com, user+tag@sub.domain.co.uk)
- Test phone detection (various formats: 555-1234, (555) 123-4567, +1-555-123-4567)
- Test SSN detection
- Test credit card detection (with/without dashes/spaces)
- Test redaction preserves non-PII text exactly
- Test multiple PII in same text
- Test overlapping PII (edge case)
- Test empty text
- Test text with no PII
- Test process() with each action type

ACCEPTANCE CRITERIA:
- Detects emails, phones, SSNs, credit cards with >90% precision
- Redaction produces correct output with correct placeholders
- No false positives on common text patterns (e.g., "I have 3 cats" should not trigger)
- All tests pass
```

---

### TASK 8: Core ContextDB Class + init()

```
Create contextdb/client.py — this is the main user-facing class:

class ContextDB:
    """Main interface for ContextDB memory operations."""

    def __init__(self, config: ContextDBConfig, user_id: str | None = None):
        self.config = config
        self.user_id = user_id
        self._store: SQLiteStore | None = None
        self._embedder: EmbeddingProvider | None = None
        self._llm: LLMProvider | None = None
        self._pii: PIIDetector | None = None
        self._initialized = False

    async def _ensure_init(self):
        if not self._initialized:
            self._store = SQLiteStore(self.config.storage_url)
            await self._store.initialize()
            self._embedder = get_embedding_provider(self.config.embedding_model, self.config.llm_api_key)
            self._llm = get_llm_provider(self.config.llm_model, self.config.llm_api_key)
            self._pii = PIIDetector(action=self.config.pii_action)
            self._initialized = True

    async def add(self, content: str, memory_type: MemoryType = MemoryType.FACTUAL,
                  metadata: dict | None = None, event_time: datetime | None = None,
                  source: str = "") -> MemoryItem:
        """Add a memory. PII is automatically handled per config."""
        await self._ensure_init()

        # 1. PII detection and processing
        processed_content, pii_annotations = self._pii.process(content)

        # 2. Generate embedding
        embeddings = await self._embedder.embed([processed_content])

        # 3. Create MemoryItem
        item = MemoryItem(
            content=processed_content,
            embedding=embeddings[0],
            memory_type=memory_type,
            source=source,
            metadata=metadata or {},
            event_time=event_time or datetime.utcnow(),
            pii_annotations=pii_annotations,
            entity_mentions=[],  # TODO: entity extraction in Phase 2
        )

        # 4. Store
        return await self._store.add(item)

    async def search(self, query: str, top_k: int = 10,
                     memory_type: MemoryType | None = None,
                     time_range: tuple[datetime, datetime] | None = None) -> list[MemoryItem]:
        """Search memories by semantic similarity."""
        await self._ensure_init()
        embeddings = await self._embedder.embed([query])
        results = await self._store.search_by_embedding(
            embeddings[0], top_k=top_k,
            filters={"memory_type": memory_type.value if memory_type else None}
        )
        return results

    async def get(self, memory_id: str) -> MemoryItem | None:
        await self._ensure_init()
        return await self._store.get(memory_id)

    async def update(self, memory_id: str, content: str | None = None,
                     metadata: dict | None = None) -> MemoryItem:
        await self._ensure_init()
        kwargs = {}
        if content is not None:
            processed, pii = self._pii.process(content)
            embeddings = await self._embedder.embed([processed])
            kwargs["content"] = processed
            kwargs["embedding"] = embeddings[0]
            kwargs["pii_annotations"] = pii
        if metadata is not None:
            kwargs["metadata"] = metadata
        return await self._store.update(memory_id, **kwargs)

    async def delete(self, memory_id: str) -> None:
        await self._ensure_init()
        await self._store.delete(memory_id)

    async def forget(self, user_id: str | None = None, entity: str | None = None,
                     older_than: timedelta | None = None) -> int:
        """Bulk delete memories matching criteria. Returns count deleted."""
        await self._ensure_init()
        # Implement bulk deletion based on criteria
        # Returns number of memories deleted
        ...

    async def stats(self) -> dict:
        """Return memory statistics."""
        await self._ensure_init()
        count = await self._store.count(self.user_id)
        return {"total_memories": count, "user_id": self.user_id}

    async def close(self):
        if self._store:
            await self._store.close()

    async def __aenter__(self):
        await self._ensure_init()
        return self

    async def __aexit__(self, *args):
        await self.close()

Now update contextdb/__init__.py:

from contextdb.client import ContextDB
from contextdb.core.config import ContextDBConfig
from contextdb.core.models import MemoryItem, MemoryType, MemoryStatus

__version__ = "0.1.0"

def init(user_id: str | None = None, config: ContextDBConfig | None = None,
         **kwargs) -> ContextDB:
    """Initialize a ContextDB instance.

    Usage:
        db = contextdb.init(user_id="user_123")
        db = contextdb.init(config=ContextDBConfig(storage_url="postgresql://..."))
    """
    if config is None:
        config = ContextDBConfig(**kwargs)
    return ContextDB(config=config, user_id=user_id)

Write integration tests in tests/integration/test_end_to_end.py:
- Test the full 3-line quickstart flow (init → add → search)
- Test add returns MemoryItem with id
- Test search returns relevant results (add 3 memories, search for one)
- Test update changes content
- Test delete removes from search results
- Test PII redaction happens automatically
- Test context manager (async with)
- Use MockEmbedding and MockLLM (no API keys needed)
- Use tmp_path for database

ACCEPTANCE CRITERIA:
- The 3-line quickstart works:
    db = contextdb.init(user_id="test", embedding_model="mock")
    await db.add("Alex prefers email over phone")
    results = await db.search("How does Alex prefer to be contacted?")
- All integration tests pass
- No API keys required for tests
- ruff + mypy pass
```

---

## PHASE 2: GRAPH INTELLIGENCE (Tasks 9-16)

---

### TASK 9: Graph Base + Semantic Graph

```
Create contextdb/store/graph/base_graph.py:

class BaseGraph(ABC):
    @abstractmethod
    async def add_node(self, memory_id: str, data: dict) -> None: ...
    @abstractmethod
    async def add_edge(self, edge: Edge) -> None: ...
    @abstractmethod
    async def get_neighbors(self, memory_id: str, depth: int = 1,
                            max_results: int = 20) -> list[tuple[str, float]]: ...
    @abstractmethod
    async def remove_node(self, memory_id: str) -> None: ...
    @abstractmethod
    async def get_edges(self, memory_id: str) -> list[Edge]: ...

Create contextdb/store/graph/semantic_graph.py:

class SemanticGraph(BaseGraph):
    """Edges represent embedding similarity above a threshold."""

    def __init__(self, store: BaseStore, threshold: float = 0.7):
        self.store = store
        self.threshold = threshold
        # Edges stored in SQLite table: semantic_edges(source_id, target_id, weight, created_at)

    async def add_node(self, memory_id: str, data: dict):
        """When a new memory is added, find all existing memories with similarity > threshold
        and create bidirectional edges."""
        embedding = data.get("embedding")
        if embedding is None:
            return
        # Search for similar memories
        similar = await self.store.search_by_embedding(embedding, top_k=50)
        for item in similar:
            similarity = cosine_similarity(embedding, item.embedding)
            if similarity > self.threshold and item.id != memory_id:
                await self.add_edge(Edge(
                    source_id=memory_id, target_id=item.id,
                    graph_type="semantic", weight=similarity
                ))

    async def get_neighbors(self, memory_id: str, depth: int = 1,
                            max_results: int = 20) -> list[tuple[str, float]]:
        """BFS traversal up to depth, returning (memory_id, cumulative_weight) pairs."""
        # Implement BFS/DFS with depth limit
        # Return sorted by weight descending

Write tests:
- Test edge creation when similar memories exist
- Test no edges when memories are dissimilar
- Test get_neighbors returns correct depth traversal
- Test remove_node cleans up edges

ACCEPTANCE CRITERIA:
- Semantic edges are created automatically on memory addition
- Threshold filtering works correctly
- Neighbor traversal returns correct results
- Tests pass
```

---

### TASK 10: Entity Graph

```
Create contextdb/store/graph/entity_graph.py:

class EntityGraph(BaseGraph):
    """Nodes include memories AND named entities. Edges link memories to entities
    and entities to each other."""

    def __init__(self, store: BaseStore, llm: LLMProvider):
        self.store = store
        self.llm = llm
        # Tables:
        #   entities(id, name, entity_type, attributes JSON, created_at, updated_at)
        #   entity_edges(source_id, target_id, relation_type, weight, created_at)
        #   memory_entity_edges(memory_id, entity_id, relation, created_at)

    async def extract_entities(self, content: str) -> list[Entity]:
        """Use LLM to extract named entities from text."""
        prompt = '''Extract all named entities from this text. Return JSON:
        {"entities": [{"name": "...", "type": "PERSON|ORG|PRODUCT|LOCATION|EVENT|OTHER", "attributes": {}}]}

        Text: {content}'''
        response = await self.llm.generate(prompt.format(content=content))
        # Parse JSON response into Entity objects
        # Deduplicate against existing entities (fuzzy name match)

    async def add_node(self, memory_id: str, data: dict):
        """Extract entities from memory content and create entity-memory edges."""
        content = data.get("content", "")
        entities = await self.extract_entities(content)
        for entity in entities:
            existing = await self._find_existing_entity(entity.name)
            if existing:
                entity_id = existing.id
                # Update attributes if new info
            else:
                entity_id = await self._create_entity(entity)
            await self._link_memory_to_entity(memory_id, entity_id)

    async def get_entity(self, name: str) -> Entity | None: ...
    async def get_entity_memories(self, entity_name: str) -> list[str]: ...
    async def get_related_entities(self, entity_name: str) -> list[tuple[str, str]]: ...

    async def get_neighbors(self, memory_id: str, depth=1, max_results=20):
        """Get memories connected through shared entities."""
        # 1. Find entities mentioned in this memory
        # 2. Find other memories mentioning same entities
        # 3. Rank by number of shared entities
        # Optional: Personalized PageRank for multi-hop

Write tests using MockLLM:
- Test entity extraction from sample text
- Test entity deduplication (same entity mentioned differently)
- Test memory-entity linking
- Test get_neighbors via shared entities
- Test get_entity_memories

ACCEPTANCE CRITERIA:
- Entities are extracted and stored correctly
- Memory-entity edges are bidirectional
- Shared entity retrieval works
- All tests pass with MockLLM
```

---

### TASK 11: Temporal Graph

```
Create contextdb/store/graph/temporal_graph.py:

class TemporalGraph(BaseGraph):
    """Edges represent temporal ordering and proximity between memories."""

    def __init__(self, store: BaseStore, proximity_window: timedelta = timedelta(hours=24)):
        self.store = store
        self.proximity_window = proximity_window
        # Table: temporal_edges(source_id, target_id, relation TEXT, weight, time_diff_seconds)
        # relation: "BEFORE", "AFTER", "CONCURRENT" (within 5 min), "SAME_SESSION"

    async def add_node(self, memory_id: str, data: dict):
        """Create temporal edges to nearby memories based on event_time."""
        event_time = data.get("event_time")
        if not event_time:
            return
        # Find memories within proximity_window
        nearby = await self._find_temporally_nearby(event_time)
        for other_id, other_time in nearby:
            if other_id == memory_id:
                continue
            diff = (event_time - other_time).total_seconds()
            relation = "CONCURRENT" if abs(diff) < 300 else ("AFTER" if diff > 0 else "BEFORE")
            weight = 1.0 / (1.0 + abs(diff) / 3600)  # Decay by hours
            await self.add_edge(Edge(
                source_id=memory_id, target_id=other_id,
                graph_type="temporal", weight=weight,
                metadata={"relation": relation, "time_diff_seconds": diff}
            ))

    async def get_timeline(self, entity: str | None = None,
                           start: datetime | None = None,
                           end: datetime | None = None) -> list[MemoryItem]:
        """Get chronologically ordered memories, optionally filtered."""

    async def get_neighbors(self, memory_id: str, depth=1, max_results=20):
        """Recency-weighted neighborhood expansion."""
        # Weight by temporal proximity (closer in time = higher weight)

Write tests:
- Test temporal edges created for nearby events
- Test BEFORE/AFTER/CONCURRENT relation assignment
- Test weight decay with time distance
- Test get_timeline ordering
- Test temporal neighbor retrieval

ACCEPTANCE CRITERIA:
- Temporal edges correctly reflect time ordering
- Weight decay works as expected
- Timeline retrieval returns chronological order
- Tests pass
```

---

### TASK 12: Causal Graph

```
Create contextdb/store/graph/causal_graph.py:

class CausalGraph(BaseGraph):
    """Edges represent causal or logical dependencies inferred by the LLM."""

    def __init__(self, store: BaseStore, llm: LLMProvider):
        self.store = store
        self.llm = llm
        # Table: causal_edges(source_id, target_id, relation_type, confidence, explanation)
        # relation_type: "CAUSED_BY", "LED_TO", "CONTRADICTS", "SUPPORTS", "DEPENDS_ON"

    async def add_node(self, memory_id: str, data: dict):
        """Check for causal relationships with recent/related memories.
        This is async and can be batched for efficiency."""
        content = data.get("content", "")
        # Get recent memories (last 20) and semantically similar (top 10)
        candidates = await self._get_candidates(memory_id, content)

        if not candidates:
            return

        # Use LLM to infer causal relationships
        prompt = '''Analyze if there are causal or logical relationships between these memories.
        New memory: "{content}"
        Existing memories:
        {candidates_text}

        Return JSON array of relationships found:
        [{"existing_memory_index": 0, "relation": "CAUSED_BY|LED_TO|CONTRADICTS|SUPPORTS|DEPENDS_ON",
          "confidence": 0.0-1.0, "explanation": "brief reason"}]
        Return empty array [] if no relationships found.'''

        response = await self.llm.generate(prompt.format(
            content=content,
            candidates_text=self._format_candidates(candidates)
        ))
        # Parse and create edges

    async def get_causal_chain(self, memory_id: str, direction: str = "forward",
                                max_depth: int = 5) -> list[tuple[str, str, float]]:
        """Follow causal chain forward (effects) or backward (causes).
        Returns list of (memory_id, relation_type, confidence)."""

Write tests using MockLLM:
- Test causal edge creation
- Test chain following (forward and backward)
- Test confidence thresholding
- Test contradiction detection

ACCEPTANCE CRITERIA:
- LLM-based causal inference works (with mock)
- Chain traversal returns correct paths
- Tests pass
```

---

### TASK 13: Multi-Graph Retrieval Engine

```
Create contextdb/dynamics/retrieval/query_classifier.py:

class QueryClassifier:
    """Determines which graphs are most relevant for a given query."""

    # Rule-based classifier (MVP). Can be replaced with trained model later.
    TEMPORAL_KEYWORDS = {"when", "before", "after", "last", "first", "recent", "history",
                         "timeline", "ago", "yesterday", "earlier", "previously", "next"}
    CAUSAL_KEYWORDS = {"why", "because", "caused", "led to", "result", "reason", "effect",
                       "consequence", "due to", "therefore"}
    ENTITY_KEYWORDS = {"who", "which", "what is", "tell me about", "profile", "details"}

    def classify(self, query: str) -> dict[str, float]:
        """Return graph weights: {"semantic": w, "temporal": w, "causal": w, "entity": w}"""
        query_lower = query.lower()
        weights = {"semantic": 0.25, "temporal": 0.25, "causal": 0.25, "entity": 0.25}

        # Boost weights based on keyword presence
        temporal_score = sum(1 for kw in self.TEMPORAL_KEYWORDS if kw in query_lower)
        causal_score = sum(1 for kw in self.CAUSAL_KEYWORDS if kw in query_lower)
        entity_score = sum(1 for kw in self.ENTITY_KEYWORDS if kw in query_lower)

        # Normalize and blend with base weights
        total = temporal_score + causal_score + entity_score + 1  # +1 for semantic baseline
        weights["semantic"] = max(0.1, 1.0 / total)
        weights["temporal"] = max(0.05, temporal_score / total) if temporal_score else 0.1
        weights["causal"] = max(0.05, causal_score / total) if causal_score else 0.1
        weights["entity"] = max(0.05, entity_score / total) if entity_score else 0.1

        # Normalize to sum to 1
        total_w = sum(weights.values())
        return {k: v / total_w for k, v in weights.items()}


Create contextdb/dynamics/retrieval/fuser.py:

class RetrievalFuser:
    """Combines results from multiple graph traversals using weighted reciprocal rank fusion."""

    def fuse(self, results: dict[str, list[tuple[str, float]]],
             weights: dict[str, float], top_k: int = 10) -> list[tuple[str, float]]:
        """
        Args:
            results: {graph_type: [(memory_id, score), ...]} from each graph
            weights: {graph_type: weight} from query classifier
            top_k: number of results to return
        Returns:
            [(memory_id, fused_score)] sorted by score descending
        """
        # Reciprocal Rank Fusion:
        # For each memory, compute: sum over graphs of (weight * 1/(rank + k))
        # where k=60 (standard RRF constant)
        scores: dict[str, float] = {}
        for graph_type, graph_results in results.items():
            w = weights.get(graph_type, 0)
            for rank, (mem_id, _score) in enumerate(graph_results):
                rrf_score = w * (1.0 / (rank + 60))
                scores[mem_id] = scores.get(mem_id, 0) + rrf_score

        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]


Create contextdb/dynamics/retrieval/engine.py:

class RetrievalEngine:
    """Orchestrates multi-graph retrieval."""

    def __init__(self, store, graphs: dict[str, BaseGraph],
                 classifier: QueryClassifier, fuser: RetrievalFuser):
        self.store = store
        self.graphs = graphs
        self.classifier = classifier
        self.fuser = fuser

    async def search(self, query: str, query_embedding: list[float],
                     top_k: int = 10) -> list[MemoryItem]:
        # 1. Classify query
        weights = self.classifier.classify(query)

        # 2. Search each graph in parallel
        results = {}
        # Semantic: always use embedding search
        results["semantic"] = await self._semantic_search(query_embedding, top_k * 3)
        # Other graphs: use graph traversal from semantic top results
        if "temporal" in self.graphs:
            results["temporal"] = await self._temporal_search(query, top_k * 3)
        if "causal" in self.graphs:
            results["causal"] = await self._causal_search(query, top_k * 3)
        if "entity" in self.graphs:
            results["entity"] = await self._entity_search(query, top_k * 3)

        # 3. Fuse
        fused = self.fuser.fuse(results, weights, top_k)

        # 4. Fetch full MemoryItems
        items = []
        for mem_id, score in fused:
            item = await self.store.get(mem_id)
            if item:
                items.append(item)
        return items

Write tests:
- Test QueryClassifier assigns correct weights for different query types
- Test RetrievalFuser produces correct rankings
- Test RetrievalEngine end-to-end with mock graphs
- Test that temporal queries boost temporal graph weight
- Test that "who is X" queries boost entity graph weight

ACCEPTANCE CRITERIA:
- Query classification produces reasonable weights
- Fusion correctly combines multi-graph results
- End-to-end retrieval returns relevant memories
- Tests pass
```

---

### TASK 14: Formation Pipeline

```
Create contextdb/dynamics/formation/segmenter.py:

class Segmenter:
    """Splits raw input into topically coherent segments."""

    async def segment(self, text: str, method: str = "sentence") -> list[str]:
        """
        Methods:
        - "turn": Split by conversation turns (\\n\\n or speaker labels)
        - "sentence": Split by sentences (using nltk or regex)
        - "topic": LLM-based topic segmentation (higher quality, slower)
        - "sliding": Sliding window with embedding drop detection
        """


Create contextdb/dynamics/formation/extractor.py:

class MemoryExtractor:
    """Extracts structured memory items from text segments."""

    def __init__(self, llm: LLMProvider):
        self.llm = llm

    async def extract(self, segment: str) -> list[dict]:
        """Extract factual, experiential, and working memory items from a segment.

        Returns list of dicts with keys:
        - content: str (the memory text)
        - memory_type: "FACTUAL" | "EXPERIENTIAL" | "WORKING"
        - entities: list[str]
        - event_time: str | None (if mentioned or inferable)
        - confidence: float
        """
        prompt = '''Analyze this text segment and extract distinct memory items.

For each memory item, classify it as:
- FACTUAL: Facts, preferences, attributes, states (e.g., "User prefers dark mode")
- EXPERIENTIAL: Actions, outcomes, lessons, patterns (e.g., "Restarting the server fixed the timeout")
- WORKING: Active goals, open questions, temporary context (e.g., "User is currently debugging auth flow")

Text: "{segment}"

Return JSON:
{{"memories": [
  {{"content": "...", "memory_type": "FACTUAL|EXPERIENTIAL|WORKING",
    "entities": ["entity1", "entity2"], "confidence": 0.0-1.0}}
]}}'''

        response = await self.llm.generate(prompt.format(segment=segment))
        # Parse and return


Create contextdb/dynamics/formation/compressor.py:

class MemoryCompressor:
    """Compresses memory content via denoising."""

    def __init__(self, llm: LLMProvider):
        self.llm = llm

    async def compress(self, content: str, ratio: float = 0.5) -> str:
        """Compress content to approximately ratio * original length.
        Uses compression-as-denoising: remove filler, keep signal."""

        target_words = max(5, int(len(content.split()) * ratio))
        prompt = f'''Compress this text to approximately {target_words} words.
Keep all factual content, entity names, dates, and numbers.
Remove filler words, pleasantries, and redundancy.
Output only the compressed text, nothing else.

Text: "{content}"'''

        return await self.llm.generate(prompt)


Create contextdb/dynamics/formation/pipeline.py:

class FormationPipeline:
    """Orchestrates: Raw Input → Segment → Extract → Compress → MemoryItems"""

    def __init__(self, segmenter, extractor, compressor, pii_detector, embedder):
        ...

    async def process(self, raw_input: str, source: str = "",
                      compress: bool = True) -> list[MemoryItem]:
        """Process raw input into structured, compressed, PII-safe memory items."""
        # 1. Segment
        segments = await self.segmenter.segment(raw_input)
        # 2. Extract from each segment
        all_extracted = []
        for seg in segments:
            extracted = await self.extractor.extract(seg)
            all_extracted.extend(extracted)
        # 3. Compress each
        items = []
        for ext in all_extracted:
            content = ext["content"]
            if compress:
                content = await self.compressor.compress(content)
            # 4. PII processing
            content, pii = self.pii_detector.process(content)
            # 5. Embed
            embedding = (await self.embedder.embed([content]))[0]
            # 6. Create MemoryItem
            item = MemoryItem(
                content=content,
                embedding=embedding,
                memory_type=MemoryType(ext.get("memory_type", "FACTUAL")),
                source=source,
                pii_annotations=pii,
                confidence=ext.get("confidence", 0.8),
                entity_mentions=ext.get("entities", []),
            )
            items.append(item)
        return items

Write tests using MockLLM and MockEmbedding:
- Test segmenter with each method
- Test extractor returns correct types
- Test compressor reduces length
- Test full pipeline produces valid MemoryItems
- Test PII is handled in pipeline

ACCEPTANCE CRITERIA:
- Full pipeline works end-to-end
- Each component works independently
- PII detection integrated into pipeline
- Tests pass
```

---

### TASK 15: Evolution Engine

```
Create contextdb/dynamics/evolution/auto_linker.py:

class AutoLinker:
    """When a new memory is added, creates edges across all available graphs."""

    def __init__(self, graphs: dict[str, BaseGraph]):
        self.graphs = graphs

    async def link(self, memory_id: str, memory_data: dict):
        """Add the memory as a node in all graphs, creating edges to existing memories."""
        for graph_name, graph in self.graphs.items():
            await graph.add_node(memory_id, memory_data)


Create contextdb/dynamics/evolution/consolidator.py:

class Consolidator:
    """Periodically consolidates memories into hierarchical summaries."""

    def __init__(self, store: BaseStore, semantic_graph: SemanticGraph, llm: LLMProvider):
        self.store = store
        self.graph = semantic_graph
        self.llm = llm

    async def consolidate(self, min_cluster_size: int = 5, similarity_threshold: float = 0.8):
        """Find clusters of related memories and create summary memories."""
        # 1. Find connected components in semantic graph above threshold
        # 2. For each component with >= min_cluster_size memories:
        #    a. Fetch all memory contents
        #    b. Use LLM to generate a summary
        #    c. Create a new FACTUAL memory with the summary
        #    d. Tag with metadata: {"is_summary": True, "source_ids": [...]}
        # 3. Return list of summary MemoryItems created


Create contextdb/dynamics/evolution/pruner.py:

class Pruner:
    """Removes low-value memories based on heuristic rules."""

    def __init__(self, store: BaseStore):
        self.store = store

    async def prune(self, strategy: str = "decay",
                    max_age_days: int | None = None,
                    min_access_count: int = 0,
                    min_confidence: float = 0.0) -> int:
        """Remove memories matching criteria. Returns count pruned."""
        # strategy "decay": score = confidence * recency_factor * access_factor
        # strategy "age": delete older than max_age_days
        # strategy "unused": delete with access_count <= min_access_count
        # Always soft-delete (set status=DELETED)

Write tests:
- Test AutoLinker creates edges across all graphs
- Test Consolidator creates summary memories
- Test Pruner with each strategy
- Test pruner doesn't delete high-value memories

ACCEPTANCE CRITERIA:
- AutoLinker works with multiple graphs
- Consolidation produces valid summaries
- Pruning correctly identifies low-value memories
- Tests pass
```

---

### TASK 16: Update ContextDB Client with Graph Support

```
Update contextdb/client.py to integrate graphs and formation pipeline:

class ContextDB:
    # Add to __init__:
    self._graphs: dict[str, BaseGraph] = {}
    self._retrieval_engine: RetrievalEngine | None = None
    self._formation: FormationPipeline | None = None
    self._auto_linker: AutoLinker | None = None

    # Update _ensure_init to set up graphs:
    async def _ensure_init(self):
        ...
        # Always create semantic graph (free tier)
        self._graphs["semantic"] = SemanticGraph(self._store)
        await self._graphs["semantic"].initialize()

        # Entity graph (free tier — single graph mode)
        if self.config.enable_entity_graph:  # Add to config
            self._graphs["entity"] = EntityGraph(self._store, self._llm)

        # Temporal + Causal (paid tier check)
        if self.config.enable_multi_graph:  # Paid tier flag
            self._graphs["temporal"] = TemporalGraph(self._store)
            self._graphs["causal"] = CausalGraph(self._store, self._llm)

        self._auto_linker = AutoLinker(self._graphs)
        self._retrieval_engine = RetrievalEngine(
            self._store, self._graphs,
            QueryClassifier(), RetrievalFuser()
        )
        self._formation = FormationPipeline(...)

    # Update add() to use auto-linking:
    async def add(self, content, ...):
        ...
        item = await self._store.add(item)
        # Auto-link in all graphs
        await self._auto_linker.link(item.id, {
            "content": item.content,
            "embedding": item.embedding,
            "event_time": item.event_time,
        })
        return item

    # Update search() to use multi-graph retrieval:
    async def search(self, query, top_k=10, ...):
        ...
        return await self._retrieval_engine.search(query, embeddings[0], top_k)

    # Add new methods:
    async def add_conversation(self, conversation: str, source: str = "") -> list[MemoryItem]:
        """Process a full conversation through the formation pipeline."""
        items = await self._formation.process(conversation, source)
        stored = []
        for item in items:
            stored_item = await self._store.add(item)
            await self._auto_linker.link(stored_item.id, {...})
            stored.append(stored_item)
        return stored

    async def get_timeline(self, entity=None, start=None, end=None) -> list[MemoryItem]:
        """Get chronologically ordered memories."""
        if "temporal" in self._graphs:
            return await self._graphs["temporal"].get_timeline(entity, start, end)
        # Fallback: sort by event_time
        ...

    async def get_entity(self, name: str) -> dict:
        """Get entity profile with all associated memories."""
        if "entity" in self._graphs:
            return await self._graphs["entity"].get_entity(name)
        ...

    async def consolidate(self):
        """Trigger manual consolidation."""
        ...

    async def prune(self, strategy="decay", **kwargs) -> int:
        """Prune low-value memories."""
        ...

Write comprehensive integration tests:
- Test add → auto-link → search flow with semantic graph
- Test add_conversation processes multi-turn input
- Test get_timeline returns chronological order
- Test get_entity returns entity profile
- Test multi-graph search returns better results than single-graph
- Test free tier (single graph) vs paid tier (multi-graph) config flags

ACCEPTANCE CRITERIA:
- Full add → link → search pipeline works
- Graph-enhanced search returns better results than embedding-only
- Free vs paid tier feature gating works
- All tests pass
```

---

## PHASE 3: MEMORY FUNCTIONS (Tasks 17-20)

---

### TASK 17: Factual Memory API

```
Create contextdb/functions/factual.py:

class FactualMemory:
    """High-level API for declarative knowledge management."""

    def __init__(self, db: ContextDB, user_id: str | None = None):
        self.db = db
        self.user_id = user_id

    async def remember(self, content: str, entity: str | None = None,
                       event_time: datetime | None = None) -> MemoryItem:
        """Store a factual memory. Auto-detects and extracts entities."""
        metadata = {}
        if entity:
            metadata["primary_entity"] = entity
        return await self.db.add(content, memory_type=MemoryType.FACTUAL,
                                  metadata=metadata, event_time=event_time)

    async def recall(self, query: str, entity: str | None = None,
                     top_k: int = 5) -> list[MemoryItem]:
        """Recall factual memories. Optionally filter by entity."""
        results = await self.db.search(query, top_k=top_k, memory_type=MemoryType.FACTUAL)
        if entity:
            results = [r for r in results if entity.lower() in
                      " ".join(r.entity_mentions).lower()]
        return results

    async def get_entity(self, name: str) -> dict:
        """Get full entity profile: attributes, related memories, timeline."""
        return await self.db.get_entity(name)

    async def list_entities(self) -> list[Entity]:
        """List all known entities."""
        ...

    async def update_entity(self, name: str, attributes: dict) -> None:
        """Update entity attributes."""
        ...

    async def forget_entity(self, name: str) -> int:
        """Delete all memories about an entity. Returns count deleted."""
        ...

Wire into ContextDB:
    @property
    def factual(self) -> FactualMemory:
        return FactualMemory(self, self.user_id)

Write tests:
- Test remember + recall flow
- Test entity filtering
- Test entity profile retrieval
- Test forget_entity

ACCEPTANCE CRITERIA:
- db.factual.remember("...") and db.factual.recall("...") work
- Entity filtering narrows results correctly
- Tests pass
```

---

### TASK 18: Experiential Memory API

```
Create contextdb/functions/experiential.py:

class ExperientialMemory:
    """High-level API for procedural knowledge: trajectories, workflows, reflections."""

    async def record_trajectory(self, steps: list[dict], outcome: str,
                                 score: float = 0.0) -> MemoryItem:
        """Record an agent trajectory with its outcome.
        steps: [{"action": "...", "observation": "...", "thought": "..."}, ...]"""
        # Format trajectory into structured text
        content = self._format_trajectory(steps, outcome, score)
        return await self.db.add(content, memory_type=MemoryType.EXPERIENTIAL,
                                  metadata={"trajectory_score": score, "outcome": outcome})

    async def record_reflection(self, content: str, context: str = "") -> MemoryItem:
        """Record a verbal reflection (Reflexion-style)."""
        full = f"Reflection: {content}"
        if context:
            full += f"\nContext: {context}"
        return await self.db.add(full, memory_type=MemoryType.EXPERIENTIAL,
                                  metadata={"subtype": "reflection"})

    async def record_workflow(self, name: str, steps: list[str],
                               trigger: str = "") -> MemoryItem:
        """Record a reusable workflow (AWM-style)."""
        content = f"Workflow: {name}\nTrigger: {trigger}\nSteps:\n"
        content += "\n".join(f"  {i+1}. {s}" for i, s in enumerate(steps))
        return await self.db.add(content, memory_type=MemoryType.EXPERIENTIAL,
                                  metadata={"subtype": "workflow", "workflow_name": name})

    async def recall_workflow(self, situation: str, top_k: int = 3) -> list[MemoryItem]:
        """Find relevant workflows for a given situation."""
        return await self.db.search(situation, top_k=top_k, memory_type=MemoryType.EXPERIENTIAL)

    async def get_similar_experiences(self, situation: str, top_k: int = 5) -> list[MemoryItem]:
        """Find past experiences similar to current situation."""
        return await self.db.search(situation, top_k=top_k, memory_type=MemoryType.EXPERIENTIAL)

Wire into ContextDB:
    @property
    def experiential(self) -> ExperientialMemory:
        return ExperientialMemory(self, self.user_id)

Write tests:
- Test trajectory recording and recall
- Test reflection recording
- Test workflow recording and recall
- Test similar experience retrieval

ACCEPTANCE CRITERIA:
- All experiential memory operations work
- Workflows are retrievable by situation description
- Tests pass
```

---

### TASK 19: Working Memory API

```
Create contextdb/functions/working.py:

class WorkingMemory:
    """Active context manager with token-budget awareness and paging."""

    def __init__(self, db: ContextDB, session_id: str, max_tokens: int = 4000):
        self.db = db
        self.session_id = session_id
        self.max_tokens = max_tokens
        self._buffer: list[MemoryItem] = []  # Active context items, ordered by recency
        self._token_count: int = 0

    async def push(self, content: str, metadata: dict | None = None) -> MemoryItem:
        """Add to active context. Auto-pages out oldest items if over budget."""
        item = await self.db.add(content, memory_type=MemoryType.WORKING,
                                  metadata={**(metadata or {}), "session_id": self.session_id})
        self._buffer.append(item)
        self._token_count += self._count_tokens(content)

        # Auto-page out if over budget
        while self._token_count > self.max_tokens and len(self._buffer) > 1:
            oldest = self._buffer.pop(0)
            self._token_count -= self._count_tokens(oldest.content)
            # Move to long-term factual store
            await self.db.update(oldest.id, metadata={**oldest.metadata, "paged_out": True})

        return item

    async def get_context(self, max_tokens: int | None = None) -> str:
        """Get current active context as formatted text."""
        budget = max_tokens or self.max_tokens
        context_parts = []
        tokens_used = 0
        for item in reversed(self._buffer):  # Most recent first
            item_tokens = self._count_tokens(item.content)
            if tokens_used + item_tokens > budget:
                break
            context_parts.append(item.content)
            tokens_used += item_tokens
        return "\n---\n".join(reversed(context_parts))

    async def compress(self, target_tokens: int | None = None) -> str:
        """Compress active context to fit within budget."""
        target = target_tokens or self.max_tokens // 2
        full_context = await self.get_context()
        compressed = await self.db._formation.compressor.compress(full_context,
                                                                    ratio=target/self._token_count)
        # Replace buffer with compressed version
        self._buffer = [MemoryItem(content=compressed, memory_type=MemoryType.WORKING)]
        self._token_count = self._count_tokens(compressed)
        return compressed

    async def clear(self):
        """Clear active context. Optionally save summary to long-term."""
        self._buffer = []
        self._token_count = 0

    def _count_tokens(self, text: str) -> int:
        """Approximate token count. ~4 chars per token."""
        return len(text) // 4

Wire into ContextDB:
    def working(self, session_id: str, max_tokens: int = 4000) -> WorkingMemory:
        return WorkingMemory(self, session_id, max_tokens)

Write tests:
- Test push adds to buffer
- Test auto-paging when over budget
- Test get_context respects token limit
- Test compress reduces context size
- Test clear empties buffer

ACCEPTANCE CRITERIA:
- Working memory manages active context within token budget
- Auto-paging works when buffer exceeds limit
- Compression produces shorter context
- Tests pass
```

---

### TASK 20: Framework Integrations

```
Create contextdb/integrations/langchain.py:

from langchain.memory import BaseMemory (or langchain_core)

class ContextDBLangChainMemory(BaseMemory):
    """Drop-in LangChain memory replacement backed by ContextDB."""

    db: ContextDB
    memory_key: str = "history"
    input_key: str = "input"
    output_key: str = "output"
    return_messages: bool = False

    @property
    def memory_variables(self) -> list[str]:
        return [self.memory_key]

    async def aload_memory_variables(self, inputs: dict) -> dict:
        query = inputs.get(self.input_key, "")
        memories = await self.db.search(query, top_k=5)
        if self.return_messages:
            # Return as message objects
            ...
        context = "\n".join(m.content for m in memories)
        return {self.memory_key: context}

    async def asave_context(self, inputs: dict, outputs: dict) -> None:
        input_text = inputs.get(self.input_key, "")
        output_text = outputs.get(self.output_key, "")
        await self.db.add(f"User: {input_text}\nAssistant: {output_text}")

    async def aclear(self) -> None:
        # Clear working memory for current session
        pass


Create contextdb/integrations/openai_agents.py:

class ContextDBOpenAITool:
    """Exposes ContextDB as an OpenAI function-calling tool."""

    def __init__(self, db: ContextDB):
        self.db = db

    def get_tools(self) -> list[dict]:
        """Return OpenAI-compatible tool definitions."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "remember",
                    "description": "Store a piece of information in long-term memory",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "content": {"type": "string", "description": "The information to remember"},
                            "memory_type": {"type": "string", "enum": ["factual", "experiential"]},
                        },
                        "required": ["content"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "recall",
                    "description": "Search long-term memory for relevant information",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "What to search for"},
                        },
                        "required": ["query"]
                    }
                }
            }
        ]

    async def handle_tool_call(self, name: str, arguments: dict) -> str:
        if name == "remember":
            item = await self.db.add(arguments["content"])
            return f"Remembered: {item.id}"
        elif name == "recall":
            results = await self.db.search(arguments["query"])
            return "\n".join(r.content for r in results)


Create stub integrations for CrewAI and AutoGen (similar pattern).

Add langchain-core, crewai, autogen to optional dep groups.

Write tests:
- Test LangChain memory load/save cycle
- Test OpenAI tool definitions are valid
- Test OpenAI tool call handling

ACCEPTANCE CRITERIA:
- LangChain integration works as drop-in replacement
- OpenAI tools integration works
- Tests pass (with mocks, no API keys)
```

---

## PHASE 4: PRODUCTION FEATURES (Tasks 21-26)

---

### TASK 21: Multi-Agent Memory Bus

```
Create contextdb/multiagent/memory_bus.py:

class MemoryEvent(BaseModel):
    event_type: Literal["MEMORY_CREATED", "MEMORY_UPDATED", "MEMORY_DELETED",
                         "CONFLICT_DETECTED", "WORKFLOW_DISCOVERED"]
    memory_id: str
    agent_id: str
    timestamp: datetime
    payload: dict = {}

class MemoryBus:
    """In-process pub/sub for multi-agent memory coordination."""

    def __init__(self):
        self._subscribers: dict[str, list[Callable]] = {}  # event_type → [callbacks]
        self._history: list[MemoryEvent] = []

    def subscribe(self, agent_id: str, event_types: list[str], callback: Callable):
        for et in event_types:
            self._subscribers.setdefault(et, []).append((agent_id, callback))

    async def publish(self, event: MemoryEvent):
        self._history.append(event)
        for agent_id, callback in self._subscribers.get(event.event_type, []):
            if agent_id != event.agent_id:  # Don't send to originator
                await callback(event)

    def get_history(self, event_type: str | None = None, limit: int = 100) -> list[MemoryEvent]:
        events = self._history
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        return events[-limit:]


Create contextdb/multiagent/role_router.py:

class RoleRouter:
    """Routes memories to relevant agents based on role profiles."""

    def __init__(self, llm: LLMProvider):
        self.llm = llm
        self._roles: dict[str, dict] = {}  # agent_id → role profile

    def register_agent(self, agent_id: str, role: str,
                       topics: list[str], permissions: list[str]):
        self._roles[agent_id] = {
            "role": role, "topics": topics, "permissions": permissions
        }

    async def route(self, memory: MemoryItem) -> list[str]:
        """Determine which agents should receive this memory. Returns list of agent_ids."""
        relevant_agents = []
        for agent_id, profile in self._roles.items():
            # Check topic relevance (keyword matching for MVP)
            for topic in profile["topics"]:
                if topic.lower() in memory.content.lower():
                    relevant_agents.append(agent_id)
                    break
        return relevant_agents


Create contextdb/multiagent/conflict_resolver.py:

class ConflictResolver:
    """Detects and resolves contradicting memories."""

    async def detect_conflicts(self, store: BaseStore, entity_graph: EntityGraph | None = None) -> list[dict]:
        """Find memories that contradict each other about the same entity."""
        # Strategy: for each entity, find memories with opposing sentiment/facts
        # Return list of {"entity": ..., "memory_1": ..., "memory_2": ..., "conflict_type": ...}

    async def resolve(self, conflict: dict, strategy: str = "recent_wins") -> str:
        """Resolve a conflict.
        Strategies: "recent_wins" (newest event_time), "confidence_wins", "merge", "escalate"
        """

Write tests:
- Test pub/sub event flow
- Test role-based routing
- Test conflict detection
- Test conflict resolution strategies

ACCEPTANCE CRITERIA:
- Events flow correctly between agents
- Routing filters by role relevance
- Conflicts are detected and resolvable
- Tests pass
```

---

### TASK 22: Audit Trail

```
Create contextdb/privacy/audit.py:

class AuditEntry(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    operation: Literal["CREATE", "READ", "UPDATE", "DELETE", "SEARCH", "EXPORT", "ERASE"]
    memory_id: str | None = None
    agent_id: str | None = None
    user_id: str | None = None
    details: dict = {}
    prev_hash: str = ""    # Hash chain for tamper detection
    entry_hash: str = ""

class AuditLogger:
    """Append-only, hash-chained audit log."""

    def __init__(self, storage_path: str):
        self._db: aiosqlite connection
        # Table: audit_log(id, timestamp, operation, memory_id, agent_id, user_id, details JSON, prev_hash, entry_hash)

    async def log(self, operation, memory_id=None, agent_id=None, user_id=None, details=None):
        """Log an operation. Computes hash chain."""
        prev = await self._get_last_hash()
        entry = AuditEntry(operation=operation, memory_id=memory_id,
                           agent_id=agent_id, user_id=user_id, details=details or {})
        entry.prev_hash = prev
        entry.entry_hash = self._compute_hash(entry)
        await self._store(entry)

    async def export(self, format: str = "json", since: datetime | None = None) -> str:
        """Export audit log in specified format."""

    async def verify_integrity(self) -> bool:
        """Verify hash chain integrity. Returns True if no tampering detected."""

    def _compute_hash(self, entry: AuditEntry) -> str:
        import hashlib
        data = f"{entry.prev_hash}:{entry.timestamp}:{entry.operation}:{entry.memory_id}"
        return hashlib.sha256(data.encode()).hexdigest()

Integrate into ContextDB:
- Log every add(), search(), update(), delete(), forget() call
- Make audit optional via config flag (default: True)

Write tests:
- Test log creation
- Test hash chain integrity
- Test integrity verification detects tampering
- Test export in JSON and CSV

ACCEPTANCE CRITERIA:
- All operations are logged
- Hash chain is correct
- Tampering is detectable
- Export works
- Tests pass
```

---

### TASK 23: Retention Policy Enforcement

```
Create contextdb/privacy/retention.py:

class RetentionManager:
    """Enforces memory retention policies automatically."""

    def __init__(self, store: BaseStore, audit: AuditLogger, policy: RetentionPolicy):
        self.store = store
        self.audit = audit
        self.policy = policy

    async def enforce(self) -> int:
        """Check all memories against retention policy. Delete expired ones. Returns count."""
        deleted = 0
        now = datetime.utcnow()

        memories = await self.store.list_memories(status=MemoryStatus.ACTIVE, limit=10000)
        for memory in memories:
            ttl = self._get_ttl(memory.memory_type)
            if ttl and (now - memory.created_at) > ttl:
                await self.store.delete(memory.id, hard=True)  # Hard delete for retention
                await self.audit.log("DELETE", memory_id=memory.id,
                                      details={"reason": "retention_policy_expired"})
                deleted += 1
        return deleted

    async def erase_user(self, user_id: str) -> int:
        """Right-to-erasure: permanently delete all memories for a user."""
        memories = await self.store.list_memories(user_id=user_id, limit=100000)
        for memory in memories:
            await self.store.delete(memory.id, hard=True)
            await self.audit.log("ERASE", memory_id=memory.id, user_id=user_id)
        return len(memories)

    def _get_ttl(self, memory_type: MemoryType) -> timedelta | None:
        if memory_type == MemoryType.FACTUAL:
            return self.policy.factual_ttl
        elif memory_type == MemoryType.EXPERIENTIAL:
            return self.policy.experiential_ttl
        elif memory_type == MemoryType.WORKING:
            return self.policy.working_ttl
        return self.policy.default_ttl

Integrate:
- Run retention check on ContextDB startup (async background)
- Add db.privacy.enforce_retention() for manual trigger
- Add db.privacy.erase_user(user_id) for GDPR requests

Write tests:
- Test expired memories are deleted
- Test non-expired memories are kept
- Test per-type TTL works
- Test right-to-erasure deletes all user memories
- Test audit entries created for each deletion

ACCEPTANCE CRITERIA:
- Retention enforcement works correctly
- Right-to-erasure is complete (no orphaned data)
- Audit trail records all deletions
- Tests pass
```

---

### TASK 24: RL Memory Manager (Paid Tier)

```
Create contextdb/rl/memory_manager.py:

class RLMemoryManager:
    """RL-trained agent that decides optimal memory operations."""

    def __init__(self, model_path: str | None = None):
        # Load pre-trained model or initialize new
        self.model = self._load_model(model_path)

    async def decide(self, new_input: str, current_memories: list[MemoryItem],
                     context: dict | None = None) -> dict:
        """Decide what to do with new input.
        Returns: {
            "action": "ADD" | "UPDATE" | "DELETE" | "NOOP",
            "target_memory_id": str | None,  # For UPDATE/DELETE
            "content": str | None,            # For ADD/UPDATE
            "confidence": float,
            "reasoning": str
        }"""

        # Format input for the model
        prompt = self._format_prompt(new_input, current_memories, context)

        # For MVP: Use LLM with structured output as the "RL policy"
        # This can be replaced with an actual RL-trained model later
        response = await self._llm.generate(prompt, response_format=MemoryDecision)
        return response

    def _format_prompt(self, new_input, memories, context):
        return f'''You are a Memory Manager. Decide the optimal memory operation.

Current memories ({len(memories)} total):
{self._format_memories(memories)}

New input: "{new_input}"

Decide:
- ADD: Store as new memory (the input contains novel, useful information)
- UPDATE(memory_id): Merge with existing memory (updates/contradicts existing)
- DELETE(memory_id): Remove an existing memory (now outdated/wrong)
- NOOP: Ignore (input is noise, duplicate, or transient)

Return JSON: {{"action": "...", "target_memory_id": "...", "content": "...",
               "confidence": 0.0-1.0, "reasoning": "brief explanation"}}'''


Create contextdb/rl/training.py:

class RLTrainingPipeline:
    """Training pipeline for the Memory Manager."""

    async def collect_data(self, db: ContextDB, conversations: list[str],
                           qa_pairs: list[dict]) -> list[dict]:
        """Collect training data by running memory operations and evaluating downstream QA."""

    async def train(self, data: list[dict], algorithm: str = "ppo",
                    epochs: int = 3, lr: float = 1e-5) -> dict:
        """Train the memory manager. Returns training metrics."""
        # For MVP: Fine-tune a small LLM on the collected data
        # Full RL (PPO/GRPO) in future version

    async def evaluate(self, db: ContextDB, test_data: list[dict]) -> dict:
        """Evaluate memory manager on held-out data. Returns accuracy metrics."""

Wire into ContextDB:
- If config.enable_rl_manager (paid tier flag):
  - Route all add() calls through RLMemoryManager.decide() first
  - Manager can override: NOOP skips storage, UPDATE modifies existing, etc.

Write tests:
- Test decide() returns valid actions
- Test NOOP correctly skips storage
- Test UPDATE correctly modifies existing memory
- Test training data collection
- Test evaluation metrics

ACCEPTANCE CRITERIA:
- RL manager produces reasonable decisions (with LLM backend)
- Feature-gated behind paid tier flag
- Training pipeline scaffolded
- Tests pass
```

---

### TASK 25: Migration Tools

```
Create contextdb/utils/migrations.py:

class Mem0Importer:
    """Import memories from Mem0 export."""

    async def import_from_json(self, json_path: str, db: ContextDB) -> int:
        """Import Mem0 JSON export into ContextDB. Returns count imported."""
        # Mem0 format: [{"id": ..., "memory": ..., "user_id": ..., "metadata": {...}, "created_at": ...}]
        # Map to ContextDB MemoryItems
        # Re-embed content with ContextDB's embedding model

class ZepImporter:
    """Import from Zep export."""

    async def import_from_json(self, json_path: str, db: ContextDB) -> int:
        # Zep format: episodes with temporal edges
        # Preserve temporal information in event_time

class LangChainImporter:
    """Import from LangChain ConversationBufferMemory or similar."""

    async def import_from_messages(self, messages: list[dict], db: ContextDB) -> int:
        # messages: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
        # Process through formation pipeline

class JSONExporter:
    """Export ContextDB memories to portable JSON."""

    async def export(self, db: ContextDB, output_path: str,
                     include_embeddings: bool = False) -> int:
        """Export all memories to JSON. Returns count exported."""

Write tests with sample data fixtures.

ACCEPTANCE CRITERIA:
- Import from Mem0 JSON works
- Import from LangChain messages works
- Export to JSON works
- Round-trip (export → import) preserves data
- Tests pass
```

---

### TASK 26: CLI Tool

```
Create contextdb/cli.py using click or typer:

Commands:
  contextdb init                    # Initialize a new ContextDB in current directory
  contextdb add "content here"      # Add a memory from CLI
  contextdb search "query"          # Search memories
  contextdb stats                   # Show memory statistics
  contextdb export output.json      # Export all memories
  contextdb import input.json       # Import memories
  contextdb migrate mem0 data.json  # Migrate from Mem0
  contextdb prune --strategy decay  # Prune low-value memories
  contextdb audit --since 2024-01-01  # View audit log
  contextdb erase --user user_123   # Right-to-erasure

Add [tool.poetry.scripts] or [project.scripts] to pyproject.toml:
  contextdb = "contextdb.cli:app"

Write tests:
- Test each CLI command
- Use click.testing.CliRunner

ACCEPTANCE CRITERIA:
- `contextdb --help` shows all commands
- Each command works end-to-end
- Tests pass
```

---

## PHASE 5: EXAMPLES & DOCUMENTATION (Tasks 27-32)

---

### TASK 27: Example — Customer Support Agent

```
Create examples/customer_support_agent.py:

A complete working example showing:
1. Initialize ContextDB with PII redaction
2. Process incoming customer conversation through formation pipeline
3. Recall customer history when new message arrives
4. Record resolution workflows from successful interactions
5. Multi-agent: support agent + escalation agent sharing memory

Use OpenAI's chat completions API for the agent logic.
Include inline comments explaining each ContextDB feature used.
Make it runnable with: OPENAI_API_KEY=... python examples/customer_support_agent.py

Should demonstrate:
- db.factual.remember() for customer profile
- db.experiential.record_workflow() for resolution patterns
- db.working.push() for active conversation
- db.search() for history recall
- PII redaction in action
```

---

### TASK 28: Example — Research Assistant

```
Create examples/research_assistant.py:

A complete working example showing:
1. Feed multiple documents/papers into ContextDB
2. Build cross-document knowledge graph
3. Ask questions that require multi-document reasoning
4. Track reading trajectory and surface connections

Demonstrate:
- db.add_conversation() for processing documents
- db.get_entity() for concept profiles
- db.get_timeline() for research chronology
- Graph-enhanced retrieval finding cross-document connections
```

---

### TASK 29: Example — AI Phone Agent

```
Create examples/phone_agent.py:

A complete working example simulating an AI phone agent:
1. Pre-load caller history during "ring" interval
2. Real-time working memory during call
3. Post-call memory formation (extract facts, experiences)
4. Cross-call learning (workflow induction)

Demonstrate:
- db.working() for real-time context
- db.factual.recall() for caller history
- db.experiential.record_trajectory() for call outcomes
- Temporal queries ("when was last service visit?")
```

---

### TASK 30: Example — Multi-Agent Team

```
Create examples/multi_agent_team.py:

A complete working example showing 3 agents sharing memory:
1. Research Agent: discovers information, stores facts
2. Writing Agent: reads research, produces content
3. Review Agent: checks content against facts, flags inconsistencies

Demonstrate:
- MemoryBus event flow
- Role-aware routing
- Conflict detection and resolution
- Shared vs private memory scopes
```

---

### TASK 31: Documentation

```
Create comprehensive MkDocs documentation in docs/:

docs/
├── index.md              # Landing page with value prop + quickstart
├── quickstart.md         # 5-minute getting started guide
├── concepts.md           # Memory forms, functions, dynamics explained simply
├── architecture.md       # System architecture with diagrams
├── api-reference/
│   ├── core.md          # ContextDB, init(), config
│   ├── factual.md       # Factual memory API
│   ├── experiential.md  # Experiential memory API
│   ├── working.md       # Working memory API
│   ├── graph.md         # Graph operations
│   ├── multiagent.md    # Multi-agent APIs
│   └── privacy.md       # Privacy, PII, retention, audit
├── guides/
│   ├── customer-support.md    # Building a support agent
│   ├── research-assistant.md  # Building a research assistant
│   ├── phone-agent.md         # Building a phone agent
│   └── migration.md           # Migrating from Mem0, Zep, LangChain
├── advanced/
│   ├── rl-training.md   # Training the RL memory manager
│   ├── custom-graphs.md # Creating custom graph types
│   └── deployment.md    # Production deployment guide
└── changelog.md

Configure mkdocs.yml with Material theme.
Add `mkdocs build` to CI.

ACCEPTANCE CRITERIA:
- `mkdocs serve` runs and renders all pages
- Every public API method is documented
- Quickstart is copy-paste runnable
- Architecture diagrams included
```

---

### TASK 32: README + PyPI Release

```
Polish the README.md:

# ContextDB

> The unified context layer for AI agents.
> Replace your patchwork of Pinecone + Redis + Postgres + glue code with one system that understands memory.

[Badges: PyPI version, Python 3.10+, License Apache 2.0, Tests passing, Docs]

## The Problem

Every team building AI agents assembles the same fragile stack: a vector DB for embeddings, Redis for sessions, Postgres for profiles, and custom code to link them. It takes months, breaks at every seam, and never learns.

Databricks Lakebase gives your agent a hard drive. **ContextDB gives your agent a brain.**

## Quickstart

```bash
pip install contextdb
```

```python
import contextdb

db = contextdb.init(user_id="user_123")
db.add("Alex prefers morning calls and uses a Carrier AC unit")
result = db.search("What kind of AC does Alex have?")
# → "Alex uses a Carrier AC unit"
```

## What ContextDB Replaces

| Your current patchwork | ContextDB replacement |
|---|---|
| Pinecone / Qdrant (vectors only) | Multi-graph store (semantic + temporal + causal + entity) |
| Redis (session state, ephemeral) | Working Memory with paging + compression |
| PostgreSQL (static user profiles) | Factual Memory with entity graph + bitemporality |
| S3 / flat files (conversation logs) | Formation pipeline: segment → extract → compress |
| Custom glue code (brittle) | Dynamics Engine: auto-linking, consolidation, retrieval |
| *(nothing — no learning)* | Experiential Memory + RL Memory Manager |
| *(nothing — no privacy)* | Privacy Layer: PII detection, retention, audit |

## Features

[Feature comparison table vs Mem0, Zep, MemGPT, Databricks Lakebase]

## Architecture

[ASCII diagram or link to docs]

## Examples

[Links to examples/]

## Documentation

[Link to hosted docs]

## Contributing

[Standard contributing section]

## License

Apache 2.0


Prepare for PyPI release:
1. Ensure pyproject.toml has all metadata (description, URLs, classifiers)
2. Build: `python -m build`
3. Test: `twine check dist/*`
4. Publish to TestPyPI first, then PyPI

ACCEPTANCE CRITERIA:
- README renders correctly on GitHub
- `pip install contextdb` installs from PyPI
- Package imports correctly after install
- All 31 previous tasks' tests still pass
```

---

## APPENDIX: Key Architecture Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Default storage | SQLite | Zero-config, fast for <1M items, single file |
| Embedding default | text-embedding-3-small | Good balance of quality/cost/speed |
| Graph storage | Same DB as memories | Avoids separate graph DB dependency for free tier |
| PII detection | Regex + NER | No external service dependency |
| Multi-agent bus | In-process pub/sub | Simple, no infra deps; Redis for production |
| RL manager | LLM-as-policy (MVP) | Ships faster; real RL training in v2 |
| Async-first | aiosqlite, async/await everywhere | AI apps are I/O bound |
| Feature gating | Config flags | Clean separation, easy to test both tiers |
